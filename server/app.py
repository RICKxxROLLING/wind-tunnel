"""Wind tunnel server: runs the Warp LBM solver on the GPU and streams flow data to browser viewers."""
import asyncio
import gzip
import json
import os
import queue
import struct
import threading
import time

import numpy as np
import warp as wp
from aiohttp import WSMsgType, web

from lbm import BYTES_PER_CELL, RESOLUTIONS, Solver

WEB_DIR = os.environ.get('WEB_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'web'))
PORT = int(os.environ.get('PORT', '8765'))
FRAME_HZ = 60
PAUSED_HZ = 12


def pack(header, parts):
    """[u32 header length][JSON header, space-padded so the payload is 8-byte aligned][payload]."""
    offsets, pos = {}, 0
    for name, arr in parts:
        offsets[name] = pos
        pos += arr.nbytes
        pos += (-pos) % 8
    header = dict(header, offsets=offsets)
    hb = json.dumps(header).encode()
    hb += b' ' * ((-(4 + len(hb))) % 8)
    out = bytearray(4 + len(hb) + pos)
    struct.pack_into('<I', out, 0, len(hb))
    out[4:4 + len(hb)] = hb
    base = 4 + len(hb)
    for name, arr in parts:
        b = arr.tobytes()
        o = base + offsets[name]
        out[o:o + len(b)] = b
    return bytes(out)


class Client:
    def __init__(self, ws):
        self.ws = ws
        self.ready = True


class Engine(threading.Thread):
    def __init__(self, loop):
        super().__init__(daemon=True)
        self.loop = loop
        self.cmds = queue.Queue()
        self.clients = set()
        self.device = wp.get_preferred_device()
        self.is_cuda = self.device.is_cuda
        self.gpu_name = self.device.name if self.is_cuda else 'CPU (no CUDA device found)'
        self.mem = self.device.total_memory if self.is_cuda else 2 * 1024 ** 3
        self.solver = None
        self.res = None
        self.p = dict(uIn=0.1, tau=0.51, smag=0.16, steps=10, maxAge=1000.0, lref=40.0,
                      seedX=2.0, seedCy=20.0, seedCz=20.0, seedHy=10.0, seedHz=10.0,
                      slice=None, cp=False, running=True, particles=5000)
        self.steps_since = 0
        self.perf_steps = 0
        self.perf_t = time.perf_counter()
        self.mlups = 0.0
        self.frame_no = 0

    # ---- info ----
    def fits(self, dims):
        n = dims[0] * dims[1] * dims[2]
        if not self.is_cuda and n > 600_000:
            return False
        return n * BYTES_PER_CELL < self.mem * 0.85

    def hello(self):
        res = [{'name': k, 'dims': v, 'fits': self.fits(v)} for k, v in RESOLUTIONS.items()]
        default = 'high' if self.is_cuda and self.fits(RESOLUTIONS['high']) else 'test'
        return {'type': 'hello', 'gpu': self.gpu_name, 'cuda': self.is_cuda, 'memory': self.mem,
                'resolutions': res, 'default': default, 'res': self.res}

    def send_json(self, obj, clients=None):
        targets = list(clients or self.clients)
        async def go():
            for c in targets:
                try:
                    await c.ws.send_json(obj)
                except Exception:
                    pass
        asyncio.run_coroutine_threadsafe(go(), self.loop)

    def send_frame(self, data, targets):
        async def go():
            for c in targets:
                try:
                    await c.ws.send_bytes(data)
                except Exception:
                    pass
        asyncio.run_coroutine_threadsafe(go(), self.loop)

    # ---- commands (all GPU work stays on this thread) ----
    def handle(self, kind, msg, payload, client):
        t = msg.get('type')
        if t == 'grid':
            res = msg.get('res')
            dims = RESOLUTIONS.get(res)
            if not dims or not self.fits(dims):
                self.send_json({'type': 'error', 'message': f'Resolution {res} does not fit on this device'}, [client])
                return
            if res != self.res or self.solver is None:
                self.solver = None
                wp.synchronize()
                try:
                    self.solver = Solver(dims, self.device)
                    self.solver.reset(self.p['uIn'])
                    self.res = res
                except Exception as e:  # out of memory etc.
                    self.solver, self.res = None, None
                    self.send_json({'type': 'error', 'message': f'Could not allocate {res} grid: {e}'})
                    return
            self.send_json({'type': 'grid', 'res': self.res, 'dims': RESOLUTIONS[self.res]})
            return
        if self.solver is None:
            return
        s = self.solver
        if t == 'params':
            old = self.p.get('particles')
            self.p.update({k: v for k, v in msg.items() if k != 'type'})
            if self.p['particles'] != old and s.num_p:
                s.set_particles(int(self.p['particles']), self.p)
        elif t == 'respawn':
            s.set_particles(int(self.p['particles']), self.p)
        elif t == 'reset':
            s.reset(self.p['uIn'])
            self.steps_since = 0
        elif t == 'solid':
            if tuple(msg.get('dims', ())) != (s.nx, s.ny, s.nz):
                return
            arr = np.frombuffer(gzip.decompress(payload), dtype=np.uint8)
            if arr.size == s.n:
                s.set_solid(arr)
        elif t == 'surface':
            raw = np.frombuffer(gzip.decompress(payload), dtype=np.float32)
            nv = int(msg['numVerts'])
            if raw.size == nv * 6:
                s.set_surface(raw[:nv * 3].reshape(-1, 3), raw[nv * 3:].reshape(-1, 3))

    def run(self):
        wp.set_device(self.device)
        last_frame = 0.0
        while True:
            t0 = time.perf_counter()
            try:
                while True:
                    self.handle(*self.cmds.get_nowait())
            except queue.Empty:
                pass
            except Exception as e:
                print('command failed:', repr(e), flush=True)
            s = self.solver
            if s is None or not self.clients:
                time.sleep(0.05)
                continue
            running = bool(self.p.get('running'))
            steps = int(self.p.get('steps', 10)) if running else 0
            try:
                if steps:
                    s.step(steps, self.p)
                    self.steps_since += steps
                    self.perf_steps += steps
                ready = [c for c in self.clients if c.ready]
                now = time.perf_counter()
                if ready and (running or now - last_frame > 1 / PAUSED_HZ):
                    self.frame_no += 1
                    every = max(3, s.num_verts // 20000)
                    out = s.frame(self.p, bool(self.p.get('cp')) and self.frame_no % every == 0)
                    if now - self.perf_t > 1.0:
                        self.mlups = self.perf_steps * s.n / 1e6 / (now - self.perf_t)
                        self.perf_steps, self.perf_t = 0, now
                    header = {'type': 'frame', 'dims': [s.nx, s.ny, s.nz], 'numP': s.num_p, 'numVerts': s.num_verts,
                              'forces': out['forces'], 'steps': self.steps_since, 'mlups': self.mlups if running else 0,
                              'running': running}
                    parts = []
                    for k in ('parts16', 'parts8', 'slice', 'cp'):
                        if k in out:
                            parts.append((k, out[k]))
                    if 'slice' in out:
                        header.update(sliceW=out['sliceW'], sliceH=out['sliceH'], sliceAxis=out['sliceAxis'])
                    self.steps_since = 0
                    for c in ready:
                        c.ready = False
                    self.send_frame(pack(header, parts), ready)
                    last_frame = now
            except Exception as e:
                print('solver error:', repr(e), flush=True)
                time.sleep(0.5)
            # pace to ~FRAME_HZ iterations per second
            spare = 1 / (FRAME_HZ if running else PAUSED_HZ) - (time.perf_counter() - t0)
            if spare > 0:
                time.sleep(spare)


async def ws_handler(request):
    engine = request.app['engine']
    ws = web.WebSocketResponse(max_msg_size=1024 ** 3, heartbeat=20)
    await ws.prepare(request)
    client = Client(ws)
    engine.clients.add(client)
    await ws.send_json(engine.hello())
    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                data = json.loads(msg.data)
                if data.get('type') == 'ready':
                    client.ready = True
                else:
                    engine.cmds.put(('json', data, None, client))
            elif msg.type == WSMsgType.BINARY:
                buf = msg.data
                hl = struct.unpack_from('<I', buf, 0)[0]
                header = json.loads(buf[4:4 + hl])
                engine.cmds.put(('bin', header, buf[4 + hl:], client))
    finally:
        engine.clients.discard(client)
    return ws


async def index(request):
    return web.FileResponse(os.path.join(WEB_DIR, 'index.html'), headers={'Cache-Control': 'no-cache'})


@web.middleware
async def no_cache(request, handler):
    resp = await handler(request)
    resp.headers.setdefault('Cache-Control', 'no-cache')
    return resp


async def on_startup(app):
    engine = Engine(asyncio.get_running_loop())
    app['engine'] = engine
    engine.start()
    print(f'Wind tunnel solver on {engine.gpu_name}; open http://<this-host>:{PORT}', flush=True)


def main():
    wp.init()
    app = web.Application(middlewares=[no_cache])
    app.on_startup.append(on_startup)
    app.router.add_get('/ws', ws_handler)
    app.router.add_get('/', index)
    app.router.add_static('/', WEB_DIR)
    web.run_app(app, port=PORT, print=None)


if __name__ == '__main__':
    main()
