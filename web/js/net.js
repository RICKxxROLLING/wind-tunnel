// WebSocket link to the solver server. Every outgoing message goes through one promise chain so
// a gzip-compressed geometry upload can never be overtaken by a later command.

async function gzip(bytes) {
  const stream = new Blob([bytes]).stream().pipeThrough(new CompressionStream('gzip'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

function frameBytes(header, payload) {
  const hb = new TextEncoder().encode(JSON.stringify(header));
  const out = new Uint8Array(4 + hb.length + payload.length);
  new DataView(out.buffer).setUint32(0, hb.length, true);
  out.set(hb, 4);
  out.set(payload, 4 + hb.length);
  return out;
}

export function decodeFrame(buf) {
  const hl = new DataView(buf).getUint32(0, true);
  const h = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, hl)));
  const base = 4 + hl;
  const o = h.offsets;
  const d = { header: h };
  if ('parts16' in o) d.parts16 = new Uint16Array(buf, base + o.parts16, h.numP * 4);
  if ('parts8' in o) d.parts8 = new Uint8Array(buf, base + o.parts8, h.numP);
  if ('slice' in o) d.slice = new Uint8Array(buf, base + o.slice, h.sliceW * h.sliceH * 2);
  if ('cp' in o) d.cp = new Int8Array(buf, base + o.cp, h.numVerts);
  return d;
}

export class Remote {
  constructor(handlers) {
    this.h = handlers;
    this.params = {};
    this.dirty = false;
    this.chain = Promise.resolve();
    this.connect();
  }

  get open() { return this.ws && this.ws.readyState === WebSocket.OPEN; }

  connect() {
    const url = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`;
    const ws = this.ws = new WebSocket(url);
    ws.binaryType = 'arraybuffer';
    ws.onopen = () => this.h.onOpen?.();
    ws.onclose = () => {
      this.h.onClose?.();
      setTimeout(() => this.connect(), 1500);
    };
    ws.onmessage = (e) => {
      if (typeof e.data === 'string') this.h.onJson?.(JSON.parse(e.data));
      else this.h.onFrame?.(decodeFrame(e.data));
    };
  }

  _enqueue(fn) {
    this.chain = this.chain.then(fn).catch((e) => console.warn('send failed', e));
  }

  _flushParams() {
    if (!this.dirty) return;
    this.dirty = false;
    const msg = JSON.stringify({ type: 'params', ...this.params });
    this._enqueue(() => { if (this.open) this.ws.send(msg); });
  }

  setParams(o) {
    Object.assign(this.params, o);
    if (!this.dirty) {
      this.dirty = true;
      queueMicrotask(() => this._flushParams());
    }
  }

  send(obj) {
    this._flushParams();
    const msg = JSON.stringify(obj);
    this._enqueue(() => { if (this.open) this.ws.send(msg); });
  }

  // ready acks bypass the queue: they are pure flow control
  ready() { if (this.open) this.ws.send('{"type":"ready"}'); }

  sendBinary(header, bytes) {
    this._flushParams();
    const p = gzip(bytes);
    this._enqueue(async () => {
      const data = frameBytes(header, await p);
      if (this.open) this.ws.send(data);
    });
  }
}
