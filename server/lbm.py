"""GPU lattice-Boltzmann wind tunnel (D3Q19, regularized collision + Smagorinsky LES) on NVIDIA Warp.

Lattice frame: x = streamwise (flow +x), y = vertical, z = span. Cell centres at integer coordinates.
The hot kernels (stream/collide, init, forces) are generated as fully unrolled source and imported,
so every population lives in a register.
"""
import hashlib
import importlib.util
import os
import tempfile

import numpy as np
import warp as wp

C = [(0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
     (1, 1, 0), (-1, -1, 0), (1, -1, 0), (-1, 1, 0), (1, 0, 1), (-1, 0, -1), (1, 0, -1), (-1, 0, 1),
     (0, 1, 1), (0, -1, -1), (0, 1, -1), (0, -1, 1)]
W = [1 / 3 if i == 0 else (1 / 18 if sum(map(abs, c)) == 1 else 1 / 36) for i, c in enumerate(C)]
OPP = [C.index(tuple(-v for v in c)) for c in C]

RESOLUTIONS = {
    'test': (64, 28, 36),
    'low': (128, 56, 72),
    'medium': (176, 80, 96),
    'high': (256, 112, 136),
    'ultra': (352, 144, 176),
    'extreme': (448, 192, 224),
    'max': (544, 224, 272),
}
BYTES_PER_CELL = 19 * 4 * 2 + 16 + 1


def F(v):
    return repr(float(v))


def lin(g, name='f'):
    s = '0.0'
    for i, c in enumerate(C):
        k = g(c)
        if k > 0:
            s += f' + {name}{i}'
        elif k < 0:
            s += f' - {name}{i}'
    return s


def cu_expr(c, ux, uy, uz):
    s = '0.0'
    for d, v in zip(c, (ux, uy, uz)):
        if d:
            s += f' + {v}' if d > 0 else f' - {v}'
    return s


def off(base, d):
    return base if d == 0 else (f'{base} + {d}' if d > 0 else f'{base} - {-d}')


def gen_source():
    L = ['import warp as wp', '', '']

    # ---- init: equilibrium at rho = 1, u = (uin, 0, 0) ----
    L += ['@wp.kernel',
          'def init_kernel(fout: wp.array(dtype=wp.float32), mac: wp.array(dtype=wp.vec4), n: int, uin: float):',
          '    idx = wp.tid()']
    for i, c in enumerate(C):
        L.append(f'    fout[{i} * n + idx] = {F(W[i])} * (1.0 + {F(3 * c[0])} * uin + {F(4.5 * c[0] ** 2)} * uin * uin - 1.5 * uin * uin)')
    L += ['    mac[idx] = wp.vec4(uin, 0.0, 0.0, 1.0)', '', '']

    # ---- stream (pull) + collide ----
    L += ['@wp.kernel',
          'def step_kernel(fin: wp.array(dtype=wp.float32), fout: wp.array(dtype=wp.float32),',
          '                solid: wp.array(dtype=wp.uint8), mac: wp.array(dtype=wp.vec4),',
          '                nx: int, ny: int, nz: int, uin: float, tau: float, smag: float):',
          '    idx = wp.tid()',
          '    n = nx * ny * nz',
          '    x = idx % nx',
          '    y = (idx // nx) % ny',
          '    z = idx // (nx * ny)',
          '    if int(solid[idx]) != 0:',
          '        mac[idx] = wp.vec4(0.0, 0.0, 0.0, 1.0)',
          '        return',
          '    if x == 0 or x == nx - 1 or y == 0 or y == ny - 1 or z == 0 or z == nz - 1:',
          '        # equilibrium boundaries from the nearest interior cell: fixed inlet,',
          '        # pressure outlet (rho = 1), slip walls on the test-section sides',
          '        m = mac[wp.clamp(x, 1, nx - 2) + nx * (wp.clamp(y, 1, ny - 2) + ny * wp.clamp(z, 1, nz - 2))]',
          '        br = wp.clamp(m[3], 0.9, 1.1)',
          '        bux = m[0]',
          '        buy = m[1]',
          '        buz = m[2]',
          '        if y == 0 or y == ny - 1:',
          '            buy = 0.0',
          '        if z == 0 or z == nz - 1:',
          '            buz = 0.0',
          '        if x == 0:',
          '            br = 1.0',
          '            bux = uin',
          '            buy = 0.0',
          '            buz = 0.0',
          '        elif x == nx - 1:',
          '            br = 1.0',
          '            bux = wp.max(bux, 0.0)',
          '        bsq = 1.5 * (bux * bux + buy * buy + buz * buz)']
    for i, c in enumerate(C):
        L.append(f'        bc{i} = 3.0 * ({cu_expr(c, "bux", "buy", "buz")})')
        L.append(f'        fout[{i} * n + idx] = {F(W[i])} * br * (1.0 + bc{i} + 0.5 * bc{i} * bc{i} - bsq)')
    L += ['        mac[idx] = wp.vec4(bux, buy, buz, br)',
          '        return',
          '    f0 = fin[idx]']
    for i, (cx, cy, cz) in enumerate(C):
        if i == 0:
            continue
        L += [f'    s{i} = {off("x", -cx)} + nx * ({off("y", -cy)} + ny * ({off("z", -cz)}))',
              f'    t{i} = int(solid[s{i}])',
              f'    f{i} = float(0.0)',
              f'    if t{i} == 0:',
              f'        f{i} = fin[{i} * n + s{i}]',
              '    else:',
              f'        f{i} = fin[{OPP[i]} * n + idx]']
        if cx:
            # moving ground belt at U: bounce-back + 6 w_i rho (c_i . u_w)
            L += [f'        if t{i} == 2:',
                  f'            f{i} = f{i} + {F(6 * W[i] * cx)} * uin']
    L += [f'    rho = {lin(lambda c: 1)}',
          '    ir = 1.0 / rho',
          f'    ux = ({lin(lambda c: c[0])}) * ir',
          f'    uy = ({lin(lambda c: c[1])}) * ir',
          f'    uz = ({lin(lambda c: c[2])}) * ir',
          '    um = wp.sqrt(ux * ux + uy * uy + uz * uz)',
          '    if um > 0.35:',
          '        sc = 0.35 / um',
          '        ux = ux * sc',
          '        uy = uy * sc',
          '        uz = uz * sc',
          '    usq = 1.5 * (ux * ux + uy * uy + uz * uz)']
    for i, c in enumerate(C):
        L += [f'    cu{i} = 3.0 * ({cu_expr(c, "ux", "uy", "uz")})',
              f'    e{i} = {F(W[i])} * rho * (1.0 + cu{i} + 0.5 * cu{i} * cu{i} - usq)',
              f'    d{i} = f{i} - e{i}']
    L += [f'    pxx = {lin(lambda c: c[0] * c[0], "d")}',
          f'    pyy = {lin(lambda c: c[1] * c[1], "d")}',
          f'    pzz = {lin(lambda c: c[2] * c[2], "d")}',
          f'    pxy = {lin(lambda c: c[0] * c[1], "d")}',
          f'    pxz = {lin(lambda c: c[0] * c[2], "d")}',
          f'    pyz = {lin(lambda c: c[1] * c[2], "d")}',
          '    qn = wp.sqrt(pxx * pxx + pyy * pyy + pzz * pzz + 2.0 * (pxy * pxy + pxz * pxz + pyz * pyz))',
          '    tau0 = tau',
          '    sp = float(x) - float(nx) + 24.0',
          '    if sp > 0.0:',
          '        tau0 = tau0 + sp * 0.015',
          '    tau_e = 0.5 * (tau0 + wp.sqrt(tau0 * tau0 + 25.4558441 * smag * smag * qn * ir))',
          '    keep = 1.0 - 1.0 / tau_e',
          '    tr3 = (pxx + pyy + pzz) * (1.0 / 3.0)']
    # regularized collision: keep only the 2nd-order (Hermite) part of f_neq
    for i, (cx, cy, cz) in enumerate(C):
        t = []
        if cx: t.append('pxx')
        if cy: t.append('pyy')
        if cz: t.append('pzz')
        if cx and cy: t.append(f'{F(2 * cx * cy)} * pxy')
        if cx and cz: t.append(f'{F(2 * cx * cz)} * pxz')
        if cy and cz: t.append(f'{F(2 * cy * cz)} * pyz')
        qp = (' + '.join(t) if t else '0.0') + ' - tr3'
        L.append(f'    fout[{i} * n + idx] = e{i} + {F(4.5 * W[i])} * keep * ({qp})')
    L += ['    mac[idx] = wp.vec4(ux, uy, uz, rho)', '', '']

    # ---- momentum-exchange force on static solids ----
    L += ['@wp.kernel',
          'def force_kernel(fcur: wp.array(dtype=wp.float32), solid: wp.array(dtype=wp.uint8),',
          '                 forces: wp.array(dtype=wp.float32), nx: int, ny: int, nz: int):',
          '    idx = wp.tid()',
          '    n = nx * ny * nz',
          '    x = idx % nx',
          '    y = (idx // nx) % ny',
          '    z = idx // (nx * ny)',
          '    if x == 0 or y == 0 or z == 0 or x >= nx - 1 or y >= ny - 1 or z >= nz - 1:',
          '        return',
          '    if int(solid[idx]) != 0:',
          '        return',
          '    fx = float(0.0)',
          '    fy = float(0.0)',
          '    fz = float(0.0)',
          '    hit = int(0)']
    for i, (cx, cy, cz) in enumerate(C):
        if i == 0:
            continue
        L += [f'    if int(solid[{off("x", cx)} + nx * ({off("y", cy)} + ny * ({off("z", cz)}))]) == 1:',
              f'        v{i} = 2.0 * (fcur[{i} * n + idx] - {F(W[i])})']
        for comp, d in (('fx', cx), ('fy', cy), ('fz', cz)):
            if d:
                L.append(f'        {comp} = {comp} {"+" if d > 0 else "-"} v{i}')
        L.append('        hit = 1')
    L += ['    if hit == 1:',
          '        wp.atomic_add(forces, 0, fx)',
          '        wp.atomic_add(forces, 1, fy)',
          '        wp.atomic_add(forces, 2, fz)', '']
    return '\n'.join(L) + '\n'


@wp.func
def cidx(x: int, y: int, z: int, nx: int, ny: int, nz: int):
    return wp.clamp(x, 0, nx - 1) + nx * (wp.clamp(y, 0, ny - 1) + ny * wp.clamp(z, 0, nz - 1))


@wp.func
def sample_mac(mac: wp.array(dtype=wp.vec4), p: wp.vec3, nx: int, ny: int, nz: int):
    qx = wp.clamp(p[0], 0.0, float(nx) - 1.001)
    qy = wp.clamp(p[1], 0.0, float(ny) - 1.001)
    qz = wp.clamp(p[2], 0.0, float(nz) - 1.001)
    bx = wp.floor(qx)
    by = wp.floor(qy)
    bz = wp.floor(qz)
    tx = qx - bx
    ty = qy - by
    tz = qz - bz
    ix = int(bx)
    iy = int(by)
    iz = int(bz)
    c000 = mac[cidx(ix, iy, iz, nx, ny, nz)]
    c100 = mac[cidx(ix + 1, iy, iz, nx, ny, nz)]
    c010 = mac[cidx(ix, iy + 1, iz, nx, ny, nz)]
    c110 = mac[cidx(ix + 1, iy + 1, iz, nx, ny, nz)]
    c001 = mac[cidx(ix, iy, iz + 1, nx, ny, nz)]
    c101 = mac[cidx(ix + 1, iy, iz + 1, nx, ny, nz)]
    c011 = mac[cidx(ix, iy + 1, iz + 1, nx, ny, nz)]
    c111 = mac[cidx(ix + 1, iy + 1, iz + 1, nx, ny, nz)]
    a = wp.lerp(wp.lerp(c000, c100, tx), wp.lerp(c010, c110, tx), ty)
    b = wp.lerp(wp.lerp(c001, c101, tx), wp.lerp(c011, c111, tx), ty)
    return wp.lerp(a, b, tz)


@wp.kernel
def advect_kernel(pos: wp.array(dtype=wp.vec4), spd: wp.array(dtype=wp.float32),
                  mac: wp.array(dtype=wp.vec4), solid: wp.array(dtype=wp.uint8),
                  nx: int, ny: int, nz: int, dt: float, uin: float, max_age: float, seed: int,
                  seed_x: float, seed_cy: float, seed_cz: float, seed_hy: float, seed_hz: float):
    i = wp.tid()
    p = pos[i]
    q = wp.vec3(p[0], p[1], p[2])
    m1 = sample_mac(mac, q, nx, ny, nz)
    u1 = wp.vec3(m1[0], m1[1], m1[2])
    m2 = sample_mac(mac, q + 0.5 * dt * u1, nx, ny, nz)
    u2 = wp.vec3(m2[0], m2[1], m2[2])
    q = q + dt * u2
    age = p[3] + 1.0
    s = wp.length(u2) / uin
    dead = age > max_age or q[0] < 0.5 or q[1] < 0.5 or q[2] < 0.5
    if q[0] > float(nx) - 3.0 or q[1] > float(ny) - 1.5 or q[2] > float(nz) - 1.5 or not (s < 100.0):
        dead = True
    if not dead:
        c = cidx(int(wp.round(q[0])), int(wp.round(q[1])), int(wp.round(q[2])), nx, ny, nz)
        if int(solid[c]) != 0:
            dead = True
    if dead:
        state = wp.rand_init(seed, i)
        q = wp.vec3(seed_x + wp.randf(state) * 2.0,
                    seed_cy + (wp.randf(state) * 2.0 - 1.0) * seed_hy,
                    seed_cz + (wp.randf(state) * 2.0 - 1.0) * seed_hz)
        age = 0.0
        s = 1.0
    pos[i] = wp.vec4(q[0], q[1], q[2], age)
    spd[i] = s


@wp.kernel
def pack_particles(pos: wp.array(dtype=wp.vec4), spd: wp.array(dtype=wp.float32),
                   q16: wp.array(dtype=wp.uint16), q8: wp.array(dtype=wp.uint8), nx: int, ny: int, nz: int):
    i = wp.tid()
    p = pos[i]
    q16[4 * i + 0] = wp.uint16(int(wp.clamp(p[0] / float(nx), 0.0, 1.0) * 65535.0 + 0.5))
    q16[4 * i + 1] = wp.uint16(int(wp.clamp(p[1] / float(ny), 0.0, 1.0) * 65535.0 + 0.5))
    q16[4 * i + 2] = wp.uint16(int(wp.clamp(p[2] / float(nz), 0.0, 1.0) * 65535.0 + 0.5))
    q16[4 * i + 3] = wp.uint16(int(wp.clamp(p[3], 0.0, 65535.0)))
    q8[i] = wp.uint8(int(wp.clamp(spd[i] / 2.0, 0.0, 1.0) * 255.0 + 0.5))


@wp.func
def vort(mac: wp.array(dtype=wp.vec4), x: int, y: int, z: int, nx: int, ny: int, nz: int):
    dx = (mac[cidx(x + 1, y, z, nx, ny, nz)] - mac[cidx(x - 1, y, z, nx, ny, nz)]) * 0.5
    dy = (mac[cidx(x, y + 1, z, nx, ny, nz)] - mac[cidx(x, y - 1, z, nx, ny, nz)]) * 0.5
    dz = (mac[cidx(x, y, z + 1, nx, ny, nz)] - mac[cidx(x, y, z - 1, nx, ny, nz)]) * 0.5
    return wp.length(wp.vec3(dy[2] - dz[1], dz[0] - dx[2], dx[1] - dy[0]))


@wp.kernel
def slice_kernel(mac: wp.array(dtype=wp.vec4), solid: wp.array(dtype=wp.uint8), out: wp.array(dtype=wp.uint8),
                 nx: int, ny: int, nz: int, axis: int, index: int, field: int, w: int,
                 uin: float, lref: float, vmin: float, vmax: float):
    k = wp.tid()
    a = k % w
    b = k // w
    cx = index
    cy = b
    cz = a
    if axis == 2:
        cx = a
        cy = b
        cz = index
    elif axis == 1:
        cx = a
        cy = index
        cz = b
    idx = cx + nx * (cy + ny * cz)
    m = mac[idx]
    v = m[0] / uin
    if field == 0:
        v = wp.length(wp.vec3(m[0], m[1], m[2])) / uin
    elif field == 1:
        v = (m[3] - 1.0) / (1.5 * uin * uin)
    elif field == 2:
        v = vort(mac, cx, cy, cz, nx, ny, nz) * lref / uin
    t = wp.clamp((v - vmin) / (vmax - vmin), 0.0, 1.0)
    out[2 * k] = wp.uint8(int(t * 255.0 + 0.5))
    fl = int(255)
    if int(solid[idx]) != 0:
        fl = 0
    out[2 * k + 1] = wp.uint8(fl)


@wp.func
def rho_fluid(mac: wp.array(dtype=wp.vec4), solid: wp.array(dtype=wp.uint8), p: wp.vec3, nx: int, ny: int, nz: int):
    qx = wp.clamp(p[0], 0.0, float(nx) - 1.001)
    qy = wp.clamp(p[1], 0.0, float(ny) - 1.001)
    qz = wp.clamp(p[2], 0.0, float(nz) - 1.001)
    ix = int(wp.floor(qx))
    iy = int(wp.floor(qy))
    iz = int(wp.floor(qz))
    tx = qx - float(ix)
    ty = qy - float(iy)
    tz = qz - float(iz)
    acc = wp.vec2(0.0, 0.0)
    for c in range(8):
        ox = c & 1
        oy = (c >> 1) & 1
        oz = (c >> 2) & 1
        ci = cidx(ix + ox, iy + oy, iz + oz, nx, ny, nz)
        wx = 1.0 - tx
        if ox == 1:
            wx = tx
        wy = 1.0 - ty
        if oy == 1:
            wy = ty
        wz = 1.0 - tz
        if oz == 1:
            wz = tz
        wt = wx * wy * wz
        if int(solid[ci]) == 0:
            acc = acc + wp.vec2(wt * mac[ci][3], wt)
    return acc


@wp.kernel
def cp_kernel(mac: wp.array(dtype=wp.vec4), solid: wp.array(dtype=wp.uint8),
              verts: wp.array(dtype=wp.vec3), norms: wp.array(dtype=wp.vec3), out: wp.array(dtype=wp.int8),
              nx: int, ny: int, nz: int, uin: float, cp_range: float):
    i = wp.tid()
    p = verts[i]
    nr = norms[i]
    a = rho_fluid(mac, solid, p + nr * 1.25, nx, ny, nz)
    b = rho_fluid(mac, solid, p - nr * 1.25, nx, ny, nz)
    s = b
    if a[1] >= b[1]:
        s = a
    cp = float(0.0)
    if s[1] > 0.05:
        cp = (s[0] / s[1] - 1.0) / (1.5 * uin * uin)
    out[i] = wp.int8(int(wp.round(wp.clamp(cp / cp_range, -1.0, 1.0) * 127.0)))


_K = None


def kernels():
    global _K
    if _K is None:
        src = gen_source()
        h = hashlib.sha1(src.encode()).hexdigest()[:12]
        d = os.environ.get('WINDTUNNEL_GEN_DIR', os.path.join(tempfile.gettempdir(), 'windtunnel'))
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f'lbm_gen_{h}.py')
        if not os.path.exists(path):
            with open(path, 'w') as fh:
                fh.write(src)
        spec = importlib.util.spec_from_file_location(f'lbm_gen_{h}', path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        wp.set_module_options({'enable_backward': False}, module=mod)
        _K = mod
    return _K


wp.set_module_options({'enable_backward': False})

CP_RANGE = 1.2


class Solver:
    def __init__(self, dims, device):
        self.K = kernels()
        self.dev = device
        self.nx, self.ny, self.nz = dims
        self.n = n = self.nx * self.ny * self.nz
        self.f = [wp.zeros(19 * n, dtype=wp.float32, device=device) for _ in range(2)]
        self.solid = wp.zeros(n, dtype=wp.uint8, device=device)
        self.mac = wp.zeros(n, dtype=wp.vec4, device=device)
        self.forces = wp.zeros(3, dtype=wp.float32, device=device)
        self.slice_max = max(self.nx * self.ny, self.nx * self.nz, self.nz * self.ny)
        self.slice_buf = wp.zeros(self.slice_max * 2, dtype=wp.uint8, device=device)
        self.cur = 0
        self.num_p = 0
        self.num_verts = 0
        self.seed = 1

    def reset(self, uin):
        for buf in self.f:
            wp.launch(self.K.init_kernel, dim=self.n, inputs=[buf, self.mac, self.n, uin], device=self.dev)

    def set_solid(self, arr):
        assert arr.size == self.n
        self.solid.assign(wp.array(arr.astype(np.uint8), dtype=wp.uint8, device=self.dev))

    def set_surface(self, verts, norms):
        self.num_verts = len(verts)
        self.verts = wp.array(verts, dtype=wp.vec3, device=self.dev)
        self.norms = wp.array(norms, dtype=wp.vec3, device=self.dev)
        self.cp_buf = wp.zeros(max(1, self.num_verts), dtype=wp.int8, device=self.dev)

    def set_particles(self, count, p):
        rng = np.random.default_rng()
        d = np.zeros((count, 4), np.float32)
        d[:, 0] = 2 + rng.random(count) * (self.nx - 8)
        d[:, 1] = p['seedCy'] + (rng.random(count) * 2 - 1) * p['seedHy']
        d[:, 2] = p['seedCz'] + (rng.random(count) * 2 - 1) * p['seedHz']
        d[:, 3] = rng.random(count) * p['maxAge'] * 0.5
        self.num_p = count
        self.pos = wp.array(d, dtype=wp.vec4, device=self.dev)
        self.spd = wp.ones(count, dtype=wp.float32, device=self.dev)
        self.q16 = wp.zeros(count * 4, dtype=wp.uint16, device=self.dev)
        self.q8 = wp.zeros(count, dtype=wp.uint8, device=self.dev)

    def step(self, steps, p):
        for _ in range(steps):
            wp.launch(self.K.step_kernel, dim=self.n,
                      inputs=[self.f[self.cur], self.f[1 - self.cur], self.solid, self.mac,
                              self.nx, self.ny, self.nz, p['uIn'], p['tau'], p['smag']], device=self.dev)
            self.cur ^= 1
        if self.num_p and steps:
            self.seed += 1
            wp.launch(advect_kernel, dim=self.num_p,
                      inputs=[self.pos, self.spd, self.mac, self.solid, self.nx, self.ny, self.nz,
                              float(steps), p['uIn'], p['maxAge'], self.seed,
                              p['seedX'], p['seedCy'], p['seedCz'], p['seedHy'], p['seedHz']], device=self.dev)

    def slice_dims(self, axis):
        return {2: (self.nx, self.ny), 1: (self.nx, self.nz), 0: (self.nz, self.ny)}[axis]

    def frame(self, p, want_cp):
        out = {}
        if self.num_p:
            wp.launch(pack_particles, dim=self.num_p, inputs=[self.pos, self.spd, self.q16, self.q8, self.nx, self.ny, self.nz], device=self.dev)
            out['parts16'] = self.q16.numpy()
            out['parts8'] = self.q8.numpy()
        sl = p.get('slice')
        if sl and sl.get('field', -1) >= 0:
            axis = int(sl['axis'])
            w, h = self.slice_dims(axis)
            dim = (self.nx, self.ny, self.nz)[axis]
            index = max(0, min(dim - 1, int(sl['index'])))
            wp.launch(slice_kernel, dim=w * h,
                      inputs=[self.mac, self.solid, self.slice_buf, self.nx, self.ny, self.nz, axis, index,
                              int(sl['field']), w, p['uIn'], p['lref'], float(sl['vmin']), float(sl['vmax'])], device=self.dev)
            out['slice'] = self.slice_buf.numpy()[:w * h * 2]
            out['sliceW'], out['sliceH'], out['sliceAxis'] = w, h, axis
        if want_cp and self.num_verts:
            wp.launch(cp_kernel, dim=self.num_verts,
                      inputs=[self.mac, self.solid, self.verts, self.norms, self.cp_buf, self.nx, self.ny, self.nz, p['uIn'], CP_RANGE], device=self.dev)
            out['cp'] = self.cp_buf.numpy()
        self.forces.zero_()
        wp.launch(self.K.force_kernel, dim=self.n, inputs=[self.f[self.cur], self.solid, self.forces, self.nx, self.ny, self.nz], device=self.dev)
        out['forces'] = self.forces.numpy().tolist()
        return out
