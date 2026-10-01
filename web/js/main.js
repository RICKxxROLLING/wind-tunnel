import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';
import { Remote } from './net.js';
import { voxelize } from './voxelize.js';
import { SAMPLES } from './samples.js';
import { CMAPS, makeLUT, GLSL_CMAPS, turbo, coolwarm } from './colormaps.js';

const $ = (id) => document.getElementById(id);
const DEG = Math.PI / 180;
const U_IN = 0.1;          // inlet speed in lattice units (Mach ~0.17)
const NU_AIR = 1.5e-5;     // m^2/s
const TRAIL = 14;          // trail length in readbacks

const FIELDS = {
  speed: { code: 0, min: 0, max: 1.6, cmap: 0, label: 'Speed  |u| / U∞' },
  pressure: { code: 1, min: -1.2, max: 1.2, cmap: 1, label: 'Pressure coefficient Cp' },
  vorticity: { code: 2, min: 0, max: 12, cmap: 2, label: 'Vorticity  |ω|·L / U∞' },
  ux: { code: 3, min: -0.4, max: 1.4, cmap: 0, label: 'Streamwise velocity  u / U∞' },
};
const CP_RANGE = [-1.2, 1.2];

const state = {
  res: null,
  base: null,               // { positions, name, tris, isSample }
  baseRot: new THREE.Matrix4(),
  aoa: 0, yaw: 0, scaleMul: 1, ground: true, gap: 0,
  windKmh: 120, lengthM: 4.6,
  speed: 10, running: true,
  lines: true, lineColor: 'speed', count: 5000, rake: 'vsheet', rakeSize: 1,
  field: 'off', axis: 'z', slicePos: 0.5, sliceOpacity: 0.85,
  cp: true, smag: 0.16, refArea: 'frontal',
  model: null,              // lattice-space info after placement
};

// ---------- three.js scene ----------
const renderer = new THREE.WebGLRenderer({ canvas: $('view'), antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0d13);
const camera = new THREE.PerspectiveCamera(40, 1, 0.05, 200);
camera.position.set(-3.5, 3.2, 10.5);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.target.set(-0.6, -0.4, 0);
scene.add(new THREE.HemisphereLight(0xdfe8ff, 0x1a1d24, 1.4));
const sun = new THREE.DirectionalLight(0xffffff, 2.2);
sun.position.set(-4, 8, 6);
scene.add(sun);
const fill = new THREE.DirectionalLight(0x88aaff, 0.6);
fill.position.set(5, -2, -6);
scene.add(fill);

const tunnel = new THREE.Group();
scene.add(tunnel);

const modelMat = new THREE.MeshStandardMaterial({ color: 0xd5dae3, metalness: 0.2, roughness: 0.45, side: THREE.DoubleSide });
const modelMesh = new THREE.Mesh(new THREE.BufferGeometry(), modelMat);
modelMesh.frustumCulled = false;

const sliceMat = new THREE.ShaderMaterial({
  uniforms: { tex: { value: null }, cmapId: { value: 0 }, opacity: { value: 0.85 } },
  vertexShader: `varying vec2 vUv; void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
  fragmentShader: `precision highp float; varying vec2 vUv; uniform sampler2D tex; uniform float opacity; uniform int cmapId;
    ${GLSL_CMAPS}
    void main() {
      vec2 s = texture2D(tex, vUv).rg;
      if (s.g < 0.5) discard;
      gl_FragColor = vec4(cmap(cmapId, s.r), opacity);
    }`,
  transparent: true, side: THREE.DoubleSide, depthWrite: false,
});
const sliceMesh = new THREE.Mesh(new THREE.BufferGeometry(), sliceMat);
sliceMesh.renderOrder = 1;

const trailMat = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false });
let trails = null;
let trailState = null;

let remote = null, server = null, dims = null;
let frameMesh = null, floorMesh = null;

function resize() {
  const w = innerWidth, h = innerHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
addEventListener('resize', resize);
resize();

function buildTunnel() {
  const [nx, ny, nz] = dims;
  for (const c of [...tunnel.children]) tunnel.remove(c);
  const k = 10 / nx;
  tunnel.scale.setScalar(k);
  tunnel.position.set(-(nx - 1) / 2 * k, -(ny - 1) / 2 * k, -(nz - 1) / 2 * k);

  const box = new THREE.BoxGeometry(nx - 1, ny - 1, nz - 1);
  box.translate((nx - 1) / 2, (ny - 1) / 2, (nz - 1) / 2);
  frameMesh = new THREE.LineSegments(new THREE.EdgesGeometry(box), new THREE.LineBasicMaterial({ color: 0x3a4458, transparent: true, opacity: 0.8 }));
  tunnel.add(frameMesh);

  const floorGeo = new THREE.PlaneGeometry(nx - 1, nz - 1);
  floorGeo.rotateX(-Math.PI / 2);
  floorGeo.translate((nx - 1) / 2, 0.5, (nz - 1) / 2);
  floorMesh = new THREE.Mesh(floorGeo, new THREE.MeshStandardMaterial({ color: 0x161b25, roughness: 0.9, transparent: true, opacity: 0.9 }));
  tunnel.add(floorMesh);

  // inlet arrows
  const arrows = new THREE.Group();
  for (const [y, z] of [[0.3, 0.3], [0.7, 0.3], [0.3, 0.7], [0.7, 0.7], [0.5, 0.5]]) {
    const a = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(-0.08 * nx, y * ny, z * nz), 0.057 * nx, 0x4cc9f0, 0.017 * nx, 0.011 * nx);
    a.line.material.transparent = true; a.line.material.opacity = 0.5;
    arrows.add(a);
  }
  tunnel.add(arrows);

  tunnel.add(modelMesh);
  tunnel.add(sliceMesh);
  buildTrails();
}

function buildTrails() {
  if (trails) { tunnel.remove(trails); trails.geometry.dispose(); }
  const P = state.count;
  const segs = P * (TRAIL - 1);
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(segs * 6), 3).setUsage(THREE.DynamicDrawUsage));
  g.setAttribute('color', new THREE.BufferAttribute(new Float32Array(segs * 6), 3).setUsage(THREE.DynamicDrawUsage));
  trails = new THREE.LineSegments(g, trailMat);
  trails.frustumCulled = false;
  trails.renderOrder = 2;
  trails.visible = state.lines;
  tunnel.add(trails);
  trailState = { P, head: 0, hist: new Float32Array(P * TRAIL * 3), spd: new Float32Array(P * TRAIL), valid: new Uint8Array(P), lastAge: new Float32Array(P) };
}

// ---------- model placement ----------
function setBase(positions, name, opts = {}) {
  // center the raw geometry on its bbox
  const b = new THREE.Box3().setFromArray(positions);
  const c = b.getCenter(new THREE.Vector3());
  const p = new Float32Array(positions.length);
  for (let i = 0; i < p.length; i += 3) { p[i] = positions[i] - c.x; p[i + 1] = positions[i + 1] - c.y; p[i + 2] = positions[i + 2] - c.z; }
  state.base = { positions: p, name, tris: p.length / 9 };
  state.baseRot.identity();
  if (opts.zUp) {
    state.baseRot.makeRotationX(-Math.PI / 2);
    const s = b.getSize(new THREE.Vector3());
    if (s.y > s.x * 1.15) state.baseRot.premultiply(new THREE.Matrix4().makeRotationY(Math.PI / 2));
  }
}

function loadSample(key) {
  const s = SAMPLES[key];
  setBase(s.make(), s.label);
  Object.assign(state, { aoa: s.aoa, yaw: 0, scaleMul: 1, ground: s.ground, gap: s.gap, lengthM: s.length, rake: s.rake, refArea: s.refArea || 'frontal' });
  syncInputs();
  applyModel(true);
}

async function loadFile(file) {
  if (!/\.stl$/i.test(file.name)) { toast('Please drop an .stl file'); return; }
  showBusy(true, 'Loading…');
  try {
    const buf = await file.arrayBuffer();
    const geom = new STLLoader().parse(buf);
    const pos = geom.attributes.position.array;
    if (!pos.length) throw new Error('empty mesh');
    setBase(pos, file.name, { zUp: true });
    const size = new THREE.Box3().setFromArray(pos).getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z);
    Object.assign(state, { aoa: 0, yaw: 0, scaleMul: 1, gap: 0, refArea: 'frontal', lengthM: +(Math.min(100, Math.max(0.01, maxDim / 1000))).toPrecision(3), rake: 'volume' });
    syncInputs();
    applyModel(true);
  } catch (e) {
    console.error(e);
    toast('Could not read that STL: ' + e.message);
  } finally { showBusy(false); }
}

let applyTimer = 0;
function scheduleApply() {
  clearTimeout(applyTimer);
  applyTimer = setTimeout(() => applyModel(false), 60);
}

function applyModel(fresh) {
  if (!state.base || !dims || !remote?.open) return;
  const [nx, ny, nz] = dims;
  const src = state.base.positions;
  const m = new THREE.Matrix4().makeRotationZ(-state.aoa * DEG)
    .multiply(new THREE.Matrix4().makeRotationY(state.yaw * DEG))
    .multiply(state.baseRot);
  const e = m.elements;
  const N = src.length / 3;
  const out = new Float32Array(src.length);
  let mnx = Infinity, mny = Infinity, mnz = Infinity, mxx = -Infinity, mxy = -Infinity, mxz = -Infinity;
  for (let i = 0; i < N; i++) {
    const x = src[i * 3], y = src[i * 3 + 1], z = src[i * 3 + 2];
    const X = e[0] * x + e[4] * y + e[8] * z, Y = e[1] * x + e[5] * y + e[9] * z, Z = e[2] * x + e[6] * y + e[10] * z;
    out[i * 3] = X; out[i * 3 + 1] = Y; out[i * 3 + 2] = Z;
    if (X < mnx) mnx = X; if (X > mxx) mxx = X;
    if (Y < mny) mny = Y; if (Y > mxy) mxy = Y;
    if (Z < mnz) mnz = Z; if (Z > mxz) mxz = Z;
  }
  const ex = Math.max(mxx - mnx, 1e-9), ey = Math.max(mxy - mny, 1e-9), ez = Math.max(mxz - mnz, 1e-9);
  const s = Math.min(0.36 * nx / ex, (state.ground ? 0.42 : 0.4) * ny / ey, 0.6 * nz / ez) * state.scaleMul;
  const cx = Math.max(0.3 * nx, ex * s / 2 + 10);
  const gapCells = state.gap * s;
  for (let i = 0; i < N; i++) {
    out[i * 3] = (out[i * 3] - (mnx + mxx) / 2) * s + cx;
    out[i * 3 + 1] = state.ground ? (out[i * 3 + 1] - mny) * s + 1 + gapCells : (out[i * 3 + 1] - (mny + mxy) / 2) * s + (ny - 1) / 2;
    out[i * 3 + 2] = (out[i * 3 + 2] - (mnz + mxz) / 2) * s + (nz - 1) / 2;
  }
  // face normals
  const nrm = new Float32Array(out.length);
  for (let t = 0; t < out.length; t += 9) {
    const ux = out[t + 3] - out[t], uy = out[t + 4] - out[t + 1], uz = out[t + 5] - out[t + 2];
    const vx = out[t + 6] - out[t], vy = out[t + 7] - out[t + 1], vz = out[t + 8] - out[t + 2];
    let qx = uy * vz - uz * vy, qy = uz * vx - ux * vz, qz = ux * vy - uy * vx;
    const l = Math.hypot(qx, qy, qz) || 1;
    qx /= l; qy /= l; qz /= l;
    for (let k = 0; k < 9; k += 3) { nrm[t + k] = qx; nrm[t + k + 1] = qy; nrm[t + k + 2] = qz; }
  }

  const t0 = performance.now();
  const vox = voxelize(out, nx, ny, nz, state.ground);
  const vt = performance.now() - t0;
  remote.sendBinary({ type: 'solid', dims }, vox.solid);
  const surf = new Float32Array(out.length * 2);
  surf.set(out); surf.set(nrm, out.length);
  remote.sendBinary({ type: 'surface', numVerts: N }, new Uint8Array(surf.buffer));

  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(out, 3));
  g.setAttribute('normal', new THREE.BufferAttribute(nrm, 3));
  const col = new Float32Array(out.length).fill(0.7);
  g.setAttribute('color', new THREE.BufferAttribute(col, 3));
  modelMesh.geometry.dispose();
  modelMesh.geometry = g;

  const bbox = { min: [cx - ex * s / 2, state.ground ? 1 + gapCells : (ny - 1) / 2 - ey * s / 2, (nz - 1) / 2 - ez * s / 2] };
  bbox.max = [bbox.min[0] + ex * s, bbox.min[1] + ey * s, bbox.min[2] + ez * s];
  state.model = { bbox, lx: ex * s, height: ey * s, frontal: vox.frontal, planform: vox.planform, cells: vox.count };
  floorMesh.visible = state.ground;

  $('modelInfo').textContent = `${state.base.name} · ${state.base.tris.toLocaleString()} triangles · ${vox.count.toLocaleString()} solid cells (${vt.toFixed(0)} ms)`;
  updateRake(fresh);
  updateFlowParams();
  updateSlice();
  forceAvg = null;
  if (fresh) remote.send({ type: 'reset' });
}

function updateRake(respawn) {
  if (!state.model || !remote) return;
  const [, ny, nz] = dims;
  const b = state.model.bbox;
  const cy = (b.min[1] + b.max[1]) / 2, cz = (b.min[2] + b.max[2]) / 2;
  let hy = ((b.max[1] - b.min[1]) / 2 * 1.3 + 2) * state.rakeSize;
  let hz = ((b.max[2] - b.min[2]) / 2 * 1.3 + 2) * state.rakeSize;
  let ycen = cy;
  if (state.rake === 'vsheet') hz = 0.4;
  if (state.rake === 'hsheet') { hy = 0.4; if (state.ground) ycen = b.min[1] + (b.max[1] - b.min[1]) * 0.45; }
  const lo = state.ground ? 1.6 : 1.5;
  const y0 = Math.max(lo, ycen - hy), y1 = Math.min(ny - 2.5, ycen + hy);
  const z0 = Math.max(1.5, cz - hz), z1 = Math.min(nz - 2.5, cz + hz);
  remote.setParams({ particles: state.count, seedX: 2, seedCy: (y0 + y1) / 2, seedHy: (y1 - y0) / 2, seedCz: (z0 + z1) / 2, seedHz: (z1 - z0) / 2 });
  if (respawn) { remote.send({ type: 'respawn' }); buildTrails(); }
}

function reynolds() { return (state.windKmh / 3.6) * state.lengthM / NU_AIR; }

function updateFlowParams() {
  if (!remote) return;
  const Re = reynolds();
  const lx = state.model ? state.model.lx : 40;
  // viscosity floor: below ~0.507 the near-inviscid lattice rings with grid-scale noise
  const tau = Math.max(0.507, 0.5 + 3 * U_IN * lx / Re);
  remote.setParams({
    uIn: U_IN, tau, smag: state.smag, steps: state.speed,
    maxAge: 2.2 * (dims ? dims[0] : 200) / (U_IN * state.speed),
    lref: state.model ? state.model.lx : 40,
  });
  const txt = fmtRe(Re);
  $('reVal').innerHTML = txt;
  $('reHud').innerHTML = txt;
}

function fmtRe(x) {
  if (x < 1e4) return Math.round(x).toLocaleString();
  const e = Math.floor(Math.log10(x));
  return `${(x / 10 ** e).toFixed(1)}×10<sup>${e}</sup>`;
}

function updateSlice() {
  if (!dims) return;
  const [nx, ny, nz] = dims;
  const on = state.field !== 'off';
  sliceMesh.visible = on;
  const axis = { x: 0, y: 1, z: 2 }[state.axis];
  const dim = [nx, ny, nz][axis];
  const idx = Math.round(state.slicePos * (dim - 1));
  const f = FIELDS[state.field] || FIELDS.speed;
  remote.setParams({ slice: on ? { axis, index: idx, field: f.code, vmin: f.min, vmax: f.max } : null });
  sliceMat.uniforms.cmapId.value = f.cmap;
  sliceMat.uniforms.opacity.value = state.sliceOpacity;
  const X = nx - 1, Y = ny - 1, Z = nz - 1, s = idx;
  const corners = axis === 2 ? [[0, 0, s], [X, 0, s], [X, Y, s], [0, Y, s]]
    : axis === 1 ? [[0, s, 0], [X, s, 0], [X, s, Z], [0, s, Z]]
    : [[s, 0, 0], [s, 0, Z], [s, Y, Z], [s, Y, 0]];
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(corners.flat()), 3));
  g.setAttribute('uv', new THREE.BufferAttribute(new Float32Array([0, 0, 1, 0, 1, 1, 0, 1]), 2));
  g.setIndex([0, 1, 2, 0, 2, 3]);
  sliceMesh.geometry.dispose();
  sliceMesh.geometry = g;
  drawLegend();
}

// ---------- readback handling ----------
const LUTS = { turbo: makeLUT(turbo), coolwarm: makeLUT(coolwarm) };
let sliceTex = null;
let forceAvg = null;
let rbCount = 0;

function onFrame(d) {
  const h = d.header;
  $('perf').textContent = h.running ? `${Math.round(h.mlups).toLocaleString()} MLUPS` : 'paused';
  if (!dims || h.dims.join() !== dims.join()) return;
  if (d.parts16 && trailState && h.numP === trailState.P) updateTrails(d.parts16, d.parts8);
  if (d.slice && state.field !== 'off') updateSliceTex(d.slice, h.sliceW, h.sliceH);
  if (d.cp && modelMesh.geometry.attributes.position.count === h.numVerts) updateCp(d.cp);
  if (state.model && h.steps > 0) {
    const [fx, fy] = h.forces;
    const q = 0.5 * U_IN * U_IN * Math.max(1, state.refArea === 'planform' ? state.model.planform : state.model.frontal);
    const cd = fx / q, cl = fy / q;
    if (!Number.isFinite(cd) || Math.abs(cd) > 1e3) return diverged();
    const dt = h.steps;
    if (!forceAvg) forceAvg = { cd: 0, cl: 0, w: 0, steps: 0 };
    forceAvg.steps += dt;
    if (forceAvg.steps < 800) { $('cd').textContent = $('cl').textContent = 'settling'; return; }
    // running mean while young, then an exponential window of ~2500 steps
    forceAvg.w = Math.min(forceAvg.w + dt, 2500);
    const a = dt / forceAvg.w;
    forceAvg.cd += (cd - forceAvg.cd) * a;
    forceAvg.cl += (cl - forceAvg.cl) * a;
    $('cd').textContent = forceAvg.cd.toFixed(3);
    $('cl').textContent = forceAvg.cl.toFixed(3);
  }
}

function updateTrails(p16, p8) {
  const ts = trailState;
  const { P, hist, spd, valid, lastAge } = ts;
  const T = TRAIL;
  ts.head = (ts.head + 1) % T;
  const head = ts.head;
  const [sx, sy, sz] = dims.map(d => d / 65535);
  for (let p = 0; p < P; p++) {
    const o = p * 4;
    const age = p16[o + 3];
    if (!(age >= lastAge[p])) valid[p] = 0;
    lastAge[p] = age;
    const h = p * T + head;
    hist[h * 3] = p16[o] * sx; hist[h * 3 + 1] = p16[o + 1] * sy; hist[h * 3 + 2] = p16[o + 2] * sz;
    spd[h] = p8[p] * (2 / 255);
    if (valid[p] < T) valid[p]++;
  }
  if (!state.lines) return;
  const pos = trails.geometry.attributes.position.array;
  const col = trails.geometry.attributes.color.array;
  const smoke = state.lineColor === 'smoke';
  const gain = smoke ? (P > 8000 ? 0.18 : 0.3) : (P > 8000 ? 0.45 : 0.75);
  const lut = LUTS.turbo;
  let v = 0;
  for (let p = 0; p < P; p++) {
    const vp = valid[p];
    for (let k = 0; k < T - 1; k++, v += 6) {
      if (k + 1 >= vp) {
        pos[v] = pos[v + 1] = pos[v + 2] = pos[v + 3] = pos[v + 4] = pos[v + 5] = 0;
        col[v] = col[v + 1] = col[v + 2] = col[v + 3] = col[v + 4] = col[v + 5] = 0;
        continue;
      }
      const a = p * T + (head - k + T) % T, b = p * T + (head - k - 1 + T) % T;
      pos[v] = hist[a * 3]; pos[v + 1] = hist[a * 3 + 1]; pos[v + 2] = hist[a * 3 + 2];
      pos[v + 3] = hist[b * 3]; pos[v + 4] = hist[b * 3 + 1]; pos[v + 5] = hist[b * 3 + 2];
      const fa = gain * (1 - k / (T - 1)), fb = gain * (1 - (k + 1) / (T - 1));
      if (smoke) {
        col[v] = col[v + 1] = col[v + 2] = fa;
        col[v + 3] = col[v + 4] = col[v + 5] = fb;
      } else {
        const li = Math.min(255, Math.max(0, (spd[a] / 1.6 * 255) | 0)) * 3;
        col[v] = lut[li] * fa; col[v + 1] = lut[li + 1] * fa; col[v + 2] = lut[li + 2] * fa;
        col[v + 3] = lut[li] * fb; col[v + 4] = lut[li + 1] * fb; col[v + 5] = lut[li + 2] * fb;
      }
    }
  }
  trails.geometry.attributes.position.needsUpdate = true;
  trails.geometry.attributes.color.needsUpdate = true;
}

function updateSliceTex(data, w, h) {
  if (!sliceTex || sliceTex.image.width !== w || sliceTex.image.height !== h) {
    sliceTex?.dispose();
    sliceTex = new THREE.DataTexture(new Uint8Array(w * h * 2), w, h, THREE.RGFormat, THREE.UnsignedByteType);
    sliceTex.magFilter = sliceTex.minFilter = THREE.LinearFilter;
    sliceTex.unpackAlignment = 1;
    sliceMat.uniforms.tex.value = sliceTex;
  }
  sliceTex.image.data.set(data);
  sliceTex.needsUpdate = true;
}

function updateCp(cp) {
  const attr = modelMesh.geometry.attributes.color;
  const c = attr.array;
  const lut = LUTS.coolwarm;
  for (let i = 0; i < cp.length; i++) {
    const li = Math.min(255, Math.max(0, ((cp[i] / 127 + 1) * 0.5 * 255) | 0)) * 3;
    c[i * 3] = lut[li]; c[i * 3 + 1] = lut[li + 1]; c[i * 3 + 2] = lut[li + 2];
  }
  attr.needsUpdate = true;
}

let lastDiverge = 0;
function diverged() {
  const now = performance.now();
  if (now - lastDiverge < 3000) return;
  lastDiverge = now;
  state.smag = Math.min(0.3, state.smag + 0.03);
  syncInputs();
  updateFlowParams();
  remote.send({ type: 'reset' });
  remote.send({ type: 'respawn' });
  buildTrails();
  forceAvg = null;
  toast('Flow went unstable — reset with a stronger turbulence model');
}

// ---------- legend ----------
function drawLegend() {
  const items = [];
  if (state.lines && state.lineColor === 'speed') items.push({ label: 'Streamlines · speed |u| / U∞', fn: turbo, min: 0, max: 1.6 });
  if (state.field !== 'off') { const f = FIELDS[state.field]; items.push({ label: 'Plane · ' + f.label, fn: CMAPS[f.cmap], min: f.min, max: f.max }); }
  if (state.cp) items.push({ label: 'Surface · pressure Cp', fn: coolwarm, min: CP_RANGE[0], max: CP_RANGE[1] });
  const el = $('legend');
  el.style.display = items.length ? '' : 'none';
  el.innerHTML = '';
  for (const it of items) {
    const div = document.createElement('div');
    div.className = 'item';
    div.innerHTML = `<div class="title">${it.label}</div>`;
    const cv = document.createElement('canvas');
    cv.width = 256; cv.height = 1;
    const ctx = cv.getContext('2d');
    for (let i = 0; i < 256; i++) {
      const c = it.fn(i / 255);
      ctx.fillStyle = `rgb(${c[0] * 255 | 0},${c[1] * 255 | 0},${c[2] * 255 | 0})`;
      ctx.fillRect(i, 0, 1, 1);
    }
    div.appendChild(cv);
    const ends = document.createElement('div');
    ends.className = 'ends';
    ends.innerHTML = `<span>${it.min}</span><span>${(it.min + it.max) / 2}</span><span>${it.max}</span>`;
    div.appendChild(ends);
    el.appendChild(div);
  }
}

// ---------- UI ----------
function toast(msg) {
  const t = $('toast');
  t.textContent = msg;
  t.style.display = 'block';
  clearTimeout(toast.t);
  toast.t = setTimeout(() => (t.style.display = 'none'), 4000);
}
function showBusy(on, msg = 'Voxelizing…') { $('busy').textContent = msg; $('busy').style.display = on ? 'block' : 'none'; }

function syncInputs() {
  $('aoa').value = state.aoa; $('aoaVal').textContent = state.aoa + '°';
  $('yaw').value = state.yaw; $('yawVal').textContent = state.yaw + '°';
  $('scale').value = state.scaleMul; $('scaleVal').textContent = state.scaleMul.toFixed(2) + '×';
  $('ground').checked = state.ground;
  $('wind').value = state.windKmh; $('windVal').textContent = state.windKmh + ' km/h';
  $('length').value = state.lengthM;
  $('speed').value = state.speed; $('speedVal').textContent = state.speed + ' steps/frame';
  $('res').value = state.res;
  $('refArea').value = state.refArea;
  $('stab').value = state.smag; $('stabVal').textContent = state.smag.toFixed(2);
  $('lines').checked = state.lines;
  $('lineColor').value = state.lineColor;
  $('rake').value = state.rake;
  $('count').value = String(state.count);
  $('rakeSize').value = state.rakeSize; $('rakeSizeVal').textContent = state.rakeSize.toFixed(2) + '×';
  $('field').value = state.field;
  $('axis').value = state.axis;
  $('slicePos').value = state.slicePos; $('slicePosVal').textContent = Math.round(state.slicePos * 100) + '%';
  $('sliceOpacity').value = state.sliceOpacity; $('sliceOpacityVal').textContent = Math.round(state.sliceOpacity * 100) + '%';
  $('cp').checked = state.cp;
  $('play').textContent = state.running ? 'Pause' : 'Run';
}

function bind(id, ev, fn) { $(id).addEventListener(ev, (e) => { fn(e.target); syncInputs(); }); }

bind('aoa', 'input', (el) => { state.aoa = +el.value; scheduleApply(); });
bind('yaw', 'input', (el) => { state.yaw = +el.value; scheduleApply(); });
bind('scale', 'input', (el) => { state.scaleMul = +el.value; scheduleApply(); });
bind('ground', 'change', (el) => { state.ground = el.checked; applyModel(true); });
const rot = (axis, a) => () => { state.baseRot.premultiply(new THREE.Matrix4()[`makeRotation${axis}`](a)); applyModel(true); };
$('rotX').onclick = rot('X', Math.PI / 2);
$('rotY').onclick = rot('Y', Math.PI / 2);
$('rotZ').onclick = rot('Z', Math.PI / 2);
$('flip').onclick = rot('Y', Math.PI);
bind('wind', 'input', (el) => { state.windKmh = +el.value; updateFlowParams(); });
bind('length', 'change', (el) => { state.lengthM = Math.max(0.001, +el.value || 1); updateFlowParams(); });
bind('speed', 'input', (el) => { state.speed = +el.value; updateFlowParams(); });
bind('refArea', 'change', (el) => { state.refArea = el.value; forceAvg = null; });
bind('stab', 'input', (el) => { state.smag = +el.value; updateFlowParams(); });
bind('res', 'change', (el) => { state.res = el.value; requestGrid(); });
$('play').onclick = () => { state.running = !state.running; remote?.setParams({ running: state.running }); syncInputs(); };
$('resetFlow').onclick = () => { remote?.send({ type: 'reset' }); remote?.send({ type: 'respawn' }); buildTrails(); forceAvg = null; };
bind('lines', 'change', (el) => { state.lines = el.checked; trails.visible = state.lines; drawLegend(); });
bind('lineColor', 'change', (el) => { state.lineColor = el.value; drawLegend(); });
bind('rake', 'change', (el) => { state.rake = el.value; updateRake(true); });
bind('count', 'change', (el) => { state.count = +el.value; updateRake(true); });
bind('rakeSize', 'input', (el) => { state.rakeSize = +el.value; updateRake(false); });
bind('field', 'change', (el) => { state.field = el.value; updateSlice(); });
bind('axis', 'change', (el) => { state.axis = el.value; updateSlice(); });
bind('slicePos', 'input', (el) => { state.slicePos = +el.value; updateSlice(); });
bind('sliceOpacity', 'input', (el) => { state.sliceOpacity = +el.value; sliceMat.uniforms.opacity.value = state.sliceOpacity; });
bind('cp', 'change', (el) => {
  state.cp = el.checked;
  remote?.setParams({ cp: state.cp });
  modelMat.vertexColors = state.cp;
  modelMat.color.set(state.cp ? 0xffffff : 0xd5dae3);
  modelMat.needsUpdate = true;
  drawLegend();
});
document.querySelectorAll('[data-sample]').forEach(b => b.onclick = () => loadSample(b.dataset.sample));
$('dropzone').onclick = () => $('file').click();
$('file').onchange = (e) => { if (e.target.files[0]) loadFile(e.target.files[0]); e.target.value = ''; };
addEventListener('keydown', (e) => {
  if (e.code === 'Space' && !/INPUT|SELECT|BUTTON/.test(document.activeElement.tagName)) { e.preventDefault(); state.running = !state.running; remote?.setParams({ running: state.running }); syncInputs(); }
});
let dragDepth = 0;
addEventListener('dragenter', (e) => { e.preventDefault(); dragDepth++; document.body.classList.add('dragging'); });
addEventListener('dragleave', () => { if (--dragDepth <= 0) { dragDepth = 0; document.body.classList.remove('dragging'); } });
addEventListener('dragover', (e) => e.preventDefault());
addEventListener('drop', (e) => {
  e.preventDefault(); dragDepth = 0; document.body.classList.remove('dragging');
  const f = e.dataTransfer.files[0];
  if (f) loadFile(f);
});

// ---------- solver server link ----------
function setStatus(msg) {
  if (msg) { $('status').textContent = msg; return; }
  if (!server || !dims) { $('status').textContent = 'connecting to solver…'; return; }
  $('status').textContent = `${server.gpu} · ${dims.join(' × ')} grid · ${(dims[0] * dims[1] * dims[2] / 1e6).toFixed(2)}M cells`;
}

function requestGrid() {
  if (remote?.open && state.res) remote.send({ type: 'grid', res: state.res });
}

function onHello(h) {
  server = h;
  const sel = $('res');
  sel.innerHTML = '';
  for (const r of h.resolutions) {
    const o = document.createElement('option');
    const cells = r.dims[0] * r.dims[1] * r.dims[2] / 1e6;
    o.value = r.name;
    o.textContent = `${r.name[0].toUpperCase()}${r.name.slice(1)} · ${cells < 1 ? cells.toFixed(2) : cells.toFixed(1)}M cells`;
    o.disabled = !r.fits;
    sel.appendChild(o);
  }
  if (!h.resolutions.some(r => r.name === state.res && r.fits)) state.res = h.default;
  syncInputs();
  setStatus();
  requestGrid();
}

function onGrid(g) {
  state.res = g.res;
  const changed = !dims || dims.join() !== g.dims.join();
  dims = g.dims;
  if (changed) buildTunnel();
  remote.setParams({ running: state.running, cp: state.cp, particles: state.count });
  if (!state.base) {
    const s = SAMPLES.car;
    setBase(s.make(), s.label);
    Object.assign(state, { aoa: s.aoa, ground: s.ground, gap: s.gap, lengthM: s.length, rake: s.rake });
  }
  applyModel(true);
  syncInputs();
  setStatus();
}

function loop() {
  requestAnimationFrame(loop);
  controls.update();
  renderer.render(scene, camera);
}

function start() {
  syncInputs();
  modelMat.vertexColors = state.cp;
  modelMat.color.set(state.cp ? 0xffffff : 0xd5dae3);
  modelMat.needsUpdate = true;
  drawLegend();
  setStatus();
  remote = new Remote({
    onOpen: () => setStatus('connected · waiting for solver…'),
    onClose: () => { server = null; setStatus('solver disconnected — reconnecting…'); $('perf').textContent = 'offline'; },
    onJson: (m) => {
      if (m.type === 'hello') onHello(m);
      else if (m.type === 'grid') onGrid(m);
      else if (m.type === 'error') toast(m.message);
    },
    onFrame: (d) => { try { onFrame(d); } finally { remote.ready(); } },
  });
  loop();
}
start();
