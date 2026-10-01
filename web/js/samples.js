import * as THREE from 'three';

// Built-in test bodies. All are Y-up with the nose pointing to -x (upstream), in metres.

function positionsOf(geoms) {
  const arrs = geoms.map(g => (g.index ? g.toNonIndexed() : g).attributes.position.array);
  const out = new Float32Array(arrs.reduce((s, a) => s + a.length, 0));
  let o = 0;
  for (const a of arrs) { out.set(a, o); o += a.length; }
  return out;
}

function car() {
  const s = new THREE.Shape();
  s.moveTo(0.2, 0.3);
  s.lineTo(4.3, 0.3);
  s.quadraticCurveTo(4.56, 0.32, 4.55, 0.6);
  s.lineTo(4.5, 0.95);
  s.splineThru([
    new THREE.Vector2(4.25, 1.0), new THREE.Vector2(3.95, 1.04), new THREE.Vector2(3.55, 1.24),
    new THREE.Vector2(3.05, 1.39), new THREE.Vector2(2.55, 1.42), new THREE.Vector2(2.15, 1.35),
    new THREE.Vector2(1.6, 1.0), new THREE.Vector2(1.0, 0.9), new THREE.Vector2(0.4, 0.8),
    new THREE.Vector2(0.08, 0.66), new THREE.Vector2(0.0, 0.5),
  ]);
  s.lineTo(0.02, 0.4);
  s.quadraticCurveTo(0.05, 0.3, 0.2, 0.3);
  const body = new THREE.ExtrudeGeometry(s, { depth: 1.56, bevelEnabled: true, bevelThickness: 0.14, bevelSize: 0.1, bevelSegments: 5, curveSegments: 20 });
  const parts = [body];
  for (const x of [0.95, 3.6]) for (const z of [0.0, 1.56]) {
    const w = new THREE.CylinderGeometry(0.34, 0.34, 0.26, 32);
    w.rotateX(Math.PI / 2);
    w.translate(x, 0.34, z);
    parts.push(w);
  }
  return positionsOf(parts);
}

function ahmed() {
  // Ahmed reference body, 25° slant (1044 x 389 x 288 mm)
  const L = 1.044, H = 0.288, R = 0.1, sl = 0.222, drop = sl * Math.tan(25 * Math.PI / 180);
  const s = new THREE.Shape();
  s.moveTo(R, 0);
  s.lineTo(L, 0);
  s.lineTo(L, H - drop);
  s.lineTo(L - sl, H);
  s.lineTo(R, H);
  s.quadraticCurveTo(0, H, 0, H - R);
  s.lineTo(0, R);
  s.quadraticCurveTo(0, 0, R, 0);
  return positionsOf([new THREE.ExtrudeGeometry(s, { depth: 0.389, bevelEnabled: false, curveSegments: 16 })]);
}

function wing() {
  const pts = [];
  const N = 80;
  const yt = (x) => 5 * 0.12 * (0.2969 * Math.sqrt(x) - 0.126 * x - 0.3516 * x * x + 0.2843 * x ** 3 - 0.1036 * x ** 4);
  for (let i = N; i >= 0; i--) { const x = (1 - Math.cos(Math.PI * i / N)) / 2; pts.push(new THREE.Vector2(x, yt(x))); }
  for (let i = 1; i < N; i++) { const x = (1 - Math.cos(Math.PI * i / N)) / 2; pts.push(new THREE.Vector2(x, -yt(x))); }
  const s = new THREE.Shape(pts);
  return positionsOf([new THREE.ExtrudeGeometry(s, { depth: 1.6, bevelEnabled: false })]);
}

export const SAMPLES = {
  car: { label: 'Car', make: car, length: 4.6, ground: true, gap: 0, aoa: 0, rake: 'vsheet' },
  ahmed: { label: 'Ahmed body', make: ahmed, length: 1.044, ground: true, gap: 0.05, aoa: 0, rake: 'volume' },
  wing: { label: 'NACA 0012 wing', make: wing, length: 1.0, ground: false, gap: 0, aoa: 8, rake: 'volume', refArea: 'planform' },
  sphere: { label: 'Sphere', make: () => positionsOf([new THREE.SphereGeometry(0.5, 96, 48)]), length: 0.3, ground: false, gap: 0, aoa: 0, rake: 'volume' },
};
