// Turns a triangle soup (lattice coordinates, non-indexed) into a solid-cell mask.
// 1) ray parity along +x for every (y,z) column, 2) rasterized surface shell so thin or
// leaky geometry still blocks flow, 3) flood fill from the domain boundary so pockets
// enclosed by overlapping parts become solid. Ground layer (y = 0) is tagged 2 = moving belt.
export function voxelize(pos, nx, ny, nz, ground) {
  const n = nx * ny * nz;
  const vox = new Uint8Array(n);
  const cols = new Array(ny * nz);
  const OY = 0.01373, OZ = 0.00717; // jitter rays off exact edges/vertices
  const T = pos.length / 9;

  for (let t = 0; t < T; t++) {
    const o = t * 9;
    const ax = pos[o], ay = pos[o + 1], az = pos[o + 2];
    const bx = pos[o + 3], by = pos[o + 4], bz = pos[o + 5];
    const cx = pos[o + 6], cy = pos[o + 7], cz = pos[o + 8];
    const y0 = Math.max(1, Math.ceil(Math.min(ay, by, cy) - OY));
    const y1 = Math.min(ny - 2, Math.floor(Math.max(ay, by, cy) - OY));
    const z0 = Math.max(1, Math.ceil(Math.min(az, bz, cz) - OZ));
    const z1 = Math.min(nz - 2, Math.floor(Math.max(az, bz, cz) - OZ));
    if (y0 > y1 || z0 > z1) continue;
    const e1y = by - ay, e1z = bz - az, e2y = cy - ay, e2z = cz - az;
    const det = e1y * e2z - e2y * e1z;
    if (det === 0) continue;
    const id = 1 / det;
    for (let k = z0; k <= z1; k++) {
      const pz = k + OZ - az;
      for (let j = y0; j <= y1; j++) {
        const py = j + OY - ay;
        const w1 = (py * e2z - e2y * pz) * id;
        if (w1 < 0) continue;
        const w2 = (e1y * pz - py * e1z) * id;
        if (w2 < 0 || w1 + w2 > 1) continue;
        const c = j + ny * k;
        (cols[c] || (cols[c] = [])).push(ax + w1 * (bx - ax) + w2 * (cx - ax));
      }
    }
  }

  for (let c = 0; c < cols.length; c++) {
    const l = cols[c];
    if (!l) continue;
    l.sort((a, b) => a - b);
    const j = c % ny, k = (c / ny) | 0;
    const base = nx * (j + ny * k);
    for (let i = 0; i + 1 < l.length; i += 2) {
      const a = Math.max(1, Math.ceil(l[i])), b = Math.min(nx - 2, Math.floor(l[i + 1]));
      for (let x = a; x <= b; x++) vox[base + x] = 1;
    }
  }

  // surface shell
  for (let t = 0; t < T; t++) {
    const o = t * 9;
    const ax = pos[o], ay = pos[o + 1], az = pos[o + 2];
    const ux = pos[o + 3] - ax, uy = pos[o + 4] - ay, uz = pos[o + 5] - az;
    const vx = pos[o + 6] - ax, vy = pos[o + 7] - ay, vz = pos[o + 8] - az;
    const L = Math.sqrt(Math.max(ux * ux + uy * uy + uz * uz, vx * vx + vy * vy + vz * vz,
      (ux - vx) ** 2 + (uy - vy) ** 2 + (uz - vz) ** 2));
    const s = Math.max(1, Math.min(256, Math.ceil(L / 0.5)));
    for (let a = 0; a <= s; a++) {
      for (let b = 0; a + b <= s; b++) {
        const fa = a / s, fb = b / s;
        const x = Math.round(ax + ux * fa + vx * fb);
        const y = Math.round(ay + uy * fa + vy * fb);
        const z = Math.round(az + uz * fa + vz * fb);
        if (x < 1 || y < 1 || z < 1 || x > nx - 2 || y > ny - 2 || z > nz - 2) continue;
        vox[x + nx * (y + ny * z)] = 1;
      }
    }
  }

  // flood fill fluid from the boundary; unreachable fluid is enclosed -> solid
  const seen = new Uint8Array(n);
  const stack = new Int32Array(n);
  let sp = 0;
  const nxy = nx * ny;
  for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
    if (x === 0 || y === 0 || z === 0 || x === nx - 1 || y === ny - 1 || z === nz - 1) {
      const i = x + nx * (y + ny * z);
      if (!vox[i]) { seen[i] = 1; stack[sp++] = i; }
    }
  }
  while (sp > 0) {
    const i = stack[--sp];
    const x = i % nx, y = ((i / nx) | 0) % ny, z = (i / nxy) | 0;
    if (x > 0 && !vox[i - 1] && !seen[i - 1]) { seen[i - 1] = 1; stack[sp++] = i - 1; }
    if (x < nx - 1 && !vox[i + 1] && !seen[i + 1]) { seen[i + 1] = 1; stack[sp++] = i + 1; }
    if (y > 0 && !vox[i - nx] && !seen[i - nx]) { seen[i - nx] = 1; stack[sp++] = i - nx; }
    if (y < ny - 1 && !vox[i + nx] && !seen[i + nx]) { seen[i + nx] = 1; stack[sp++] = i + nx; }
    if (z > 0 && !vox[i - nxy] && !seen[i - nxy]) { seen[i - nxy] = 1; stack[sp++] = i - nxy; }
    if (z < nz - 1 && !vox[i + nxy] && !seen[i + nxy]) { seen[i + nxy] = 1; stack[sp++] = i + nxy; }
  }

  const solid = new Uint8Array(n);
  let count = 0;
  for (let i = 0; i < n; i++) if (!seen[i]) { solid[i] = 1; count++; }

  // frontal area (projection onto the y-z plane), in cells^2
  let frontal = 0;
  for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) {
    const base = nx * (j + ny * k);
    for (let x = 0; x < nx; x++) if (solid[base + x]) { frontal++; break; }
  }

  // planform area (projection onto the x-z plane), in cells^2
  let planform = 0;
  for (let k = 0; k < nz; k++) for (let x = 0; x < nx; x++) {
    for (let j = 0; j < ny; j++) if (solid[x + nx * (j + ny * k)]) { planform++; break; }
  }

  if (ground) for (let k = 0; k < nz; k++) for (let x = 0; x < nx; x++) solid[x + nx * ny * k] = 2;

  return { solid, frontal, planform, count };
}
