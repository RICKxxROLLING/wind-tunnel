const clamp01 = (t) => Math.min(1, Math.max(0, t));

export function turbo(t) {
  t = clamp01(t);
  const r = 0.13572138 + t * (4.6153926 + t * (-42.66032258 + t * (132.13108234 + t * (-152.94239396 + t * 59.28637943))));
  const g = 0.09140261 + t * (2.19418839 + t * (4.84296658 + t * (-14.18503333 + t * (4.27729857 + t * 2.82956604))));
  const b = 0.1066733 + t * (12.64194608 + t * (-60.58204836 + t * (110.36276771 + t * (-89.90310912 + t * 27.34824973))));
  return [clamp01(r), clamp01(g), clamp01(b)];
}

function ramp(stops, t) {
  t = clamp01(t) * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(t)), f = t - i;
  const a = stops[i], b = stops[i + 1];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

const COOLWARM = [[0.23, 0.3, 0.75], [0.55, 0.69, 0.99], [0.87, 0.87, 0.87], [0.96, 0.6, 0.48], [0.71, 0.02, 0.15]];
const INFERNO = [[0, 0, 0.02], [0.26, 0.04, 0.41], [0.58, 0.15, 0.4], [0.87, 0.32, 0.23], [0.99, 0.65, 0.04], [0.99, 1, 0.65]];

export const coolwarm = (t) => ramp(COOLWARM, t);
export const inferno = (t) => ramp(INFERNO, t);
export const CMAPS = [turbo, coolwarm, inferno];

// sRGB -> linear lookup table (three.js treats vertex colors as linear)
export function makeLUT(fn, size = 256) {
  const lut = new Float32Array(size * 3);
  for (let i = 0; i < size; i++) {
    const c = fn(i / (size - 1));
    for (let k = 0; k < 3; k++) lut[i * 3 + k] = Math.pow(c[k], 2.2);
  }
  return lut;
}

const glslRamp = (name, stops) => `vec3 ${name}(float t) {
  t = clamp(t, 0.0, 1.0) * ${(stops.length - 1).toFixed(1)};
${stops.slice(0, -1).map((s, i) => `  if (t <= ${(i + 1).toFixed(1)}) return mix(vec3(${s.join(',')}), vec3(${stops[i + 1].join(',')}), t - ${i.toFixed(1)});`).join('\n')}
  return vec3(${stops[stops.length - 1].join(',')});
}`;

export const GLSL_CMAPS = `
vec3 turbo(float t) {
  t = clamp(t, 0.0, 1.0);
  float r = 0.13572138 + t * (4.6153926 + t * (-42.66032258 + t * (132.13108234 + t * (-152.94239396 + t * 59.28637943))));
  float g = 0.09140261 + t * (2.19418839 + t * (4.84296658 + t * (-14.18503333 + t * (4.27729857 + t * 2.82956604))));
  float b = 0.1066733 + t * (12.64194608 + t * (-60.58204836 + t * (110.36276771 + t * (-89.90310912 + t * 27.34824973))));
  return clamp(vec3(r, g, b), 0.0, 1.0);
}
${glslRamp('coolwarm', COOLWARM)}
${glslRamp('inferno', INFERNO)}
vec3 cmap(int m, float t) {
  if (m == 1) return coolwarm(t);
  if (m == 2) return inferno(t);
  return turbo(t);
}
`;
