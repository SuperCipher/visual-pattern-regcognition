/**
 * Colour features, ported from matcher/colour.py. Keep the two in step: the catalog's Lab
 * colours are computed in Python, the photo crops' in the browser.
 */

export type Lab = readonly [number, number, number];
export type Rgb = readonly [number, number, number];

const M = [
  [0.4124564, 0.3575761, 0.1804375],
  [0.2126729, 0.7151522, 0.072175],
  [0.0193339, 0.119192, 0.9503041],
] as const;
const WHITE = [0.95047, 1.0, 1.08883] as const;
const EPS = (6 / 29) ** 3;

/** sRGB 0-255 to CIELAB (D65). */
export function srgbToLab([r, g, b]: Rgb): Lab {
  const lin = [r, g, b].map((v) => {
    const c = v / 255;
    return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  }) as [number, number, number];
  const f = M.map((row, i) => {
    const xyz = (row[0] * lin[0] + row[1] * lin[1] + row[2] * lin[2]) / WHITE[i]!;
    return xyz > EPS ? Math.cbrt(xyz) : xyz / (3 * (6 / 29) ** 2) + 4 / 29;
  }) as [number, number, number];
  return [116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])];
}

/** Pull channel means partly toward grey to remove a lighting cast (strength 0 = no change). */
export function grayWorld(px: Float64Array, strength = 0.5): Float64Array {
  const n = px.length / 3;
  const means = [0, 0, 0];
  for (let i = 0; i < px.length; i++) means[i % 3]! += px[i]!;
  for (let c = 0; c < 3; c++) means[c]! /= n;
  const grey = (means[0]! + means[1]! + means[2]!) / 3;
  const scale = means.map((m) => 1 + strength * (grey / Math.max(m, 1e-6) - 1));
  return px.map((v, i) => Math.min(255, Math.max(0, v * scale[i % 3]!)));
}

/** numpy.percentile with its default linear interpolation. */
export function percentile(values: ArrayLike<number>, q: number): number {
  const s = Float64Array.from(values).sort();
  if (s.length === 0) return NaN;
  const pos = ((s.length - 1) * q) / 100;
  const lo = Math.floor(pos);
  const hi = Math.min(lo + 1, s.length - 1);
  return s[lo]! + (s[hi]! - s[lo]!) * (pos - lo);
}

/**
 * Median Lab colour of RGB pixels (flat r,g,b,r,g,b... in 0-255), ignoring the darkest and
 * brightest 5% by mean brightness. Same as main_lab() after its 64x64 resize.
 */
export function mainLabOfPixels(rgb: ArrayLike<number>, correctLighting = false): Lab {
  let px = Float64Array.from(rgb);
  if (correctLighting) px = grayWorld(px);
  const n = px.length / 3;
  const lum = new Float64Array(n);
  for (let i = 0; i < n; i++) lum[i] = (px[3 * i]! + px[3 * i + 1]! + px[3 * i + 2]!) / 3;
  const lo = percentile(lum, 5);
  const hi = percentile(lum, 95);
  const keep: number[][] = [[], [], []];
  for (let i = 0; i < n; i++) {
    if (lum[i]! >= lo && lum[i]! <= hi) for (let c = 0; c < 3; c++) keep[c]!.push(px[3 * i + c]!);
  }
  const src = keep[0]!.length ? keep : [0, 1, 2].map((c) => Array.from({ length: n }, (_, i) => px[3 * i + c]!));
  return srgbToLab(src.map((ch) => percentile(ch, 50)) as unknown as Rgb);
}

/** CIE76 colour difference. */
export function deltaE(a: Lab, b: Lab): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}
