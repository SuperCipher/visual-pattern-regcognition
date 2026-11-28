/** Load the decor catalog written by catalog/export_web.py. */

import type { Catalog } from "./match/rank";

export interface DecorInfo {
  code: string;
  variant: string;
  name: string;
  category: string;
  url: string;
  swatch: string;
  lab: [number, number, number];
}

export interface WebCatalog extends Catalog {
  decors: DecorInfo[];
  model: string;
}

/** IEEE 754 half -> float (embeddings.bin is float16 to halve the download). */
export function halfToFloat(h: number): number {
  const s = h & 0x8000 ? -1 : 1;
  const e = (h >> 10) & 0x1f;
  const f = h & 0x3ff;
  if (e === 0) return s * 2 ** -14 * (f / 1024);
  if (e === 31) return f ? NaN : s * Infinity;
  return s * 2 ** (e - 15) * (1 + f / 1024);
}

export function decodeFloat16(buf: ArrayBuffer): Float32Array {
  const view = new DataView(buf);
  const out = new Float32Array(buf.byteLength / 2);
  for (let i = 0; i < out.length; i++) out[i] = halfToFloat(view.getUint16(2 * i, true));
  return out;
}

let cached: Promise<WebCatalog> | null = null;

export function loadCatalog(base = `${import.meta.env.BASE_URL}catalog/`): Promise<WebCatalog> {
  cached ??= (async () => {
    const [meta, bin] = await Promise.all([
      fetch(`${base}decors.json`).then((r) => r.json() as Promise<{ model: string; dim: number; decors: DecorInfo[] }>),
      fetch(`${base}embeddings.bin`).then((r) => r.arrayBuffer()),
    ]);
    const d = meta.decors;
    return {
      model: meta.model,
      decors: d.map((x) => ({ ...x, swatch: `${base}${x.swatch}` })),
      codes: d.map((x) => x.code),
      variants: d.map((x) => x.variant),
      names: d.map((x) => x.name),
      categories: d.map((x) => x.category),
      dim: meta.dim,
      embeddings: decodeFloat16(bin),
      labs: Float32Array.from(d.flatMap((x) => x.lab)),
    };
  })();
  cached.catch(() => (cached = null));
  return cached;
}
