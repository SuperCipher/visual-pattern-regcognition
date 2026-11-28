/** Rank catalog decors for one surface. Ported from matcher/rank.py: keep the two in step. */

import { type Lab, deltaE } from "./colour";

/** Vision-model material family -> HPL decor category. Others search the whole catalog. */
export const FAMILY_TO_CATEGORY: Readonly<Record<string, string>> = {
  wood: "wood",
  stone: "stone",
  concrete: "stone",
  paint: "solid_colour",
  plaster: "solid_colour",
  metal: "metal",
  fabric: "fabric",
  wallpaper: "fabric",
};

/** How much colour distance counts against a decor, by the decor's own category. */
export const COLOUR_WEIGHT: Readonly<Record<string, number>> = { solid_colour: 1.0, metal: 0.3, fabric: 0.3 };
export const DEFAULT_COLOUR_WEIGHT = 0.15;
export const DELTA_E_SCALE = 50;
export const SOFT_CATEGORY_BONUS = 0.1;

/** "auto": hard filter where the vision model labels reliably, soft head start for plaster. */
export const FAMILY_MODE: Readonly<Record<string, "hard" | "soft">> = { plaster: "soft" };
export const DEFAULT_FAMILY_MODE = "hard";

export type CategoryMode = "hard" | "soft" | "off" | "auto";

export interface Catalog {
  codes: string[];
  variants: string[];
  names: string[];
  categories: string[];
  /** Row-major (n, dim), L2-normalised. */
  embeddings: Float32Array;
  dim: number;
  /** Row-major (n, 3). */
  labs: Float32Array;
  /** Optional hub score per row (see hub_scores in Python). */
  hub?: Float32Array;
}

export interface Match {
  code: string;
  variant: string;
  name: string;
  category: string;
  score: number;
}

export interface RankOptions {
  k?: number;
  mode?: CategoryMode;
  hubWeight?: number;
}

/** Top-k distinct decor codes (a Flip sheet never shows next to its normal sheet). */
export function rank(
  cat: Catalog,
  embedding: ArrayLike<number>,
  lab: Lab,
  family: string | null,
  { k = 5, mode = "auto", hubWeight = 0 }: RankOptions = {},
): Match[] {
  let m: CategoryMode = mode;
  if (m === "auto") m = FAMILY_MODE[family ?? ""] ?? DEFAULT_FAMILY_MODE;
  const category = m !== "off" ? FAMILY_TO_CATEGORY[family ?? ""] : undefined;
  const n = cat.codes.length;
  const anyInCat = category !== undefined && cat.categories.includes(category);

  const score = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let dot = 0;
    const off = i * cat.dim;
    for (let j = 0; j < cat.dim; j++) dot += cat.embeddings[off + j]! * embedding[j]!;
    const decorLab: Lab = [cat.labs[3 * i]!, cat.labs[3 * i + 1]!, cat.labs[3 * i + 2]!];
    const weight = COLOUR_WEIGHT[cat.categories[i]!] ?? DEFAULT_COLOUR_WEIGHT;
    let s = dot - (weight * deltaE(decorLab, lab)) / DELTA_E_SCALE;
    if (hubWeight && cat.hub) s -= hubWeight * cat.hub[i]!;
    const inCat = category !== undefined && cat.categories[i] === category;
    if (m === "soft" && inCat) s += SOFT_CATEGORY_BONUS;
    if (m === "hard" && anyInCat && !inCat) s = -Infinity;
    score[i] = s;
  }

  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => score[b]! - score[a]!);
  const out: Match[] = [];
  for (const i of order) {
    if (!Number.isFinite(score[i]!) || out.length === k) break;
    if (out.some((x) => x.code === cat.codes[i])) continue;
    out.push({
      code: cat.codes[i]!,
      variant: cat.variants[i]!,
      name: cat.names[i]!,
      category: cat.categories[i]!,
      score: score[i]!,
    });
  }
  return out;
}
