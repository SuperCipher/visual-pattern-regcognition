/**
 * Group surfaces that use the same material, so each material gets one card.
 *
 * 1. The vision model already labels surfaces with a material group ("A", "B", ...).
 * 2. Groups it split but that look the same are merged: same decor category and near-identical
 *    mean embeddings (e.g. one oak wall labelled twice).
 * Each group is then ranked once, on its members' mean embedding and mean colour.
 */

import { FAMILY_TO_CATEGORY } from "./rank";
import type { Lab } from "./colour";

export const MERGE_COSINE = 0.92;

export interface Surface {
  box: readonly [number, number, number, number];
  family: string;
  materialGroup: string;
  embedding: Float32Array;
  lab: Lab;
}

export interface Group {
  /** Indexes into the input surfaces. */
  members: number[];
  /** Most common family among members (ties: the first seen). */
  family: string;
  /** L2-normalised mean of the members' embeddings. */
  embedding: Float32Array;
  lab: Lab;
}

function normalise(v: Float32Array): Float32Array {
  const norm = Math.hypot(...v) || 1;
  return v.map((x) => x / norm);
}

function cosine(a: Float32Array, b: Float32Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i]! * b[i]!;
  return s;
}

function summarise(surfaces: readonly Surface[], members: number[]): Group {
  const dim = surfaces[members[0]!]!.embedding.length;
  const sum = new Float32Array(dim);
  const lab = [0, 0, 0];
  const counts = new Map<string, number>();
  for (const i of members) {
    const s = surfaces[i]!;
    s.embedding.forEach((x, j) => (sum[j]! += x));
    s.lab.forEach((x, j) => (lab[j]! += x / members.length));
    counts.set(s.family, (counts.get(s.family) ?? 0) + 1);
  }
  const family = [...counts.entries()].reduce((best, e) => (e[1] > best[1] ? e : best))[0];
  return { members, family, embedding: normalise(sum), lab: lab as unknown as Lab };
}

const categoryOf = (family: string) => FAMILY_TO_CATEGORY[family] ?? `family:${family}`;

export function groupSurfaces(surfaces: readonly Surface[], mergeCosine = MERGE_COSINE): Group[] {
  const byLabel = new Map<string, number[]>();
  surfaces.forEach((s, i) => {
    const key = s.materialGroup || `#${i}`; // unlabelled surfaces start on their own
    byLabel.set(key, [...(byLabel.get(key) ?? []), i]);
  });
  let groups = [...byLabel.values()].map((m) => summarise(surfaces, m));

  // Greedy merge of the most similar pair until no pair passes the threshold.
  for (;;) {
    let best: [number, number, number] | null = null;
    for (let a = 0; a < groups.length; a++) {
      for (let b = a + 1; b < groups.length; b++) {
        const ga = groups[a]!;
        const gb = groups[b]!;
        if (categoryOf(ga.family) !== categoryOf(gb.family)) continue;
        const c = cosine(ga.embedding, gb.embedding);
        if (c >= mergeCosine && (!best || c > best[2])) best = [a, b, c];
      }
    }
    if (!best) break;
    const [a, b] = best;
    const merged = summarise(surfaces, [...groups[a]!.members, ...groups[b]!.members].sort((x, y) => x - y));
    groups = groups.filter((_, i) => i !== a && i !== b).concat(merged);
  }
  return groups.sort((x, y) => x.members[0]! - y.members[0]!);
}
