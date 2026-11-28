import { describe, expect, it } from "vitest";
import fixtures from "./fixtures.json";
import { type Lab, type Rgb, deltaE, mainLabOfPixels, srgbToLab } from "./colour";
import { type Catalog, type CategoryMode, rank } from "./rank";
import { type Surface, groupSurfaces } from "./group";

const c = fixtures.catalog;
const catalog: Catalog = {
  codes: c.codes,
  variants: c.variants,
  names: c.names,
  categories: c.categories,
  dim: c.dim,
  embeddings: Float32Array.from(c.embeddings),
  labs: Float32Array.from(c.labs),
  hub: Float32Array.from(c.hub),
};

describe("colour matches matcher/colour.py", () => {
  it.each(fixtures.srgbToLab)("srgbToLab($rgb)", ({ rgb, lab }) => {
    srgbToLab(rgb as unknown as Rgb).forEach((v, i) => expect(v).toBeCloseTo(lab[i]!, 6));
  });
  it.each(fixtures.mainLab.map((p, i) => ({ ...p, i })))("mainLab case $i", ({ rgb, lab, labCorrected }) => {
    expect(deltaE(mainLabOfPixels(rgb), lab as unknown as Lab)).toBeLessThan(1e-6);
    expect(deltaE(mainLabOfPixels(rgb, true), labCorrected as unknown as Lab)).toBeLessThan(1e-6);
  });
});

describe("rank matches matcher/rank.py", () => {
  it.each(fixtures.rank.map((r, i) => ({ ...r, i })))("case $i: $family $mode hub=$hubWeight", (r) => {
    const top = rank(catalog, r.embedding, r.lab as unknown as Lab, r.family, {
      mode: r.mode as CategoryMode,
      hubWeight: r.hubWeight,
    });
    expect(top.map((m) => m.code)).toEqual(r.codes);
    top.forEach((m, i) => expect(m.score).toBeCloseTo(r.scores[i]!, 5));
  });
  it("never lists a flip sheet next to its normal sheet", () => {
    const top = rank(catalog, catalog.embeddings.slice(0, c.dim), [50, 0, 0], null, { k: 40, mode: "off" });
    expect(new Set(top.map((m) => m.code)).size).toBe(top.length);
  });
});

describe("groupSurfaces", () => {
  const unit = (...v: number[]) => {
    const n = Math.hypot(...v);
    return Float32Array.from(v.map((x) => x / n));
  };
  const surf = (group: string, family: string, emb: Float32Array, lab: Lab = [50, 0, 0]): Surface => ({
    box: [0, 0, 10, 10],
    family,
    materialGroup: group,
    embedding: emb,
    lab,
  });

  it("keeps the vision model's groups and averages their colour", () => {
    const g = groupSurfaces([surf("A", "wood", unit(1, 0)), surf("B", "paint", unit(0, 1)), surf("A", "wood", unit(1, 0.1), [60, 0, 0])]);
    expect(g.map((x) => x.members)).toEqual([[0, 2], [1]]);
    expect(g[0]!.lab[0]).toBeCloseTo(55);
  });
  it("merges near-identical groups of the same category only", () => {
    const g = groupSurfaces([
      surf("A", "wood", unit(1, 0)),
      surf("B", "wood", unit(1, 0.05)),
      surf("C", "stone", unit(1, 0.02)),
    ]);
    expect(g.map((x) => x.members)).toEqual([[0, 1], [2]]);
  });
  it("does not merge different-looking surfaces", () => {
    expect(groupSurfaces([surf("A", "wood", unit(1, 0)), surf("B", "wood", unit(1, 1))])).toHaveLength(2);
  });
});
