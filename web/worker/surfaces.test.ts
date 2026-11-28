import { describe, expect, it } from "vitest";
import { WHOLE_PHOTO, attachEmbeddings, extractJson, parseSurfaces } from "./surfaces";

describe("extractJson (as run_vlm.extract_json)", () => {
  it("reads fenced JSON", () => expect(extractJson('Here:\n```json\n{"a": 1}\n```')).toEqual({ a: 1 }));
  it("reads JSON wrapped in prose", () => expect(extractJson('Sure {"a": {"b": 2}} done')).toEqual({ a: { b: 2 } }));
  it("throws on no JSON", () => expect(() => extractJson("nothing")).toThrow());
});

describe("parseSurfaces (as run_vlm.parse_surfaces)", () => {
  it("clamps, orders the box, normalises family and drops bad entries", () => {
    const out = parseSurfaces({
      surfaces: [
        { box: [900, 1200, -5, 100], surface_type: "Wall", family: " Wood ", material_group: "A", confidence: 0.9 },
        { box: [0, 0, 10], family: "wood" },
        { box: [0, 0, 10, 10], family: "lava" },
        "junk",
      ],
    });
    expect(out).toHaveLength(2);
    expect(out[0]).toMatchObject({ box: [0, 100, 900, 1000], surfaceType: "wall", family: "wood", materialGroup: "A" });
    expect(out[1]!.family).toBe("other");
  });
  it("returns [] for a payload without surfaces", () => expect(parseSurfaces({})).toEqual([]));
});

describe("attachEmbeddings", () => {
  const e = (x: number) => ({ embedding: [x], lab: [x, 0, 0] });
  const s = (g: string) => ({ ...WHOLE_PHOTO, box: [0, 0, 500, 500] as [number, number, number, number], materialGroup: g });
  it("drops surfaces whose crop was too small", () => {
    const out = attachEmbeddings([s("A"), s("B")], [e(1), null, e(9)]);
    expect(out.map((x) => [x.materialGroup, x.embedding[0]])).toEqual([["A", 1]]);
  });
  it("falls back to the whole photo when nothing is usable", () => {
    const out = attachEmbeddings([s("A")], [null, e(9)]);
    expect(out).toHaveLength(1);
    expect(out[0]).toMatchObject({ box: [0, 0, 1000, 1000], embedding: [9] });
    expect(attachEmbeddings([], [e(7)])[0]!.embedding).toEqual([7]);
  });
});
