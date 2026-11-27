import { describe, expect, it } from "vitest";
import { extractJson, parseSurfaces } from "./surfaces";

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
