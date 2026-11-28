import { expect, it } from "vitest";
import { decodeFloat16, halfToFloat } from "./catalog";

it("decodes IEEE half floats", () => {
  expect(halfToFloat(0x3c00)).toBe(1);
  expect(halfToFloat(0xc000)).toBe(-2);
  expect(halfToFloat(0x3555)).toBeCloseTo(0.33325, 5);
  expect(halfToFloat(0x0001)).toBeCloseTo(5.96e-8, 9);
  expect(halfToFloat(0x7c00)).toBe(Infinity);
  const buf = new Uint16Array([0x3c00, 0xb800]).buffer;
  expect(Array.from(decodeFloat16(buf))).toEqual([1, -0.5]);
});
