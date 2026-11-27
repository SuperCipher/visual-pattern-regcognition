/** Surface-list request and parsing, ported from experiments/run_vlm.py. Pure: no I/O. */

export const FAMILIES = new Set([
  "plaster", "paint", "wood", "stone", "tile", "mosaic", "wallpaper",
  "metal", "glass", "fabric", "brick", "concrete", "other",
]); // prettier-ignore

export interface VlmSurface {
  box: [number, number, number, number];
  surfaceType: string;
  family: string;
  pattern: string;
  colour: string;
  materialGroup: string;
  confidence: number;
}

export function buildRequest(model: string, prompt: string, dataUrl: string) {
  return {
    model,
    temperature: 0,
    max_tokens: 4000,
    messages: [
      {
        role: "user",
        content: [
          { type: "image_url", image_url: { url: dataUrl } },
          { type: "text", text: prompt },
        ],
      },
    ],
  };
}

/** First JSON object in a reply, tolerating ```json fences and prose around it. */
export function extractJson(text: string): unknown {
  const fenced = /```(?:json)?\s*(\{[\s\S]*?\})\s*```/.exec(text);
  const candidate = fenced ? fenced[1]! : text.slice(text.indexOf("{"), text.lastIndexOf("}") + 1);
  return JSON.parse(candidate);
}

const clamp = (v: unknown) => Math.max(0, Math.min(1000, Number(v)));

/** Validate the model's JSON; malformed entries are dropped. */
export function parseSurfaces(payload: unknown): VlmSurface[] {
  const list = (payload as { surfaces?: unknown })?.surfaces;
  if (!Array.isArray(list)) return [];
  const out: VlmSurface[] = [];
  for (const s of list as Record<string, unknown>[]) {
    const box = s?.box;
    if (!Array.isArray(box) || box.length !== 4) continue;
    const [x1, y1, x2, y2] = box.map(clamp) as [number, number, number, number];
    if ([x1, y1, x2, y2].some(Number.isNaN)) continue;
    const family = String(s.family ?? "other").toLowerCase().trim();
    out.push({
      box: [Math.min(x1, x2), Math.min(y1, y2), Math.max(x1, x2), Math.max(y1, y2)],
      surfaceType: String(s.surface_type ?? "other").toLowerCase(),
      family: FAMILIES.has(family) ? family : "other",
      pattern: String(s.pattern ?? ""),
      colour: String(s.colour ?? ""),
      materialGroup: String(s.material_group ?? "?"),
      confidence: Number(s.confidence ?? 0) || 0,
    });
  }
  return out;
}
