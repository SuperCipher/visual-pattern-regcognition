/** The real pipeline: photo -> surfaces (Worker + Qwen) -> crops -> SigLIP -> group -> rank. */

import type { VlmSurface } from "../worker/surfaces";
import { loadCatalog } from "./catalog";
import { embedCanvases, loadEmbedder } from "./embed";
import { mainLabOfPixels } from "./match/colour";
import { type Surface, groupSurfaces } from "./match/group";
import { rank } from "./match/rank";
import { getToken } from "./turnstile";
import type { Box, PipelineResult, Progress } from "./types";

const MAX_SIDE = 1280; // same as step 1/2 (run_vlm --max-side)
const MIN_BOX_SIDE = 40; // same as match_projects.MIN_BOX_SIDE
const WHOLE: Box = [0, 0, 1000, 1000];

export class PipelineError extends Error {
  constructor(public code: string) {
    super(code);
  }
}

async function toCanvas(file: Blob): Promise<HTMLCanvasElement> {
  const bmp = await createImageBitmap(file, { imageOrientation: "from-image" });
  const scale = Math.min(1, MAX_SIDE / Math.max(bmp.width, bmp.height));
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(bmp.width * scale);
  canvas.height = Math.round(bmp.height * scale);
  canvas.getContext("2d")!.drawImage(bmp, 0, 0, canvas.width, canvas.height);
  bmp.close();
  return canvas;
}

function crop(src: HTMLCanvasElement, [x1, y1, x2, y2]: Box): HTMLCanvasElement | null {
  const [w, h] = [src.width, src.height];
  const px = [x1 * w, y1 * h, x2 * w, y2 * h].map((v) => Math.trunc(v / 1000)) as [number, number, number, number];
  if (px[2] - px[0] < MIN_BOX_SIDE || px[3] - px[1] < MIN_BOX_SIDE) return null;
  const c = document.createElement("canvas");
  c.width = px[2] - px[0];
  c.height = px[3] - px[1];
  c.getContext("2d")!.drawImage(src, px[0], px[1], c.width, c.height, 0, 0, c.width, c.height);
  return c;
}

/** Same as main_lab(correct_lighting=True): 64x64 resize, grey-world, trimmed median. */
function labOf(c: HTMLCanvasElement) {
  const small = document.createElement("canvas");
  small.width = small.height = 64;
  const ctx = small.getContext("2d", { willReadFrequently: true })!;
  ctx.drawImage(c, 0, 0, 64, 64);
  const rgba = ctx.getImageData(0, 0, 64, 64).data;
  const rgb = new Uint8ClampedArray(64 * 64 * 3);
  for (let i = 0, j = 0; i < rgba.length; i += 4) {
    rgb[j++] = rgba[i]!;
    rgb[j++] = rgba[i + 1]!;
    rgb[j++] = rgba[i + 2]!;
  }
  return mainLabOfPixels(rgb, true);
}

async function fetchSurfaces(image: string, turnstileBox: HTMLElement): Promise<VlmSurface[]> {
  const token = await getToken(turnstileBox).catch(() => {
    throw new PipelineError("bot_check_failed");
  });
  const res = await fetch("/api/surfaces", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ image, token }),
  });
  const body = (await res.json().catch(() => ({}))) as { surfaces?: VlmSurface[]; error?: string };
  if (!res.ok || !body.surfaces) throw new PipelineError(body.error ?? `http_${res.status}`);
  return body.surfaces;
}

export async function runPipeline(
  file: Blob,
  turnstileBox: HTMLElement,
  onProgress: (p: Progress) => void,
): Promise<PipelineResult> {
  const canvas = await toCanvas(file);
  const photo = canvas.toDataURL("image/jpeg", 0.88);

  // The vision model call and the model/catalog downloads run in parallel.
  onProgress({ step: "model", fraction: 0 });
  const ready = Promise.all([loadEmbedder((f) => onProgress({ step: "model", fraction: f })), loadCatalog()]);
  const vlm = fetchSurfaces(photo, turnstileBox);
  const [, catalog] = await ready;
  onProgress({ step: "surfaces" });
  let found = await vlm;

  onProgress({ step: "matching" });
  let crops = found.map((s) => crop(canvas, s.box));
  found = found.filter((_, i) => crops[i]);
  crops = crops.filter(Boolean);
  if (!found.length) {
    // Never "no match": with no usable surface, match the whole photo.
    found = [{ box: [...WHOLE], surfaceType: "other", family: "other", pattern: "", colour: "", materialGroup: "A", confidence: 0 }];
    crops = [canvas];
  }
  const embeddings = await embedCanvases(crops as HTMLCanvasElement[]);
  const surfaces: Surface[] = found.map((s, i) => ({
    box: s.box,
    family: s.family,
    materialGroup: s.materialGroup,
    embedding: embeddings[i]!,
    lab: labOf(crops[i]!),
  }));

  const groups = groupSurfaces(surfaces).map((g) => ({
    family: g.family,
    members: g.members,
    matches: rank(catalog, g.embedding, g.lab, g.family, { k: 5, mode: "auto" }).map((m) => {
      const d = catalog.decors[catalog.codes.indexOf(m.code)]!;
      return { code: m.code, name: m.name, category: m.category, url: d.url, swatch: d.swatch, score: m.score };
    }),
  }));
  return {
    photo,
    surfaces: found.map((s) => ({ box: s.box, family: s.family, surfaceType: s.surfaceType })),
    groups,
  };
}
