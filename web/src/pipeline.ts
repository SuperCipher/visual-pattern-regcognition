/**
 * The real pipeline: photo -> Worker (Qwen surfaces + SigLIP on Modal) -> group -> rank.
 * Grouping and ranking run here, against the catalog files from catalog/export_web.py.
 */

import type { MatchedSurface } from "../worker/surfaces";
import { loadCatalog } from "./catalog";
import type { Lab } from "./match/colour";
import { type Surface, groupSurfaces } from "./match/group";
import { rank } from "./match/rank";
import { getToken } from "./turnstile";
import type { PipelineResult, Progress } from "./types";

const MAX_SIDE = 1280; // same as steps 1 and 2

export class PipelineError extends Error {
  constructor(public code: string) {
    super(code);
  }
}

/** Downscale (respecting EXIF rotation) and encode as JPEG, as run_vlm.encode_image does. */
async function toJpeg(file: Blob): Promise<string> {
  const bmp = await createImageBitmap(file, { imageOrientation: "from-image" });
  const scale = Math.min(1, MAX_SIDE / Math.max(bmp.width, bmp.height));
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(bmp.width * scale);
  canvas.height = Math.round(bmp.height * scale);
  canvas.getContext("2d")!.drawImage(bmp, 0, 0, canvas.width, canvas.height);
  bmp.close();
  return canvas.toDataURL("image/jpeg", 0.88);
}

async function fetchMatches(image: string, turnstileBox: HTMLElement): Promise<MatchedSurface[]> {
  const token = await getToken(turnstileBox).catch(() => {
    throw new PipelineError("bot_check_failed");
  });
  const res = await fetch("/api/match", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ image, token }),
  });
  const body = (await res.json().catch(() => ({}))) as { surfaces?: MatchedSurface[]; error?: string };
  if (!res.ok || !body.surfaces) throw new PipelineError(body.error ?? `http_${res.status}`);
  return body.surfaces;
}

export async function runPipeline(
  file: Blob,
  turnstileBox: HTMLElement,
  onProgress: (p: Progress) => void,
): Promise<PipelineResult> {
  onProgress({ step: "surfaces" });
  const photo = await toJpeg(file);
  const [found, catalog] = await Promise.all([fetchMatches(photo, turnstileBox), loadCatalog()]);
  if (!found.length) throw new PipelineError("no_surfaces");

  onProgress({ step: "matching" });
  const surfaces: Surface[] = found.map((s) => ({
    box: s.box,
    family: s.family,
    materialGroup: s.materialGroup,
    embedding: Float32Array.from(s.embedding),
    lab: s.lab as Lab,
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
