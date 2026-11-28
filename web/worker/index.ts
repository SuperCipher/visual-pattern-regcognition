/**
 * Cloudflare Worker: serves the static site (dist/) and POST /api/match.
 *
 * /api/match takes {image: "data:image/jpeg;base64,...", token: "<turnstile token>"}:
 *   1. Qwen3-VL on OpenRouter lists the surfaces (boxes, material family, material group);
 *   2. the SigLIP service on Modal embeds each surface crop and measures its colour;
 * and returns {surfaces: MatchedSurface[]}. Ranking against the catalog happens in the browser.
 * The OpenRouter key and the Modal token never leave the Worker.
 * Abuse limits: Turnstile (bots), a per-IP rate limit, a 2 MB image cap, and the credit limit
 * Pat sets on the OpenRouter key.
 */

import prompt from "../../experiments/prompts/surface_list.md";
import { type MatchedSurface, type VlmSurface, WHOLE_PHOTO, attachEmbeddings, buildRequest, extractJson, parseSurfaces } from "./surfaces";

export interface Env {
  ASSETS: Fetcher;
  LIMITER: { limit(opts: { key: string }): Promise<{ success: boolean }> };
  OPENROUTER_API_KEY: string;
  OPENROUTER_MODEL: string;
  MODAL_EMBED_URL: string;
  MODAL_WARM_URL: string;
  MODAL_KEY: string;
  MODAL_SECRET: string;
  /** Unset only for local development (wrangler dev): then Turnstile is skipped. */
  TURNSTILE_SECRET?: string;
}

const MAX_IMAGE_CHARS = 2_800_000; // ~2 MB of JPEG as base64

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });

async function verifyTurnstile(secret: string, token: string, ip: string): Promise<boolean> {
  const form = new FormData();
  form.append("secret", secret);
  form.append("response", token);
  form.append("remoteip", ip);
  const res = await fetch("https://challenges.cloudflare.com/turnstile/v0/siteverify", { method: "POST", body: form });
  return ((await res.json()) as { success?: boolean }).success === true;
}

class ApiError extends Error {
  constructor(
    public code: string,
    public status: number,
  ) {
    super(code);
  }
}

async function listSurfaces(env: Env, image: string): Promise<VlmSurface[]> {
  for (let attempt = 0; attempt < 2; attempt++) {
    const res = await fetch("https://openrouter.ai/api/v1/chat/completions", {
      method: "POST",
      headers: { authorization: `Bearer ${env.OPENROUTER_API_KEY}`, "content-type": "application/json" },
      body: JSON.stringify(buildRequest(env.OPENROUTER_MODEL, prompt, image)),
    });
    if (!res.ok) {
      console.error("openrouter", res.status, (await res.text()).slice(0, 300));
      continue;
    }
    const data = (await res.json()) as { choices?: { message?: { content?: string } }[] };
    try {
      return parseSurfaces(extractJson(data.choices?.[0]?.message?.content ?? ""));
    } catch {
      console.error("unparseable reply");
    }
  }
  throw new ApiError("vision_model_failed", 502);
}

const modalHeaders = (env: Env) => ({ "Modal-Key": env.MODAL_KEY, "Modal-Secret": env.MODAL_SECRET });

async function embed(env: Env, image: string, surfaces: VlmSurface[]): Promise<MatchedSurface[]> {
  // The whole photo goes last, as the fallback when no surface crop is usable.
  const boxes = [...surfaces.map((s) => s.box), WHOLE_PHOTO.box];
  const res = await fetch(env.MODAL_EMBED_URL, {
    method: "POST",
    headers: { ...modalHeaders(env), "content-type": "application/json" },
    body: JSON.stringify({ image, boxes }),
  });
  if (!res.ok) {
    console.error("modal", res.status, (await res.text()).slice(0, 300));
    throw new ApiError("embedding_failed", 502);
  }
  const { surfaces: embedded } = (await res.json()) as { surfaces: ({ embedding: number[]; lab: number[] } | null)[] };
  return attachEmbeddings(surfaces, embedded);
}

async function match(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
  const ip = request.headers.get("cf-connecting-ip") ?? "unknown";
  if (!(await env.LIMITER.limit({ key: ip })).success) return json({ error: "rate_limited" }, 429);

  let body: { image?: unknown; token?: unknown };
  try {
    body = await request.json();
  } catch {
    return json({ error: "bad_request" }, 400);
  }
  const image = typeof body.image === "string" ? body.image : "";
  if (!image.startsWith("data:image/jpeg;base64,") || image.length > MAX_IMAGE_CHARS) {
    return json({ error: "bad_image" }, 400);
  }
  if (env.TURNSTILE_SECRET && !(await verifyTurnstile(env.TURNSTILE_SECRET, String(body.token ?? ""), ip))) {
    return json({ error: "bot_check_failed" }, 403);
  }

  // Wake the Modal container now, so its cold start overlaps the vision-model call.
  ctx.waitUntil(fetch(env.MODAL_WARM_URL, { headers: modalHeaders(env) }).catch(() => undefined));
  try {
    const surfaces = await listSurfaces(env, image);
    return json({ surfaces: await embed(env, image, surfaces) });
  } catch (e) {
    if (e instanceof ApiError) return json({ error: e.code }, e.status);
    console.error(e);
    return json({ error: "failed" }, 500);
  }
}

export default {
  async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    const url = new URL(request.url);
    if (url.pathname === "/api/match") {
      return request.method === "POST" ? match(request, env, ctx) : json({ error: "method_not_allowed" }, 405);
    }
    return env.ASSETS.fetch(request);
  },
} satisfies ExportedHandler<Env>;
