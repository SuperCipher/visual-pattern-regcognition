/**
 * Cloudflare Worker: serves the static site (dist/) and POST /api/surfaces.
 *
 * /api/surfaces takes {image: "data:image/jpeg;base64,...", token: "<turnstile token>"} and
 * returns {surfaces: VlmSurface[]}. The OpenRouter key never leaves the Worker.
 * Abuse limits: Turnstile (bots), a per-IP rate limit, a 2 MB image cap, and the credit limit
 * Pat sets on the OpenRouter key.
 */

import prompt from "../../experiments/prompts/surface_list.md";
import { buildRequest, extractJson, parseSurfaces } from "./surfaces";

export interface Env {
  ASSETS: Fetcher;
  LIMITER: { limit(opts: { key: string }): Promise<{ success: boolean }> };
  OPENROUTER_API_KEY: string;
  OPENROUTER_MODEL: string;
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

async function surfaces(request: Request, env: Env): Promise<Response> {
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
      return json({ surfaces: parseSurfaces(extractJson(data.choices?.[0]?.message?.content ?? "")) });
    } catch {
      console.error("unparseable reply");
    }
  }
  return json({ error: "vision_model_failed" }, 502);
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);
    if (url.pathname === "/api/surfaces") {
      return request.method === "POST" ? surfaces(request, env) : json({ error: "method_not_allowed" }, 405);
    }
    return env.ASSETS.fetch(request);
  },
} satisfies ExportedHandler<Env>;
