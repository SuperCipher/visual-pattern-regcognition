/** Cloudflare Turnstile (bot check for the API). No site key configured = skipped (local dev). */

const SITE_KEY = (import.meta.env.VITE_TURNSTILE_SITE_KEY as string | undefined) ?? "";

interface TurnstileApi {
  render(el: HTMLElement, opts: Record<string, unknown>): string;
  reset(id: string): void;
}
declare global {
  interface Window {
    turnstile?: TurnstileApi;
  }
}

let script: Promise<TurnstileApi> | null = null;

function loadScript(): Promise<TurnstileApi> {
  script ??= new Promise((resolve, reject) => {
    const s = document.createElement("script");
    s.src = "https://challenges.cloudflare.com/turnstile/v0/api.js?render=explicit";
    s.async = true;
    s.onload = () => (window.turnstile ? resolve(window.turnstile) : reject(new Error("turnstile missing")));
    s.onerror = () => reject(new Error("turnstile failed to load"));
    document.head.append(s);
  });
  return script;
}

/** A fresh token. Shows a challenge only when Cloudflare is unsure (most visitors see nothing). */
export async function getToken(container: HTMLElement): Promise<string> {
  if (!SITE_KEY) return "";
  const api = await loadScript();
  return new Promise((resolve, reject) => {
    container.replaceChildren();
    api.render(container, {
      sitekey: SITE_KEY,
      appearance: "interaction-only",
      callback: (token: string) => resolve(token),
      "error-callback": () => reject(new Error("bot check failed")),
    });
  });
}
