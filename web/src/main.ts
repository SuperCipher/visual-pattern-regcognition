/**
 * Blackbone Decor Finder: upload -> processing -> results.
 * Uploads run the real pipeline; "Try an example" (and ?mock) replays a saved result at no API cost.
 */

import "./style.css";
import { CONTACT_URL, GROUP_COLOURS, NO_HPL_EQUIVALENT } from "./config";
import { type Lang, STRINGS, detectLang } from "./i18n";
import { runMock } from "./mock";
import type { PipelineResult, Progress, Step } from "./types";

type View =
  | { kind: "upload" }
  | { kind: "processing"; photo: string; progress: Progress }
  | { kind: "results"; result: PipelineResult }
  | { kind: "error"; code: string };

const app = document.querySelector<HTMLDivElement>("#app")!;
let lang: Lang = detectLang();
let view: View = { kind: "upload" };
/** Which alternative is featured per group (index into its matches). */
let featured: number[] = [];
let litGroup: number | null = null;
/** ?mock: uploads also replay the saved example (layout work without API calls). */
const MOCK = new URLSearchParams(location.search).has("mock");
let hasExample = false;

const t = () => STRINGS[lang];
const esc = (s: unknown) =>
  String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]!);
const safeHref = (u: string) => (/^https:\/\//.test(u) ? u : "#");
const safeImg = (u: string) => (/^(data:image\/|https:\/\/|\/|swatches\/)/.test(u) ? u : "");
const colour = (i: number) => GROUP_COLOURS[i % GROUP_COLOURS.length]!;
const familyName = (f: string) => t().family[f] ?? f;

function header(): string {
  const b = (l: Lang, label: string) => `<button type="button" data-lang="${l}" aria-pressed="${lang === l}">${label}</button>`;
  return `<header class="wrap top">
    <a class="brand" href="./">Blackbone <span>Decor Finder</span></a>
    <div class="lang" role="group" aria-label="Language">${b("en", "EN")}${b("th", "ไทย")}</div>
  </header>`;
}

function uploadView(): string {
  const s = t();
  const strip = ["#8a5a35", "#c9b8a3", "#2b2b2b", "#d9d6cf", "#a88b5d", "#5d6b6a"].map((c) => `<i style="background:${c}"></i>`).join("");
  return `<main class="wrap hero">
    <div>
      <h1>${esc(s.title)}</h1>
      <p class="muted">${esc(s.lead)}</p>
    </div>
    <div class="drop" id="drop">
      <div class="swatch-strip" aria-hidden="true">${strip}</div>
      <div class="actions">
        <label class="btn primary" for="file">${esc(s.choose)}</label>
        <label class="btn" for="camera">${esc(s.takePhoto)}</label>
        ${hasExample ? `<button class="btn" type="button" id="example">${esc(s.example)}</button>` : ""}
      </div>
      <input id="file" type="file" accept="image/*" hidden />
      <input id="camera" type="file" accept="image/*" capture="environment" hidden />
      <span class="muted">${esc(s.drop)}</span>
      <p class="fine">${esc(s.tips)}<br />${esc(s.privacy)}</p>
      ${MOCK ? `<p class="note">${esc(s.previewNote)}</p>` : ""}
    </div>
  </main>`;
}

function photoFrame(photo: string, result: PipelineResult | null, scanning: boolean): string {
  const boxes = (result?.groups ?? [])
    .flatMap((g, gi) =>
      g.members.map((m) => {
        const [x1, y1, x2, y2] = result!.surfaces[m]!.box;
        const geo = `x="${x1}" y="${y1}" width="${x2 - x1}" height="${y2 - y1}"`;
        return `<g data-group="${gi}" class="${litGroup === gi ? "lit" : ""}">
          <rect class="fill" ${geo} fill="${colour(gi)}"/><rect class="out" ${geo} stroke="${colour(gi)}"/></g>`;
      }),
    )
    .join("");
  return `<div class="frame ${litGroup !== null ? "dim" : ""}">
    <img src="${esc(safeImg(photo))}" alt="" />
    <svg viewBox="0 0 1000 1000" preserveAspectRatio="none" aria-hidden="true">${boxes}</svg>
    ${scanning ? '<div class="scan"></div>' : ""}
  </div>`;
}

function processingView(v: Extract<View, { kind: "processing" }>): string {
  const s = t();
  const order: Step[] = ["surfaces", "matching"];
  const at = order.indexOf(v.progress.step);
  const label = { surfaces: s.stepSurfaces, matching: s.stepMatching };
  const items = order
    .map((step, i) => `<li class="${i < at ? "done" : i === at ? "active" : ""}">${esc(label[step])}</li>`)
    .join("");
  return `<main class="wrap layout">${photoFrame(v.photo, null, true)}
    <section class="steps" aria-live="polite"><ol>${items}</ol></section></main>`;
}

function groupCard(r: PipelineResult, gi: number): string {
  const s = t();
  const g = r.groups[gi]!;
  const pick = featured[gi] ?? 0;
  const best = g.matches[pick] ?? g.matches[0]!;
  const others = g.matches
    .map((m, i) => ({ m, i }))
    .filter(({ i }) => i !== pick)
    .slice(0, 4)
    .map(
      ({ m, i }) => `<button class="alt" type="button" data-group="${gi}" data-pick="${i}" aria-pressed="false" title="${esc(m.name)}">
        <div class="chip"><img src="${esc(safeImg(m.swatch))}" alt="" loading="lazy" /></div>
        <span class="code">${esc(m.code)}</span></button>`,
    )
    .join("");
  const caveat = NO_HPL_EQUIVALENT.has(g.family)
    ? `<p class="caveat">${esc(s.noEquivalent(familyName(g.family).toLowerCase()))}</p>`
    : "";
  return `<article class="card ${litGroup === gi ? "lit" : ""}" data-group="${gi}" style="--group:${colour(gi)}">
    <div class="card-head"><span class="dot"></span><strong>${esc(s.material(gi + 1))} · ${esc(familyName(g.family))}</strong>
      <span class="muted">${esc(s.surfaces(g.members.length))}</span></div>
    <div class="best">
      <div class="chip"><img src="${esc(safeImg(best.swatch))}" alt="${esc(best.name)}" /></div>
      <div class="best-info">
        <span class="label">${esc(s.bestMatch)}</span>
        <span class="code">${esc(best.code)}</span>
        <span class="name">${esc(best.name)}</span>
        <div class="actions">
          <a class="btn small primary" href="${esc(safeHref(best.url))}" target="_blank" rel="noopener">${esc(s.view)}</a>
          <a class="btn small" href="${esc(safeHref(CONTACT_URL))}" target="_blank" rel="noopener">${esc(s.ask)}</a>
        </div>
      </div>
    </div>
    ${caveat}
    ${others ? `<span class="others-label">${esc(s.others)}</span><div class="others">${others}</div>` : ""}
  </article>`;
}

function resultsView(r: PipelineResult): string {
  const s = t();
  return `<main class="wrap layout">${photoFrame(r.photo, r, false)}
    <section class="results">
      <div class="results-head"><h2>${esc(s.materials(r.groups.length))}</h2>
        <button class="btn small" type="button" id="again">${esc(s.again)}</button></div>
      ${r.groups.map((_, gi) => groupCard(r, gi)).join("")}
    </section></main>`;
}

function errorView(code: string): string {
  const s = t();
  const body = code === "rate_limited" ? s.errorBusy : code === "bot_check_failed" ? s.errorBot : s.errorBody;
  return `<main class="wrap error"><h2>${esc(s.errorTitle)}</h2><p class="muted">${esc(body)}</p>
    <button class="btn primary" type="button" id="again">${esc(s.again)}</button></main>`;
}

function render(): void {
  document.documentElement.lang = lang;
  const body =
    view.kind === "upload"
      ? uploadView()
      : view.kind === "processing"
        ? processingView(view)
        : view.kind === "results"
          ? resultsView(view.result)
          : errorView(view.code);
  app.innerHTML = `${header()}${body}<footer class="wrap">© Blackbone HPL · ${esc(t().privacy)}</footer>`;
}

function setView(v: View): void {
  view = v;
  render();
}

async function start(photo: string, run: (p: (x: Progress) => void) => Promise<PipelineResult>): Promise<void> {
  featured = [];
  litGroup = null;
  setView({ kind: "processing", photo, progress: { step: "surfaces" } });
  try {
    const result = await run((progress) => {
      if (view.kind === "processing") setView({ ...view, progress });
    });
    setView({ kind: "results", result });
  } catch (e) {
    console.error(e);
    const code = (e as { code?: unknown })?.code;
    setView({ kind: "error", code: typeof code === "string" ? code : "failed" });
  }
}

function readFile(file: File | undefined): void {
  if (!file || !file.type.startsWith("image/")) return;
  const box = document.querySelector<HTMLElement>("#turnstile")!;
  void start(URL.createObjectURL(file), async (p) =>
    MOCK ? runMock(p) : (await import("./pipeline")).runPipeline(file, box, p),
  );
}

async function startExample(): Promise<void> {
  try {
    const res = await fetch(`${import.meta.env.BASE_URL}mock/example.json`);
    const { photo } = (await res.json()) as PipelineResult;
    void start(photo, runMock);
  } catch {
    setView({ kind: "error", code: "failed" });
  }
}

app.addEventListener("click", (e) => {
  const el = e.target as HTMLElement;
  const langBtn = el.closest<HTMLButtonElement>("[data-lang]");
  if (langBtn) {
    lang = langBtn.dataset.lang as Lang;
    try {
      localStorage.setItem("lang", lang);
    } catch {
      /* storage blocked */
    }
    return render();
fetch(`${import.meta.env.BASE_URL}mock/example.json`, { method: "HEAD" })
  .then((r) => {
    hasExample = r.ok && (r.headers.get("content-type") ?? "").includes("json");
    if (hasExample && view.kind === "upload") render();
  })
  .catch(() => {});
  }
  if (el.closest("#example")) return void startExample();
  if (el.closest("#again")) return setView({ kind: "upload" });
  const alt = el.closest<HTMLButtonElement>(".alt");
  if (alt) {
    featured[Number(alt.dataset.group)] = Number(alt.dataset.pick);
    return render();
fetch(`${import.meta.env.BASE_URL}mock/example.json`, { method: "HEAD" })
  .then((r) => {
    hasExample = r.ok && (r.headers.get("content-type") ?? "").includes("json");
    if (hasExample && view.kind === "upload") render();
  })
  .catch(() => {});
  }
  const card = el.closest<HTMLElement>(".card");
  if (card && !el.closest("a")) {
    const gi = Number(card.dataset.group);
    litGroup = litGroup === gi ? null : gi;
    render();
fetch(`${import.meta.env.BASE_URL}mock/example.json`, { method: "HEAD" })
  .then((r) => {
    hasExample = r.ok && (r.headers.get("content-type") ?? "").includes("json");
    if (hasExample && view.kind === "upload") render();
  })
  .catch(() => {});
  }
});

app.addEventListener("change", (e) => {
  const input = e.target as HTMLInputElement;
  if (input.type === "file") readFile(input.files?.[0]);
});

app.addEventListener("dragover", (e) => {
  e.preventDefault();
  document.querySelector("#drop")?.classList.add("over");
});
app.addEventListener("dragleave", () => document.querySelector("#drop")?.classList.remove("over"));
app.addEventListener("drop", (e) => {
  e.preventDefault();
  readFile(e.dataTransfer?.files[0]);
});

render();
fetch(`${import.meta.env.BASE_URL}mock/example.json`, { method: "HEAD" })
  .then((r) => {
    hasExample = r.ok && (r.headers.get("content-type") ?? "").includes("json");
    if (hasExample && view.kind === "upload") render();
  })
  .catch(() => {});
