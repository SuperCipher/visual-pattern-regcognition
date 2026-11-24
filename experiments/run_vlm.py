# /// script
# requires-python = ">=3.11"
# dependencies = ["pillow>=10"]
# ///
"""Run sample interior photos through vision-language models on OpenRouter.

For every (model, image) pair this saves the parsed surface list, latency, token usage and
cost to experiments/out/<model-slug>/<image>.json, plus an overlay PNG with the boxes drawn.

Usage (from the repo root):
    export OPENROUTER_API_KEY=...            # never commit this
    uv run experiments/run_vlm.py                         # default models, all samples
    uv run experiments/run_vlm.py --models google/gemini-2.5-flash --images experiments/samples
    uv run experiments/run_vlm.py --list-models            # show vision models + prices
    uv run experiments/run_vlm.py --dry-run                # build requests, call nothing
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

API_BASE = "https://openrouter.ai/api/v1"
HERE = Path(__file__).resolve().parent

# Model ids change often: check them with --list-models before a run.
DEFAULT_MODELS = [
    "qwen/qwen3-vl-235b-a22b-instruct",
    "google/gemini-2.5-flash",
]

FAMILIES = {
    "plaster",
    "paint",
    "wood",
    "stone",
    "tile",
    "mosaic",
    "wallpaper",
    "metal",
    "glass",
    "fabric",
    "brick",
    "concrete",
    "other",
}

# One colour per material group, so same-material surfaces share a colour in the overlay.
PALETTE = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#46f0f0", "#f032e6"]


@dataclass(frozen=True)
class Surface:
    """One finished surface as returned by a model, with a normalized 0-1000 box."""

    box: tuple[float, float, float, float]
    surface_type: str
    family: str
    pattern: str
    colour: str
    material_group: str
    confidence: float


@dataclass
class RunResult:
    model: str
    image: str
    ok: bool
    latency_s: float
    cost_usd: float | None
    prompt_tokens: int | None
    completion_tokens: int | None
    surfaces: list[Surface] = field(default_factory=list)
    error: str | None = None
    raw: str | None = None


# ---------------------------------------------------------------- pure helpers


def encode_image(path: Path, max_side: int) -> tuple[str, tuple[int, int]]:
    """Downscale so the long side is <= max_side and return a JPEG data URL + new size."""
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(), img.size


def build_request(model: str, prompt: str, data_url: str) -> dict[str, Any]:
    return {
        "model": model,
        "temperature": 0,
        "max_tokens": 4000,
        "usage": {"include": True},  # OpenRouter returns the exact cost when asked
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }


def extract_json(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of a reply, tolerating ```json fences and prose."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text[text.find("{") : text.rfind("}") + 1]
    return json.loads(candidate)


def clamp(v: float) -> float:
    return max(0.0, min(1000.0, float(v)))


def parse_surfaces(payload: dict[str, Any]) -> list[Surface]:
    """Validate the model's JSON into Surface objects; silently drop malformed entries."""
    out: list[Surface] = []
    for s in payload.get("surfaces", []):
        try:
            x1, y1, x2, y2 = (clamp(v) for v in s["box"])
        except (KeyError, TypeError, ValueError):
            continue
        family = str(s.get("family", "other")).lower().strip()
        out.append(
            Surface(
                box=(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)),
                surface_type=str(s.get("surface_type", "other")).lower(),
                family=family if family in FAMILIES else "other",
                pattern=str(s.get("pattern", "")),
                colour=str(s.get("colour", "")),
                material_group=str(s.get("material_group", "?")),
                confidence=float(s.get("confidence", 0) or 0),
            )
        )
    return out


def slug(model: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")


# ---------------------------------------------------------------- side effects


def http_json(method: str, url: str, body: dict | None, key: str | None, timeout: int = 120) -> dict:
    req = urllib.request.Request(url, method=method)
    req.add_header("Content-Type", "application/json")
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    data = json.dumps(body).encode() if body is not None else None
    with urllib.request.urlopen(req, data=data, timeout=timeout) as resp:
        return json.load(resp)


def call_model(model: str, image: Path, prompt: str, key: str, max_side: int) -> RunResult:
    data_url, _ = encode_image(image, max_side)
    t0 = time.perf_counter()
    try:
        resp = http_json("POST", f"{API_BASE}/chat/completions", build_request(model, prompt, data_url), key)
    except urllib.error.HTTPError as e:
        return RunResult(
            model,
            image.name,
            False,
            time.perf_counter() - t0,
            None,
            None,
            None,
            error=f"HTTP {e.code}: {e.read().decode(errors='replace')[:500]}",
        )
    except (urllib.error.URLError, TimeoutError) as e:
        return RunResult(model, image.name, False, time.perf_counter() - t0, None, None, None, error=str(e))
    latency = time.perf_counter() - t0

    usage = resp.get("usage", {}) or {}
    text = (resp.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    base = {
        "model": model,
        "image": image.name,
        "latency_s": latency,
        "cost_usd": usage.get("cost"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "raw": text,
    }
    try:
        return RunResult(ok=True, surfaces=parse_surfaces(extract_json(text)), **base)
    except (json.JSONDecodeError, ValueError) as e:
        return RunResult(ok=False, error=f"could not parse JSON: {e}", **base)


def draw_overlay(image: Path, surfaces: list[Surface], dest: Path) -> None:
    img = Image.open(image).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)
    groups = sorted({s.material_group for s in surfaces})
    for s in surfaces:
        colour = PALETTE[groups.index(s.material_group) % len(PALETTE)]
        x1, y1, x2, y2 = s.box
        draw.rectangle([x1 * w / 1000, y1 * h / 1000, x2 * w / 1000, y2 * h / 1000], outline=colour, width=4)
        draw.text((x1 * w / 1000 + 6, y1 * h / 1000 + 4), f"{s.material_group}:{s.family}", fill=colour)
    img.save(dest)


def list_models() -> None:
    models = http_json("GET", f"{API_BASE}/models", None, None)["data"]
    for m in models:
        if "image" in (m.get("architecture", {}).get("input_modalities") or []):
            p = m.get("pricing", {})
            print(f"{m['id']:60s} prompt ${p.get('prompt')}/tok  completion ${p.get('completion')}/tok")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS), help="comma-separated OpenRouter ids")
    ap.add_argument("--images", type=Path, default=HERE / "samples", help="folder of .png/.jpg photos")
    ap.add_argument("--prompt", type=Path, default=HERE / "prompts" / "surface_list.md")
    ap.add_argument("--out", type=Path, default=HERE / "out")
    ap.add_argument("--max-side", type=int, default=1280, help="resize long side before upload")
    ap.add_argument("--list-models", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="build requests but do not call the API")
    args = ap.parse_args()

    if args.list_models:
        list_models()
        return 0

    images = sorted(p for p in args.images.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if not images:
        sys.exit(f"No images in {args.images}. Copy the sample photos there first (see README).")
    prompt = args.prompt.read_text()
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    if args.dry_run:
        for img in images:
            data_url, size = encode_image(img, args.max_side)
            print(f"{img.name}: resized to {size}, payload {len(data_url) / 1024:.0f} KB")
        print(f"Would call {len(models)} model(s) x {len(images)} image(s).")
        return 0

    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        sys.exit("Set OPENROUTER_API_KEY first.")

    for model in models:
        out_dir = args.out / slug(model)
        out_dir.mkdir(parents=True, exist_ok=True)
        for img in images:
            res = call_model(model, img, prompt, key, args.max_side)
            (out_dir / f"{img.stem}.json").write_text(json.dumps(asdict(res), indent=2))
            if res.ok:
                draw_overlay(img, res.surfaces, out_dir / f"{img.stem}.overlay.png")
            status = f"{len(res.surfaces)} surfaces" if res.ok else f"FAILED: {res.error}"
            print(f"[{model}] {img.name}: {status} in {res.latency_s:.1f}s, cost ${res.cost_usd}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
