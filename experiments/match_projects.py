"""Step 2 check: do Pat's real portfolio photos match the decors each project says it used?

For every photo in catalog/projects/<id>/:
  1. list surfaces with the vision model (cached in experiments/out/projects/, so re-runs are free)
  2. crop each surface, embed it, rank catalog decors (matcher/rank.py)
  3. compare the top-k decor codes with the codes named on the project page

Usage (from the repo root, after catalog/scrape_site.py and catalog/build_index.py):
    export OPENROUTER_API_KEY=...
    uv run --extra ml experiments/match_projects.py
    uv run --extra ml experiments/match_projects.py --models colorhist   # fast, no model download

Writes experiments/results_step2.md and experiments/out/projects/matches.json.

Metrics, per embedding model and mode:
  decor recall@k  share of (project, named decor) pairs found in the top-k of any surface in any
                  photo of that project
  photo hit@k     share of photos where some named decor is in the top-k of some surface
  chance          photo hit@k a random ranking would get with the same number of surfaces
Modes:
  surfaces        VLM boxes, category filter from the material family (the planned app)
  no-category     VLM boxes, ranking over the whole catalog
  whole-photo     no VLM, the full photo as one surface (baseline)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import run_vlm

from catalog.build_index import slug
from matcher.colour import main_lab
from matcher.embed import load_embedder
from matcher.rank import CatalogIndex, rank

CATALOG = ROOT / "catalog"
OUT = ROOT / "experiments" / "out" / "projects"
VLM = "qwen/qwen3-vl-235b-a22b-instruct"
MODES = ("surfaces", "no-category", "whole-photo")
MIN_BOX_SIDE = 40  # ignore slivers narrower than this many pixels


@dataclass(frozen=True)
class PhotoResult:
    project: int
    photo: str
    model: str
    mode: str
    expected: tuple[str, ...]
    surfaces: tuple[tuple[str, tuple[str, ...]], ...]  # (family, top-k codes) per surface


# ---------------------------------------------------------------- pure helpers


def crop(img: Image.Image, box: Iterable[float]) -> Image.Image | None:
    """Crop a 0-1000 normalised box; None if it is too small to say anything about the material."""
    w, h = img.size
    x1, y1, x2, y2 = box
    px = (int(x1 * w / 1000), int(y1 * h / 1000), int(x2 * w / 1000), int(y2 * h / 1000))
    if px[2] - px[0] < MIN_BOX_SIDE or px[3] - px[1] < MIN_BOX_SIDE:
        return None
    return img.crop(px)


def hit(result: PhotoResult, k: int) -> bool:
    return any(code in result.expected for _, top in result.surfaces for code in top[:k])


def found_codes(results: list[PhotoResult], k: int) -> set[tuple[int, str]]:
    return {(r.project, c) for r in results for _, top in r.surfaces for c in top[:k] if c in r.expected}


def chance_hit(n_surfaces: int, n_expected: int, n_codes: int, k: int) -> float:
    """P(some expected code appears) if each surface showed k random distinct codes."""
    miss_one = 1.0
    for i in range(k):
        miss_one *= max(n_codes - n_expected - i, 0) / max(n_codes - i, 1)
    return 1 - miss_one**n_surfaces


def summarise(results: list[PhotoResult], n_codes: int) -> list[dict]:
    rows = []
    for model in sorted({r.model for r in results}):
        for mode in MODES:
            rs = [r for r in results if r.model == model and r.mode == mode]
            if not rs:
                continue
            pairs = {(r.project, c) for r in rs for c in r.expected}
            rows.append(
                {
                    "model": model,
                    "mode": mode,
                    "photos": len(rs),
                    "recall@1": len(found_codes(rs, 1)) / len(pairs),
                    "recall@5": len(found_codes(rs, 5)) / len(pairs),
                    "hit@1": sum(hit(r, 1) for r in rs) / len(rs),
                    "hit@5": sum(hit(r, 5) for r in rs) / len(rs),
                    "chance@5": sum(chance_hit(len(r.surfaces), len(r.expected), n_codes, 5) for r in rs)
                    / len(rs),
                }
            )
    return rows


def render(rows: list[dict], n_projects: int, n_codes: int) -> str:
    lines = [
        "# Step 2 results: portfolio photos vs the decors each project names",
        "",
        (
            f"{n_projects} projects with named decors, {rows[0]['photos'] if rows else 0} photos, "
            f"{n_codes} decor codes in the catalog."
        ),
        "",
        "| Embedding | Mode | Decor recall@1 | Decor recall@5 | Photo hit@1 | Photo hit@5 | Chance hit@5 |",
        "|---|---|---|---|---|---|---|",
    ]
    lines += [
        f"| `{r['model']}` | {r['mode']} | {r['recall@1']:.0%} | {r['recall@5']:.0%} | {r['hit@1']:.0%} "
        f"| {r['hit@5']:.0%} | {r['chance@5']:.0%} |"
        for r in rows
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- side effects


def surfaces_for(photo: Path, project: int, prompt: str, key: str | None) -> list[run_vlm.Surface]:
    """Vision-model surfaces for a photo, from cache when possible."""
    cache = OUT / run_vlm.slug(VLM) / str(project) / f"{photo.stem}.json"
    if cache.exists():
        return run_vlm.parse_surfaces(json.loads(cache.read_text()))
    if not key:
        sys.exit(f"No cached surfaces for {photo} and OPENROUTER_API_KEY is not set.")
    res = run_vlm.call_model(VLM, photo, prompt, key, max_side=1280)
    if not res.ok:
        print(f"  ! {photo.name}: {res.error}", file=sys.stderr)
        return []
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"surfaces": [asdict(s) for s in res.surfaces], "cost_usd": res.cost_usd}))
    return res.surfaces


def match_photo(
    img: Image.Image, surfaces: list[run_vlm.Surface], embedder, index: CatalogIndex, mode: str, k: int
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    if mode == "whole-photo":
        regions = [(None, img)]
    else:
        regions = [(s.family, c) for s in surfaces if (c := crop(img, s.box)) is not None]
    if not regions:
        return ()
    embs = embedder.embed([c for _, c in regions])
    out = []
    for (family, c), emb in zip(regions, embs, strict=True):
        top = rank(
            index, emb, main_lab(c, correct_lighting=True), family, k=k, use_category=mode == "surfaces"
        )
        out.append((family or "", tuple(m.code for m in top)))
    return tuple(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", help="comma-separated; default: every index in catalog/index/")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--report", type=Path, default=ROOT / "experiments" / "results_step2.md")
    args = ap.parse_args()

    projects = [p for p in json.loads((CATALOG / "projects.json").read_text()) if p["decor_codes"]]
    prompt = (ROOT / "experiments" / "prompts" / "surface_list.md").read_text()
    key = os.environ.get("OPENROUTER_API_KEY")
    index_files = (
        [CATALOG / "index" / f"{slug(m)}.npz" for m in args.models.split(",")]
        if args.models
        else sorted((CATALOG / "index").glob("*.npz"))
    )

    photos = [
        (pr, ph)
        for pr in projects
        for ph in sorted((CATALOG / "projects" / str(pr["id"])).glob("*"))
        if ph.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    ]
    surfaces = {ph: surfaces_for(ph, pr["id"], prompt, key) for pr, ph in photos}
    print(f"{len(photos)} photos, {sum(map(len, surfaces.values()))} surfaces")

    results: list[PhotoResult] = []
    n_codes = 0
    for f in index_files:
        index = CatalogIndex.load(f)
        n_codes = len(set(index.codes))
        embedder = load_embedder(index.model)
        for pr, ph in photos:
            img = Image.open(ph).convert("RGB")
            for mode in MODES:
                tops = match_photo(img, surfaces[ph], embedder, index, mode, args.k)
                results.append(
                    PhotoResult(pr["id"], ph.name, index.model, mode, tuple(pr["decor_codes"]), tops)
                )
        print(f"matched with {index.model}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "matches.json").write_text(json.dumps([asdict(r) for r in results], indent=1))
    report = render(summarise(results, n_codes), len(projects), n_codes)
    args.report.write_text(report)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
