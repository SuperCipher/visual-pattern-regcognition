"""Bundle a matching run into one JSON file for the visual review page.

Reads experiments/out/projects/matches.json (from match_projects.py) plus the scraped catalog,
and writes experiments/out/review_data.json with small JPEG thumbnails inlined as data URIs,
so the page needs no access to the photos or the website.

Usage (from the repo root):
    uv run experiments/export_review.py
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import match_projects as mp

PHOTO_SIDE = 720
SWATCH_SIDE = 160


def data_uri(path: Path, side: int) -> str:
    img = Image.open(path).convert("RGB")
    img.thumbnail((side, side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=78)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def summary_rows(raw: list[dict], n_codes: int) -> list[dict]:
    results = [
        mp.PhotoResult(
            r["project"],
            r["photo"],
            r["model"],
            r["mode"],
            tuple(r["expected"]),
            tuple(
                mp.SurfaceMatch(s["family"], tuple(s["box"]), tuple(s["codes"]), tuple(s["scores"]))
                for s in r["surfaces"]
            ),
        )
        for r in raw
    ]
    return mp.summarise(results, n_codes)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matches", type=Path, default=mp.OUT / "matches.json")
    ap.add_argument("--out", type=Path, default=ROOT / "experiments" / "out" / "review_data.json")
    ap.add_argument("--catalog", type=Path, default=ROOT / "catalog")
    args = ap.parse_args()

    raw = json.loads(args.matches.read_text())
    catalog: Path = args.catalog
    products = json.loads((catalog / "products.json").read_text())
    projects = {p["id"]: p for p in json.loads((catalog / "projects.json").read_text())}

    by_photo: dict[tuple[int, str], dict] = {}
    for r in raw:
        key = (r["project"], r["photo"])
        if key not in by_photo:
            pr = projects[r["project"]]
            by_photo[key] = {
                "project": r["project"],
                "project_title": pr["title"],
                "project_url": pr["url"],
                "photo": r["photo"],
                "named": pr["decor_codes"],
                "expected": r["expected"],
                "img": data_uri(catalog / "projects" / str(r["project"]) / r["photo"], PHOTO_SIDE),
                "results": {},
            }
        by_photo[key]["results"][f"{r['model']}|{r['mode']}"] = r["surfaces"]

    used = {c for r in raw for s in r["surfaces"] for c in s["codes"]} | {
        c for p in by_photo.values() for c in p["expected"]
    }
    decors: dict[str, dict] = {}
    for p in products:  # first entry per code wins, so the normal sheet beats its Flip
        if p["code"] in used and p["code"] not in decors and (catalog / p["image_file"]).exists():
            decors[p["code"]] = {
                "name": p["name"],
                "category": p["decor_category"],
                "url": p["url"],
                "img": data_uri(catalog / p["image_file"], SWATCH_SIDE),
            }

    n_codes = len({p["code"] for p in products})
    bundle = {
        "configs": sorted({f"{r['model']}|{r['mode']}" for r in raw}),
        "summary": summary_rows(raw, n_codes),
        "photos": sorted(by_photo.values(), key=lambda p: (p["project"], p["photo"])),
        "decors": decors,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(bundle, separators=(",", ":"), ensure_ascii=False))
    size = args.out.stat().st_size / 1e6
    print(
        f"{len(bundle['photos'])} photos, {len(decors)} decors, {len(bundle['configs'])} configs -> {args.out} ({size:.1f} MB)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
