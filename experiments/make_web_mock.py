"""Build web/public/mock/example.json (gitignored) from a review export, for the UI layout phase.

Usage (from the repo root, after experiments/export_review.py):
    uv run experiments/make_web_mock.py --project 2122 --photo 1BC46F9A
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--review", type=Path, default=ROOT / "experiments" / "out" / "review_data.json")
    ap.add_argument("--config", default="google/siglip-base-patch16-224|auto")
    ap.add_argument("--project", type=int, required=True)
    ap.add_argument("--photo", required=True, help="start of the photo file name")
    ap.add_argument("--out", type=Path, default=ROOT / "web" / "public" / "mock" / "example.json")
    args = ap.parse_args()

    data = json.loads(args.review.read_text())
    photo = next(
        p for p in data["photos"] if p["project"] == args.project and p["photo"].startswith(args.photo)
    )
    surfaces = photo["results"][args.config]
    groups: dict[str, list[int]] = {}
    for i, s in enumerate(surfaces):  # the review export has no material groups: group by family
        groups.setdefault(s["family"], []).append(i)

    def decor(code: str, score: float) -> dict:
        d = data["decors"][code]
        return {"code": code, "name": d["name"], "category": d["category"], "url": d["url"],
                "swatch": d["img"], "score": score}  # fmt: skip

    mock = {
        "photo": photo["img"],
        "source": {"title": photo["project_title"], "url": photo["project_url"]},
        "surfaces": [{"box": s["box"], "family": s["family"], "surfaceType": "wall"} for s in surfaces],
        "groups": [
            {
                "family": fam,
                "members": idx,
                "matches": [
                    decor(c, sc)
                    for c, sc in zip(surfaces[idx[0]]["codes"], surfaces[idx[0]]["scores"], strict=True)
                ],
            }
            for fam, idx in groups.items()
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(mock, ensure_ascii=False))
    print(f"{len(surfaces)} surfaces, {len(groups)} groups -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
