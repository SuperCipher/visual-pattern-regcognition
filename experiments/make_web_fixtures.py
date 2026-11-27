"""Write web/src/match/fixtures.json: inputs and Python outputs the TypeScript port must reproduce.

Usage (from the repo root):  uv run experiments/make_web_fixtures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from matcher.colour import main_lab, srgb_to_lab
from matcher.rank import CatalogIndex, rank

OUT = ROOT / "web" / "src" / "match" / "fixtures.json"
CATEGORIES = ["wood", "stone", "solid_colour", "metal", "fabric", "other"]
FAMILIES = ["wood", "paint", "plaster", "metal", "glass", "stone", None]


def main() -> int:
    rng = np.random.default_rng(7)
    n, dim = 40, 16
    emb = rng.normal(size=(n, dim)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    codes = [f"BB-{i // 2 if i < 8 else i:03d}" for i in range(n)]  # first 8 rows: sheet + flip pairs
    index = CatalogIndex(
        codes=codes,
        variants=["flip" if i < 8 and i % 2 else "" for i in range(n)],
        names=[f"Decor {i}" for i in range(n)],
        categories=[CATEGORIES[i % len(CATEGORIES)] for i in range(n)],
        embeddings=emb,
        labs=(rng.uniform([20, -30, -30], [90, 30, 30], size=(n, 3))).astype(np.float32),
        model="fixture",
        hub=rng.uniform(0.2, 0.6, size=n).astype(np.float32),
    )
    cases = []
    for q in range(len(FAMILIES) * 2):
        family = FAMILIES[q % len(FAMILIES)]
        e = (emb[rng.integers(n)] + 0.6 * rng.normal(size=dim)).astype(np.float32)
        e /= np.linalg.norm(e)
        lab = rng.uniform([20, -30, -30], [90, 30, 30]).astype(np.float32)
        for mode in ("hard", "soft", "off", "auto"):
            for hub_weight in (0.0, 0.5):
                top = rank(
                    index, e, lab.astype(np.float64), family, k=5, category_mode=mode, hub_weight=hub_weight
                )
                cases.append({
                    "embedding": e.tolist(), "lab": lab.tolist(), "family": family, "mode": mode,
                    "hubWeight": hub_weight, "codes": [m.code for m in top], "scores": [m.score for m in top],
                })  # fmt: skip

    colours = [[255, 255, 255], [0, 0, 0], [200, 40, 40], [12, 180, 90], [128, 128, 128], [240, 220, 180]]
    pixels = []
    for seed in range(4):
        r = np.random.default_rng(100 + seed)
        px = np.clip(r.normal(r.uniform(40, 210, 3), 25, size=(64, 64, 3)), 0, 255).astype(np.uint8)
        img = Image.fromarray(px)
        pixels.append({
            "rgb": px.reshape(-1).tolist(),
            "lab": main_lab(img).tolist(), "labCorrected": main_lab(img, correct_lighting=True).tolist(),
        })  # fmt: skip

    fixture = {
        "catalog": {
            "codes": index.codes, "variants": index.variants, "names": index.names,
            "categories": index.categories, "dim": dim, "embeddings": emb.reshape(-1).tolist(),
            "labs": index.labs.reshape(-1).tolist(), "hub": index.hub.tolist(),
        },
        "rank": cases,
        "srgbToLab": [{"rgb": c, "lab": srgb_to_lab(c).tolist()} for c in colours],
        "mainLab": pixels,
    }  # fmt: skip
    OUT.write_text(json.dumps(fixture))
    print(f"{len(cases)} rank cases, {len(pixels)} colour cases -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
