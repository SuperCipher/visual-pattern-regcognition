"""Embed every catalog swatch and save a search index per embedding model.

Usage (from the repo root, after catalog/scrape_site.py):
    uv run --extra ml catalog/build_index.py                       # default models
    uv run --extra ml catalog/build_index.py --models colorhist,facebook/dinov2-small

Writes catalog/index/<model-slug>.npz with codes, names, categories, embeddings and colours.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from matcher.colour import main_lab
from matcher.embed import load_embedder
from matcher.rank import CatalogIndex

HERE = Path(__file__).resolve().parent
DEFAULT_MODELS = ["colorhist", "facebook/dinov2-small", "openai/clip-vit-base-patch32"]


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def swatch(path: Path) -> Image.Image:
    """Centre square of the swatch: product photos can have borders or labels at the edges."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    side = int(min(w, h) * 0.8)
    left, top = (w - side) // 2, (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def load_products(catalog_dir: Path) -> list[dict]:
    products = json.loads((catalog_dir / "products.json").read_text())
    present = [p for p in products if (catalog_dir / p["image_file"]).exists()]
    if missing := len(products) - len(present):
        print(f"  ! {missing} product images missing; re-run catalog/scrape_site.py", file=sys.stderr)
    return present


def build(products: list[dict], catalog_dir: Path, model: str) -> CatalogIndex:
    images = [swatch(catalog_dir / p["image_file"]) for p in products]
    embedder = load_embedder(model)
    return CatalogIndex(
        codes=[p["code"] for p in products],
        variants=[p.get("variant", "") for p in products],
        names=[p["name"] for p in products],
        categories=[p["decor_category"] for p in products],
        embeddings=embedder.embed(images).astype(np.float32),
        labs=np.stack([main_lab(im) for im in images]).astype(np.float32),
        model=model,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--catalog", type=Path, default=HERE)
    args = ap.parse_args()

    products = load_products(args.catalog)
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        index = build(products, args.catalog, model)
        dest = args.catalog / "index" / f"{slug(model)}.npz"
        index.save(dest)
        print(f"{model}: {len(index.codes)} swatches, dim {index.embeddings.shape[1]} -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
