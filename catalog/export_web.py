"""Export one catalog index as static files for the web app.

Usage (from the repo root, after catalog/build_index.py with the same --model):
    uv run --extra ml catalog/build_index.py --models onnx:Xenova/siglip-base-patch16-224:quantized
    uv run catalog/export_web.py --model onnx:Xenova/siglip-base-patch16-224:quantized

Writes web/public/catalog/:
    decors.json      {model, dim, decors: [{code, variant, name, category, url, swatch, lab}]}
    embeddings.bin   float16, little-endian, row-major (len(decors), dim), same order as decors
    swatches/<file>  160 px JPEG per decor
The web app must run the same ONNX file the index was built with (see web/src/embed.ts).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from catalog.build_index import load_products, slug
from matcher.rank import CatalogIndex

SWATCH_SIDE = 160


def export(index: CatalogIndex, products: list[dict], catalog_dir: Path, out: Path) -> dict:
    """Write the files; returns the decors.json document. Index rows must follow `products`."""
    if [p["code"] for p in products] != index.codes:
        raise SystemExit("index rows don't match products.json; rebuild the index")
    (out / "swatches").mkdir(parents=True, exist_ok=True)
    decors = []
    for p, lab in zip(products, index.labs, strict=True):
        name = Path(p["image_file"]).stem + ".jpg"
        img = Image.open(catalog_dir / p["image_file"]).convert("RGB")
        img.thumbnail((SWATCH_SIDE, SWATCH_SIDE))
        img.save(out / "swatches" / name, quality=82)
        decors.append({
            "code": p["code"], "variant": p.get("variant", ""), "name": p["name"],
            "category": p["decor_category"], "url": p["url"], "swatch": f"swatches/{name}",
            "lab": [round(float(v), 3) for v in lab],
        })  # fmt: skip
    doc = {"model": index.model, "dim": int(index.embeddings.shape[1]), "decors": decors}
    (out / "decors.json").write_text(json.dumps(doc, ensure_ascii=False, separators=(",", ":")))
    index.embeddings.astype("<f2").tofile(out / "embeddings.bin")
    return doc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="onnx:Xenova/siglip-base-patch16-224:quantized")
    ap.add_argument("--catalog", type=Path, default=ROOT / "catalog")
    ap.add_argument("--out", type=Path, default=ROOT / "web" / "public" / "catalog")
    args = ap.parse_args()

    index = CatalogIndex.load(args.catalog / "index" / f"{slug(args.model)}.npz")
    doc = export(index, load_products(args.catalog), args.catalog, args.out)
    size = (args.out / "embeddings.bin").stat().st_size / 1e6
    print(f"{len(doc['decors'])} decors, dim {doc['dim']} -> {args.out} (embeddings {size:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
