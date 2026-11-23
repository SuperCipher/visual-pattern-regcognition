# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Scrape the Blackbone product catalog and portfolio photos from blackbone.co.th.

Uses the site's public WooCommerce Store API and WordPress REST API (no HTML scraping):
  - /wp-json/wc/store/v1/products  -> catalog/products.json + catalog/images/<code>.jpg
  - /wp-json/wp/v2/project         -> catalog/projects.json + catalog/projects/<id>/*.jpg
    Each portfolio project names the decor codes it used, which gives real
    "photo -> correct product" pairs for testing the matcher in step 2.
  - the PDF catalog                -> catalog/blackbone-catalog.pdf

Usage (from the repo root):
    uv run catalog/scrape_site.py              # metadata + images
    uv run catalog/scrape_site.py --no-images  # metadata only
Re-running skips files that already exist.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SITE = "https://blackbone.co.th"
PDF_URL = f"{SITE}/wp-content/uploads/2026/02/PDF-Blackbone-Catalog-V2.pdf"
HERE = Path(__file__).resolve().parent
USER_AGENT = "visual-pattern-recognition catalog sync"
DELAY_S = 0.3  # be gentle with the site

# Decor codes look like BBST-1014, BBWNT-0861, BBHG-6202 (one product has the typo BNL-1259).
CODE_RE = re.compile(r"\bB[A-Z]{1,6}-\d{3,5}[A-Z]?\b")

# Website category slug -> decor category used by the matcher.
CATEGORY_MAP = {
    "architectural-single-stone": "stone",
    "bookmatch-stone": "stone",
    "natural-woodgrains": "wood",
    "solid-colour": "solid_colour",
    "metallic": "metal",
    "fabric": "fabric",
}


@dataclass(frozen=True)
class Product:
    code: str
    name: str
    decor_category: str
    site_categories: tuple[str, ...]
    url: str
    image_url: str
    image_file: str


@dataclass(frozen=True)
class Project:
    id: int
    title: str
    url: str
    decor_codes: tuple[str, ...]
    image_urls: tuple[str, ...]


# ---------------------------------------------------------------- pure helpers


def find_code(*texts: str) -> str | None:
    """First decor code in any of the texts (image filename first, then product name)."""
    for t in texts:
        if m := CODE_RE.search(t.upper()):
            return m.group(0)
    return None


def decor_category(slugs: list[str]) -> str:
    return next((CATEGORY_MAP[s] for s in slugs if s in CATEGORY_MAP), "other")


def parse_product(raw: dict[str, Any]) -> Product | None:
    images = raw.get("images") or []
    if not images:
        return None
    src = images[0]["src"]
    code = find_code(Path(urllib.parse.urlparse(src).path).name, raw.get("name", "")) or f"ID{raw['id']}"
    slugs = [c["slug"] for c in raw.get("categories", [])]
    return Product(
        code=code,
        name=html.unescape(raw.get("name", "")).strip(),
        decor_category=decor_category(slugs),
        site_categories=tuple(slugs),
        url=raw.get("permalink", ""),
        image_url=src,
        image_file=f"images/{code}{Path(urllib.parse.urlparse(src).path).suffix.lower() or '.jpg'}",
    )


def full_size(url: str) -> str:
    """WordPress thumbnails end in -1024x682.jpg; strip that to get the original upload."""
    return re.sub(r"-\d{2,4}x\d{2,4}(?=\.\w+$)", "", url)


def parse_project(raw: dict[str, Any]) -> Project:
    content = raw.get("content", {}).get("rendered", "")
    text = html.unescape(re.sub(r"<[^>]+>", " ", content))
    srcs = re.findall(r'<img[^>]+src="([^"]+)"', content)
    return Project(
        id=raw["id"],
        title=html.unescape(raw.get("title", {}).get("rendered", "")).strip(),
        url=raw.get("link", ""),
        decor_codes=tuple(dict.fromkeys(CODE_RE.findall(text.upper()))),
        image_urls=tuple(dict.fromkeys(full_size(s) for s in srcs)),
    )


# ---------------------------------------------------------------- side effects


def get(url: str) -> tuple[bytes, dict[str, str]]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read(), dict(resp.headers)


def get_all(endpoint: str) -> list[dict[str, Any]]:
    """Fetch every page of a paginated WordPress endpoint."""
    items: list[dict[str, Any]] = []
    page, pages = 1, 1
    while page <= pages:
        sep = "&" if "?" in endpoint else "?"
        body, headers = get(f"{SITE}{endpoint}{sep}per_page=100&page={page}")
        items += json.loads(body)
        pages = int(headers.get("X-WP-TotalPages") or headers.get("x-wp-totalpages") or 1)
        page += 1
        time.sleep(DELAY_S)
    return items


def download(url: str, dest: Path) -> bool:
    """Download unless the file exists. Returns True if a new file was written."""
    if dest.exists():
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        body, _ = get(url)
    except Exception as e:  # noqa: BLE001 - keep going, report at the end
        print(f"  ! {url}: {e}", file=sys.stderr)
        return False
    dest.write_bytes(body)
    time.sleep(DELAY_S)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=HERE)
    ap.add_argument("--no-images", action="store_true", help="metadata only")
    args = ap.parse_args()
    out: Path = args.out

    products = [p for raw in get_all("/wp-json/wc/store/v1/products") if (p := parse_product(raw))]
    dupes = len(products) - len({p.code for p in products})
    (out / "products.json").write_text(
        json.dumps([asdict(p) for p in products], indent=2, ensure_ascii=False)
    )
    print(f"products: {len(products)} ({dupes} duplicate codes)")

    projects = [parse_project(raw) for raw in get_all("/wp-json/wp/v2/project?_fields=id,link,title,content")]
    (out / "projects.json").write_text(
        json.dumps([asdict(p) for p in projects], indent=2, ensure_ascii=False)
    )
    print(f"projects: {len(projects)}, {sum(len(p.image_urls) for p in projects)} photos")

    if args.no_images:
        return 0
    new = sum(download(p.image_url, out / p.image_file) for p in products)
    print(f"product images: {new} new")
    for pr in projects:
        new = sum(
            download(u, out / "projects" / str(pr.id) / Path(urllib.parse.urlparse(u).path).name)
            for u in pr.image_urls
        )
        print(f"project {pr.id} ({', '.join(pr.decor_codes) or 'no codes'}): {new} new photos")
    download(PDF_URL, out / "blackbone-catalog.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
