"""Offline tests for scrape_site.py, using trimmed copies of real API responses.

Run from the repo root:  uv run python -m unittest catalog/test_scrape_site.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import scrape_site as s

PRODUCT = {
    "id": 2274,
    "name": "HPL Tuscan Travertine Luxury Design",
    "permalink": "https://blackbone.co.th/product/hpl-tuscan-travertine-luxury-design/",
    "images": [{"src": "https://blackbone.co.th/wp-content/uploads/2026/07/BBST-1014.jpg"}],
    "categories": [{"slug": "architectural-single-stone"}],
}

PROJECT = {
    "id": 2292,
    "link": "https://blackbone.co.th/project/waris/",
    "title": {"rendered": "WARIS WELLNESS CLINIC &amp; spa"},
    "content": {
        "rendered": "<p>Featured Materials: BBWNT-0861 (Natural Wood Grain) BBST-1014 (Travertine)</p>"
        '<img src="https://blackbone.co.th/wp-content/uploads/2026/11/IMG_9475_watermarked-1024x682.jpg">'
        '<img src="https://blackbone.co.th/wp-content/uploads/2026/11/IMG_9475_watermarked-300x200.jpg">'
        "<p>again BBST-1014</p>"
    },
}


class ScrapeParseTests(unittest.TestCase):
    def test_product_code_from_image_filename(self) -> None:
        p = s.parse_product(PRODUCT)
        self.assertEqual(p.code, "BBST-1014")
        self.assertEqual(p.decor_category, "stone")
        self.assertEqual(p.image_file, "images/BBST-1014.jpg")

    def test_code_typo_and_fallbacks(self) -> None:
        self.assertEqual(s.find_code("BNL-1259-Flip-.jpg"), "BNL-1259")
        self.assertEqual(s.find_code("bbmt-8808.jpg"), "BBMT-8808")
        self.assertIsNone(s.find_code("IMG_9475.jpg"))
        raw = {**PRODUCT, "images": [{"src": "https://x/IMG_1.jpg"}], "name": "No code here"}
        self.assertEqual(s.parse_product(raw).code, "ID2274")

    def test_product_without_image_is_skipped(self) -> None:
        self.assertIsNone(s.parse_product({**PRODUCT, "images": []}))

    def test_unknown_category_maps_to_other(self) -> None:
        self.assertEqual(s.decor_category(["uncategorized"]), "other")
        self.assertEqual(s.decor_category(["uncategorized", "fabric"]), "fabric")

    def test_project_codes_and_full_size_images_deduplicated(self) -> None:
        pr = s.parse_project(PROJECT)
        self.assertEqual(pr.title, "WARIS WELLNESS CLINIC & spa")
        self.assertEqual(pr.decor_codes, ("BBWNT-0861", "BBST-1014"))
        self.assertEqual(
            pr.image_urls, ("https://blackbone.co.th/wp-content/uploads/2026/11/IMG_9475_watermarked.jpg",)
        )


if __name__ == "__main__":
    unittest.main()
