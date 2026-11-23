"""Offline tests for the matcher (no model downloads: uses the colour-histogram embedder).

Run from the repo root:  uv run python -m unittest matcher/test_matcher.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import match_projects as mp
import run_vlm

from catalog import build_index
from matcher.colour import delta_e, gray_world, main_lab, srgb_to_lab
from matcher.embed import ColourHistEmbedder
from matcher.rank import CatalogIndex, rank


def solid(rgb: tuple[int, int, int], size: int = 80) -> Image.Image:
    return Image.new("RGB", (size, size), rgb)


def toy_index() -> CatalogIndex:
    swatches = {
        ("W-1", "", "wood"): (150, 100, 60),
        ("W-1", "flip", "wood"): (150, 100, 60),
        ("W-2", "", "wood"): (90, 60, 40),
        ("S-1", "", "solid_colour"): (200, 40, 40),
        ("S-2", "", "solid_colour"): (40, 40, 200),
    }
    imgs = [solid(c) for c in swatches.values()]
    return CatalogIndex(
        codes=[k[0] for k in swatches],
        variants=[k[1] for k in swatches],
        names=[k[0] for k in swatches],
        categories=[k[2] for k in swatches],
        embeddings=ColourHistEmbedder().embed(imgs),
        labs=np.stack([main_lab(i) for i in imgs]),
        model="colorhist",
    )


class ColourTests(unittest.TestCase):
    def test_lab_reference_points(self) -> None:
        np.testing.assert_allclose(srgb_to_lab([255, 255, 255]), [100, 0, 0], atol=0.05)
        np.testing.assert_allclose(srgb_to_lab([0, 0, 0]), [0, 0, 0], atol=0.05)
        self.assertGreater(srgb_to_lab([255, 0, 0])[1], 70)  # red has strong +a

    def test_gray_world_reduces_warm_cast(self) -> None:
        warm = np.array([[220.0, 180.0, 120.0]] * 4)
        corrected = gray_world(warm, strength=1.0)
        self.assertLess(np.ptp(corrected[0]), np.ptp(warm[0]))
        np.testing.assert_allclose(gray_world(warm, strength=0.0), warm)

    def test_delta_e(self) -> None:
        self.assertAlmostEqual(float(delta_e(np.zeros(3), np.array([3.0, 4.0, 0.0]))), 5.0)


class RankTests(unittest.TestCase):
    def setUp(self) -> None:
        self.index = toy_index()
        self.emb = ColourHistEmbedder()

    def rank_colour(self, rgb, family, **kw):
        img = solid(rgb)
        return rank(self.index, self.emb.embed([img])[0], main_lab(img), family, **kw)

    def test_paint_maps_to_solid_colours_and_ranks_by_colour(self) -> None:
        top = self.rank_colour((190, 50, 50), "paint")
        self.assertEqual([m.code for m in top], ["S-1", "S-2"])

    def test_flip_sheet_never_listed_twice(self) -> None:
        codes = [m.code for m in self.rank_colour((150, 100, 60), "wood")]
        self.assertEqual(codes, ["W-1", "W-2"])

    def test_without_category_searches_everything(self) -> None:
        top = self.rank_colour((150, 100, 60), "paint", use_category=False, k=10)
        self.assertEqual(top[0].code, "W-1")
        self.assertEqual(len(top), 4)

    def test_unknown_family_falls_back_to_whole_catalog(self) -> None:
        self.assertEqual(len(self.rank_colour((0, 0, 0), "mosaic", k=10)), 4)

    def test_index_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "i.npz"
            self.index.save(path)
            loaded = CatalogIndex.load(path)
        self.assertEqual(loaded.codes, self.index.codes)
        np.testing.assert_allclose(loaded.embeddings, self.index.embeddings)


class PipelineTests(unittest.TestCase):
    def test_build_index_from_scraped_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cat = Path(tmp)
            (cat / "images").mkdir()
            solid((10, 200, 10), 200).save(cat / "images" / "BBD-1.jpg")
            products = [
                {"code": "BBD-1", "variant": "", "name": "Green", "decor_category": "solid_colour",
                 "image_file": "images/BBD-1.jpg"},
                {"code": "BBD-2", "variant": "", "name": "Gone", "decor_category": "solid_colour",
                 "image_file": "images/BBD-2.jpg"},
            ]  # fmt: skip
            (cat / "products.json").write_text(json.dumps(products))
            loaded = build_index.load_products(cat)
            index = build_index.build(loaded, cat, "colorhist")
        self.assertEqual(index.codes, ["BBD-1"])  # missing image skipped
        self.assertEqual(index.embeddings.shape, (1, 64))

    def test_match_photo_modes(self) -> None:
        img = Image.new("RGB", (1000, 1000), (150, 100, 60))
        img.paste((200, 40, 40), (500, 0, 1000, 1000))
        surfaces = [
            run_vlm.Surface((0, 0, 500, 1000), "wall", "wood", "", "", "A", 0.9),
            run_vlm.Surface((500, 0, 1000, 1000), "wall", "paint", "", "", "B", 0.9),
            run_vlm.Surface((0, 0, 10, 10), "wall", "wood", "", "", "C", 0.9),  # too small, dropped
        ]
        index, emb = toy_index(), ColourHistEmbedder()
        tops = mp.match_photo(img, surfaces, emb, index, "surfaces", 5)
        self.assertEqual([(f, t[0]) for f, t in tops], [("wood", "W-1"), ("paint", "S-1")])
        self.assertEqual(len(mp.match_photo(img, surfaces, emb, index, "whole-photo", 5)), 1)

    def test_metrics(self) -> None:
        r1 = mp.PhotoResult(1, "a.jpg", "m", "surfaces", ("X",), (("wood", ("Y", "X")),))
        r2 = mp.PhotoResult(1, "b.jpg", "m", "surfaces", ("X",), (("wood", ("Y", "Z")),))
        self.assertTrue(mp.hit(r1, 5))
        self.assertFalse(mp.hit(r1, 1))
        rows = mp.summarise([r1, r2], n_codes=100)
        self.assertEqual((rows[0]["recall@5"], rows[0]["hit@5"], rows[0]["recall@1"]), (1.0, 0.5, 0.0))
        self.assertAlmostEqual(mp.chance_hit(1, 1, 100, 5), 0.05)


try:
    import torch
    import transformers
except ImportError:  # the ml extra is optional
    torch = transformers = None


@unittest.skipIf(torch is None, "needs: uv run --extra ml")
class HFEmbedderTests(unittest.TestCase):
    """Random tiny models, so no download: checks both output styles are turned into vectors."""

    def embed_with(self, model) -> np.ndarray:
        from matcher.embed import HFEmbedder

        e = HFEmbedder.__new__(HFEmbedder)
        e.model, e.torch = model.eval(), torch
        return e._forward(torch.rand(2, 3, 32, 32))

    def test_clip_uses_projected_features(self) -> None:
        vision = {"hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 1, "num_attention_heads": 2,
                  "image_size": 32, "patch_size": 8}  # fmt: skip
        text = {"hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 1, "num_attention_heads": 2}
        cfg = transformers.CLIPConfig(vision_config=vision, text_config=text, projection_dim=16)
        self.assertEqual(tuple(self.embed_with(transformers.CLIPModel(cfg)).shape), (2, 16))

    def test_dinov2_uses_cls_token(self) -> None:
        cfg = transformers.Dinov2Config(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            image_size=32,
            patch_size=8,
        )
        self.assertEqual(tuple(self.embed_with(transformers.Dinov2Model(cfg)).shape), (2, 32))


if __name__ == "__main__":
    unittest.main()
