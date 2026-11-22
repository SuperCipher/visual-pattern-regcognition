"""Offline checks for the step 1 scripts (no API key needed).

Run from the repo root:  uv run --with pillow python -m unittest experiments/test_scripts.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import evaluate
import run_vlm

LABELS = Path(__file__).resolve().parent / "labels.json"


def fake_run(dir_: Path, image: str, surfaces: list[dict], ok: bool = True) -> None:
    run = {
        "model": "fake/model",
        "image": image,
        "ok": ok,
        "latency_s": 2.0,
        "cost_usd": 0.001,
        "surfaces": surfaces,
    }
    (dir_ / f"{Path(image).stem}.json").write_text(json.dumps(run))


class ParseTests(unittest.TestCase):
    def test_extract_json_handles_fences_and_prose(self) -> None:
        text = 'Sure!\n```json\n{"surfaces": [{"box": [1, 2, 3, 4]}]}\n```\nDone.'
        self.assertEqual(run_vlm.extract_json(text)["surfaces"][0]["box"], [1, 2, 3, 4])

    def test_parse_surfaces_clamps_sorts_and_drops_bad_rows(self) -> None:
        out = run_vlm.parse_surfaces(
            {
                "surfaces": [
                    {"box": [900, 50, 100, 1200], "family": "Wood", "material_group": "A"},
                    {"box": "nope"},
                    {"box": [0, 0, 10, 10], "family": "unobtainium"},
                ]
            }
        )
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].box, (100.0, 50.0, 900.0, 1000.0))
        self.assertEqual(out[0].family, "wood")
        self.assertEqual(out[1].family, "other")


class EvaluateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.labels = evaluate.load_labels(LABELS)

    def test_labels_cover_all_eleven_photos(self) -> None:
        self.assertEqual(sorted(self.labels), [f"{i:02d}.png" for i in range(1, 12)])

    def test_perfect_predictions_score_100_percent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            for img, labs in self.labels.items():
                fake_run(
                    d,
                    img,
                    [{"box": list(l.box), "family": l.families[0], "material_group": l.group} for l in labs],
                )
            s = evaluate.score_model(d, self.labels)
        self.assertEqual((s.recall, s.family_acc, s.group_agree), (1.0, 1.0, 1.0))
        self.assertEqual(s.photos_complete, 11)
        self.assertEqual(s.extra_boxes, 0)

    def test_swapped_boxes_are_flagged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            for img, labs in self.labels.items():
                fake_run(
                    d,
                    img,
                    [
                        {
                            "box": list(evaluate.swap_xy(l.box)),
                            "family": l.families[0],
                            "material_group": l.group,
                        }
                        for l in labs
                    ],
                )
            s = evaluate.score_model(d, self.labels)
        self.assertGreater(s.swapped_recall, s.recall + 0.1)
        self.assertIn("x/y swapped", evaluate.render([s]))

    def test_empty_predictions_score_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            for img in self.labels:
                fake_run(d, img, [], ok=False)
            s = evaluate.score_model(d, self.labels)
        self.assertEqual((s.recall, s.photos_complete, s.failures), (0.0, 0, 11))


if __name__ == "__main__":
    unittest.main()
