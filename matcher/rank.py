"""Rank catalog decors for one detected surface."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from matcher.colour import delta_e

# Material family from the vision model -> HPL decor category on the website.
# Painted/plastered walls are recreated with solid colours; tile, mosaic and brick have no HPL
# equivalent here, so they fall back to the whole catalog (Pat: always show the closest look).
FAMILY_TO_CATEGORY = {
    "wood": "wood",
    "stone": "stone",
    "concrete": "stone",
    "paint": "solid_colour",
    "plaster": "solid_colour",
    "metal": "metal",
    "fabric": "fabric",
    "wallpaper": "fabric",
}

# How much colour distance counts against a decor, per category. Solid colours are judged
# almost only on colour; patterned decors mostly on the embedding.
COLOUR_WEIGHT = {"solid_colour": 1.0, "metal": 0.3, "fabric": 0.3}
DEFAULT_COLOUR_WEIGHT = 0.15
DELTA_E_SCALE = 50.0  # a deltaE of 50 (very different colours) costs 1.0 of cosine score


@dataclass(frozen=True)
class CatalogIndex:
    codes: list[str]
    variants: list[str]
    names: list[str]
    categories: list[str]
    embeddings: np.ndarray  # (n, d), L2-normalised
    labs: np.ndarray  # (n, 3)
    model: str

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {k: getattr(self, k) for k in ("codes", "variants", "names", "categories", "model")}
        np.savez_compressed(path, embeddings=self.embeddings, labs=self.labs, meta=json.dumps(meta))

    @classmethod
    def load(cls, path: Path) -> CatalogIndex:
        data = np.load(path)
        return cls(embeddings=data["embeddings"], labs=data["labs"], **json.loads(str(data["meta"])))


@dataclass(frozen=True)
class Match:
    code: str
    variant: str
    name: str
    category: str
    score: float


SOFT_CATEGORY_BONUS = 0.1  # "soft" mode: decors in the detected family's category get this head start


def rank(
    index: CatalogIndex,
    embedding: np.ndarray,
    lab: np.ndarray,
    family: str | None,
    k: int = 5,
    category_mode: str = "soft",
) -> list[Match]:
    """Top-k distinct decor codes for one surface (a flip sheet never shows twice).

    category_mode:
      "hard"  only decors in the category mapped from the material family
      "soft"  every decor, with a bonus for that category, so a mislabelled family (glossy
              marble called "plaster") can still surface the right decor
      "off"   ignore the family
    Each decor's colour penalty uses its own category's weight (solid colours lean on colour).
    """
    if category_mode not in {"hard", "soft", "off"}:
        raise ValueError(f"unknown category_mode {category_mode!r}")
    category = FAMILY_TO_CATEGORY.get(family or "") if category_mode != "off" else None
    cats = np.array(index.categories)
    in_cat = cats == category if category else np.zeros(len(cats), bool)
    weights = np.array([COLOUR_WEIGHT.get(c, DEFAULT_COLOUR_WEIGHT) for c in index.categories])
    score = index.embeddings @ embedding - weights * delta_e(index.labs, lab) / DELTA_E_SCALE
    if category_mode == "soft":
        score = score + SOFT_CATEGORY_BONUS * in_cat
    elif category_mode == "hard" and in_cat.any():
        score = np.where(in_cat, score, -np.inf)

    out: list[Match] = []
    for i in np.argsort(-score):
        if not np.isfinite(score[i]) or len(out) == k:
            break
        if any(m.code == index.codes[i] for m in out):
            continue
        out.append(
            Match(index.codes[i], index.variants[i], index.names[i], index.categories[i], float(score[i]))
        )
    return out
