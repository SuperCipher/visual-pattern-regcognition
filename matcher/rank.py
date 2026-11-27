"""Rank catalog decors for one detected surface."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
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

# "auto" mode: how strictly to trust each material family. Qwen's wood, metal and paint labels
# are reliable, so those stay inside their category. It calls glossy marble "plaster", so plaster
# keeps the whole catalog open with only a head start for solid colours.
FAMILY_MODE = {"plaster": "soft"}
DEFAULT_FAMILY_MODE = "hard"


@dataclass(frozen=True)
class CatalogIndex:
    codes: list[str]
    variants: list[str]
    names: list[str]
    categories: list[str]
    embeddings: np.ndarray  # (n, d), L2-normalised
    labs: np.ndarray  # (n, 3)
    model: str
    # How strongly each decor resembles wall crops in general (see hub_scores); None = not computed.
    hub: np.ndarray | None = field(default=None, compare=False)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {k: getattr(self, k) for k in ("codes", "variants", "names", "categories", "model")}
        arrays = {"embeddings": self.embeddings, "labs": self.labs}
        if self.hub is not None:
            arrays["hub"] = self.hub
        np.savez_compressed(path, meta=json.dumps(meta), **arrays)

    @classmethod
    def load(cls, path: Path) -> CatalogIndex:
        data = np.load(path)
        hub = data["hub"] if "hub" in data.files else None
        return cls(embeddings=data["embeddings"], labs=data["labs"], hub=hub, **json.loads(str(data["meta"])))


@dataclass(frozen=True)
class Match:
    code: str
    variant: str
    name: str
    category: str
    score: float


SOFT_CATEGORY_BONUS = 0.1  # "soft" mode: decors in the detected family's category get this head start
HUB_WEIGHT = 0.5  # how much of a decor's hub score is subtracted when hub correction is on
HUB_K = 10


def hub_scores(embeddings: np.ndarray, queries: np.ndarray, k: int = HUB_K) -> np.ndarray:
    """Mean cosine similarity of each decor to its k most similar wall crops.

    Some decors (a dark marble, a mid-brown oak) sit close to almost any crop and crowd every
    top 5. This is the decor-side term of CSLS (Conneau et al. 2018): subtracting it lowers
    those "hubs" without needing any labels, only a set of typical wall crops.
    """
    sims = embeddings @ queries.T  # (n_decors, n_queries)
    k = min(k, sims.shape[1])
    return np.sort(sims, axis=1)[:, -k:].mean(axis=1).astype(np.float32)


def rank(
    index: CatalogIndex,
    embedding: np.ndarray,
    lab: np.ndarray,
    family: str | None,
    k: int = 5,
    category_mode: str = "soft",
    hub_weight: float = 0.0,
) -> list[Match]:
    """Top-k distinct decor codes for one surface (a flip sheet never shows twice).

    category_mode:
      "hard"  only decors in the category mapped from the material family
      "soft"  every decor, with a bonus for that category, so a mislabelled family (glossy
              marble called "plaster") can still surface the right decor
      "off"   ignore the family
      "auto"  hard or soft per family (FAMILY_MODE): soft only where the vision model is unreliable
    Each decor's colour penalty uses its own category's weight (solid colours lean on colour).
    hub_weight > 0 subtracts that share of index.hub (decors that match everything).
    """
    if category_mode not in {"hard", "soft", "off", "auto"}:
        raise ValueError(f"unknown category_mode {category_mode!r}")
    if category_mode == "auto":
        category_mode = FAMILY_MODE.get(family or "", DEFAULT_FAMILY_MODE)
    category = FAMILY_TO_CATEGORY.get(family or "") if category_mode != "off" else None
    cats = np.array(index.categories)
    in_cat = cats == category if category else np.zeros(len(cats), bool)
    weights = np.array([COLOUR_WEIGHT.get(c, DEFAULT_COLOUR_WEIGHT) for c in index.categories])
    score = index.embeddings @ embedding - weights * delta_e(index.labs, lab) / DELTA_E_SCALE
    if hub_weight and index.hub is not None:
        score = score - hub_weight * index.hub
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
