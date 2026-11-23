"""Colour features: a surface's main colour in CIELAB, with optional lighting correction."""

from __future__ import annotations

import numpy as np
from PIL import Image


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB values in 0-255, shape (..., 3), to CIELAB (D65)."""
    c = np.asarray(rgb, dtype=np.float64) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    m = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    )
    xyz = lin @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack(
        [116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], axis=-1
    )


def gray_world(pixels: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """Partly remove a colour cast (e.g. warm lighting) by pulling channel means toward grey.

    strength=0 leaves pixels unchanged, 1 is full grey-world. A partial pull is used because a
    genuinely warm material (oak, travertine) should keep some of its warmth.
    """
    px = pixels.astype(np.float64)
    means = px.reshape(-1, 3).mean(axis=0)
    gain = means.mean() / np.maximum(means, 1e-6)
    return np.clip(px * (1 + strength * (gain - 1)), 0, 255)


def main_lab(img: Image.Image, correct_lighting: bool = False) -> np.ndarray:
    """Median CIELAB colour of an image, ignoring the darkest and brightest 5% (shadows, glare)."""
    px = np.asarray(img.convert("RGB").resize((64, 64)), dtype=np.float64).reshape(-1, 3)
    if correct_lighting:
        px = gray_world(px)
    lum = px.mean(axis=1)
    lo, hi = np.percentile(lum, [5, 95])
    keep = px[(lum >= lo) & (lum <= hi)]
    return srgb_to_lab(np.median(keep if len(keep) else px, axis=0))


def delta_e(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """CIE76 colour difference; broadcasting over the leading axes."""
    return np.linalg.norm(np.asarray(a) - np.asarray(b), axis=-1)
