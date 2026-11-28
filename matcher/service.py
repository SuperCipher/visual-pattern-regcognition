"""Crop + embed + colour for the surfaces of one photo: the work the Modal service does.

Pure apart from the embedder, so it is tested locally with the colour-histogram embedder.
"""

from __future__ import annotations

import base64
import io
from collections.abc import Iterable, Sequence

from PIL import Image

from matcher.colour import main_lab
from matcher.embed import Embedder

MIN_BOX_SIDE = 40  # ignore slivers narrower than this many pixels
MAX_SIDE = 1280


def crop(img: Image.Image, box: Iterable[float]) -> Image.Image | None:
    """Crop a 0-1000 normalised box; None if it is too small to say anything about the material."""
    w, h = img.size
    x1, y1, x2, y2 = box
    px = (int(x1 * w / 1000), int(y1 * h / 1000), int(x2 * w / 1000), int(y2 * h / 1000))
    if px[2] - px[0] < MIN_BOX_SIDE or px[3] - px[1] < MIN_BOX_SIDE:
        return None
    return img.crop(px)


def decode_image(data_url: str) -> Image.Image:
    """A data:image/...;base64 URL (or bare base64) -> RGB image, long side <= MAX_SIDE."""
    raw = base64.b64decode(data_url.split(",", 1)[-1], validate=False)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    return img


def embed_surfaces(
    img: Image.Image, boxes: Sequence[Sequence[float]], embedder: Embedder
) -> list[dict | None]:
    """Per box: {"embedding": [...], "lab": [L, a, b]}, or None if the box is too small."""
    crops = [crop(img, b) for b in boxes]
    usable = [c for c in crops if c is not None]
    embs = iter(embedder.embed(usable)) if usable else iter(())
    out: list[dict | None] = []
    for c in crops:
        if c is None:
            out.append(None)
            continue
        out.append(
            {
                "embedding": [round(float(v), 6) for v in next(embs)],
                "lab": [round(float(v), 4) for v in main_lab(c, correct_lighting=True)],
            }
        )
    return out
