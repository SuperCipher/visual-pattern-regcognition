"""Image embedders. Each returns L2-normalised vectors so a dot product is cosine similarity.

- "colorhist": a tiny colour-histogram baseline with no model download (also used in tests).
- Any Hugging Face vision model id, e.g. "facebook/dinov2-small" or "openai/clip-vit-base-patch32".
  Both have ONNX ports for transformers.js, so the browser app can use the same model later.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np
from PIL import Image


class Embedder(Protocol):
    name: str

    def embed(self, images: Sequence[Image.Image]) -> np.ndarray: ...


def l2_normalise(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


class ColourHistEmbedder:
    """4x4x4 RGB histogram. Knows nothing about texture: a floor for the real models to beat."""

    name = "colorhist"

    def embed(self, images: Sequence[Image.Image]) -> np.ndarray:
        feats = []
        for img in images:
            px = np.asarray(img.convert("RGB").resize((64, 64))).reshape(-1, 3) // 64
            idx = px[:, 0] * 16 + px[:, 1] * 4 + px[:, 2]
            feats.append(np.sqrt(np.bincount(idx, minlength=64).astype(np.float64)))
        return l2_normalise(np.stack(feats))


class HFEmbedder:
    """Wraps a Hugging Face image model. Uses CLIP's image projection when present, else the CLS token."""

    def __init__(self, model_id: str, batch_size: int = 16) -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self.name = model_id
        self.batch_size = batch_size
        self.torch = torch
        self.device = (
            "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()

    def _forward(self, pixel_values):
        if hasattr(self.model, "get_image_features"):
            out = self.model.get_image_features(pixel_values=pixel_values)
            # transformers 4.x returns a tensor; 5.x wraps the projected features in pooler_output
            return out if isinstance(out, self.torch.Tensor) else out.pooler_output
        return self.model(pixel_values=pixel_values).last_hidden_state[:, 0]

    def embed(self, images: Sequence[Image.Image]) -> np.ndarray:
        out = []
        with self.torch.no_grad():
            for i in range(0, len(images), self.batch_size):
                batch = [im.convert("RGB") for im in images[i : i + self.batch_size]]
                inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
                out.append(self._forward(inputs["pixel_values"]).float().cpu().numpy())
        return l2_normalise(np.concatenate(out))


def load_embedder(name: str) -> Embedder:
    return ColourHistEmbedder() if name == "colorhist" else HFEmbedder(name)
