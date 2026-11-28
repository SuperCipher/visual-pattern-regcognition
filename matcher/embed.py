"""Image embedders. Each returns L2-normalised vectors so a dot product is cosine similarity.

- "colorhist": a tiny colour-histogram baseline with no model download (also used in tests).
- Any Hugging Face vision model id, e.g. "facebook/dinov2-small" or "openai/clip-vit-base-patch32".
  Both have ONNX ports for transformers.js, so the browser app can use the same model later.
- "onnx:<hf repo>[:<variant>]", e.g. "onnx:Xenova/siglip-base-patch16-224:quantized": the exact ONNX
  file the web app downloads, run with onnxruntime. The web catalog is built with this, so browser
  and catalog embeddings come from the same weights.
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


def preprocess(img: Image.Image, size: tuple[int, int], mean, std, rescale: float) -> np.ndarray:
    """Resize (bicubic, no crop), rescale and normalise to a (3, H, W) float32 array.

    Mirrors what web/src/embed.ts does before calling the same ONNX model.
    """
    px = np.asarray(img.convert("RGB").resize(size, Image.Resampling.BICUBIC), dtype=np.float32) * rescale
    px = (px - np.asarray(mean, np.float32)) / np.asarray(std, np.float32)
    return px.transpose(2, 0, 1)


class OnnxEmbedder:
    """A transformers.js ONNX vision model (e.g. Xenova/siglip-base-patch16-224) via onnxruntime."""

    def __init__(self, name: str, batch_size: int = 16) -> None:
        import json
        from pathlib import Path

        import onnxruntime as ort
        from huggingface_hub import hf_hub_download

        _, repo, *rest = name.split(":")
        variant = rest[0] if rest else ""
        onnx_file = f"onnx/vision_model{'_' + variant if variant else ''}.onnx"
        self.name = name
        self.batch_size = batch_size
        cfg = json.loads(Path(hf_hub_download(repo, "preprocessor_config.json")).read_text())
        size = cfg.get("size", {})
        self.size = (size.get("width", 224), size.get("height", 224))
        self.mean, self.std = cfg.get("image_mean", [0.5] * 3), cfg.get("image_std", [0.5] * 3)
        self.rescale = cfg.get("rescale_factor", 1 / 255)
        self.session = ort.InferenceSession(
            hf_hub_download(repo, onnx_file), providers=["CPUExecutionProvider"]
        )
        # fp16 exports may take float16 input; feed whatever the model declares.
        self.input_dtype = np.float16 if "float16" in self.session.get_inputs()[0].type else np.float32
        outputs = [o.name for o in self.session.get_outputs()]
        self.output = "image_embeds" if "image_embeds" in outputs else "pooler_output"

    def embed(self, images: Sequence[Image.Image]) -> np.ndarray:
        out = []
        for i in range(0, len(images), self.batch_size):
            batch = np.stack(
                [
                    preprocess(im, self.size, self.mean, self.std, self.rescale)
                    for im in images[i : i + self.batch_size]
                ]
            )
            feed = {"pixel_values": batch.astype(self.input_dtype)}
            out.append(self.session.run([self.output], feed)[0].astype(np.float32))
        return l2_normalise(np.concatenate(out).astype(np.float32))


def load_embedder(name: str) -> Embedder:
    if name == "colorhist":
        return ColourHistEmbedder()
    if name.startswith("onnx:"):
        return OnnxEmbedder(name)
    return HFEmbedder(name)
