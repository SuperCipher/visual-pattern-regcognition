"""SigLIP embedding service on Modal (serverless, billed per second, nothing while idle).

POST {"image": "data:image/jpeg;base64,...", "boxes": [[x1, y1, x2, y2], ...]}  (0-1000 boxes)
-> {"surfaces": [{"embedding": [768 floats], "lab": [L, a, b]} | null, ...], "model": "..."}

Same model and maths as step 2 (matcher/embed.py HFEmbedder, matcher/colour.py), so the web
catalog built with `catalog/build_index.py --models google/siglip-base-patch16-224` matches it.
Callers need a Modal proxy auth token (Modal-Key / Modal-Secret headers): only the Worker has one.

Deploy (from the repo root, after `uv run --extra modal modal token new` once):
    uv run --extra modal modal deploy service/modal_app.py
"""

from __future__ import annotations

import modal

MODEL = "google/siglip-base-patch16-224"


def download_model() -> None:
    from transformers import AutoImageProcessor, AutoModel

    AutoImageProcessor.from_pretrained(MODEL)
    AutoModel.from_pretrained(MODEL)


image = (
    modal.Image.debian_slim(python_version="3.12")
    # Same versions as uv.lock, so embeddings match the step 2 runs.
    .uv_pip_install("torch==2.15.0", index_url="https://download.pytorch.org/whl/cpu")
    .uv_pip_install("transformers==5.24.0", "pillow==13.0.0", "numpy==2.5.5", "fastapi[standard]>=0.115")
    .run_function(download_model)  # weights baked into the image: no download on cold start
    .add_local_python_source("matcher")
)
app = modal.App("blackbone-siglip", image=image)


@app.cls(cpu=2, memory=3072, scaledown_window=120, max_containers=4, enable_memory_snapshot=True)
class Siglip:
    @modal.enter(snap=True)
    def load(self) -> None:
        from matcher.embed import HFEmbedder

        self.embedder = HFEmbedder(MODEL)

    @modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)
    def embed(self, body: dict) -> dict:
        from fastapi import HTTPException

        from matcher.service import decode_image, embed_surfaces

        boxes = body.get("boxes")
        if not isinstance(body.get("image"), str) or not isinstance(boxes, list) or len(boxes) > 40:
            raise HTTPException(status_code=400, detail="need image and up to 40 boxes")
        try:
            img = decode_image(body["image"])
        except Exception as e:
            raise HTTPException(status_code=400, detail="unreadable image") from e
        return {"surfaces": embed_surfaces(img, boxes, self.embedder), "model": MODEL}

    @modal.fastapi_endpoint(method="GET", requires_proxy_auth=True)
    def warm(self) -> dict:
        """Cheap call the Worker makes first, so the cold start overlaps the vision-model call."""
        return {"ok": True}
