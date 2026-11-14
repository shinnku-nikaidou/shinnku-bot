"""A local NSFW check, so such pictures never leave this machine.

The model that looks at pictures runs on a personal ChatGPT account, and NSFW
pictures sent there can get that account banned. This small classifier runs on
CPU first, and a picture it flags is not sent.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np

# ViT-base at 384px, fine-tuned on photos and drawings; Apache-2.0.
REPO = "AdamCodd/vit-base-nsfw-detector"
REVISION = "8587de998f441aac03fdd57a85d2e4cb808c7d64"
FILE = "onnx/model.onnx"
SIZE = 384
# Low on purpose: a false alarm only costs a description, a miss can cost the account.
THRESHOLD = 0.25


def pixels(image: bytes) -> np.ndarray:
    """The picture as the model's input: 1x3x384x384, scaled to [-1, 1], on white."""
    from PIL import Image

    with Image.open(io.BytesIO(image)) as img:
        rgba = img.convert("RGBA")
    flat = Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba).convert("RGB")
    flat = flat.resize((SIZE, SIZE), Image.Resampling.BILINEAR)
    x = np.asarray(flat, dtype=np.float32) / 127.5 - 1.0
    return x.transpose(2, 0, 1)[None]


class NsfwCheck:
    def __init__(self, cache_dir: Path) -> None:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(REPO, FILE, revision=REVISION, cache_dir=str(cache_dir))
        self._session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self._input = self._session.get_inputs()[0].name

    def score(self, image: bytes) -> float:
        """How likely the picture is NSFW, 0 to 1."""
        logits = self._session.run(None, {self._input: pixels(image)})[0][0]
        odds = np.exp(logits - logits.max())
        return float(odds[1] / odds.sum())

    def __call__(self, image: bytes) -> bool:
        """Whether the picture must stay here."""
        return self.score(image) >= THRESHOLD
