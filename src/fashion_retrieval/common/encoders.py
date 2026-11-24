"""Model wrappers shared by the indexer and the retriever (same encoder on both sides).

All encoders return L2-normalized float32 numpy arrays, so cosine similarity == dot product.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
from PIL import Image


def pick_device(pref: str = "auto") -> str:
    if pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ClipEncoder:
    """open_clip model (Marqo-FashionSigLIP or vanilla CLIP) for images and short texts."""

    def __init__(self, name: str, pretrained: str | None = None, device: str = "cpu"):
        import open_clip

        self.device = device
        if name.startswith("hf-hub:"):
            model, _, preprocess = open_clip.create_model_and_transforms(name)
            self.tokenizer = open_clip.get_tokenizer(name)
        else:
            model, _, preprocess = open_clip.create_model_and_transforms(name, pretrained=pretrained)
            self.tokenizer = open_clip.get_tokenizer(name)
        self.model = model.to(device).eval()
        self.preprocess = preprocess
        self.logit_scale = float(self.model.logit_scale.exp().item())

    @torch.no_grad()
    def encode_images(self, images: list[Image.Image], batch_size: int = 32) -> np.ndarray:
        out = []
        for i in range(0, len(images), batch_size):
            x = torch.stack([self.preprocess(im) for im in images[i:i + batch_size]]).to(self.device)
            out.append(self.model.encode_image(x, normalize=True).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, self.dim), np.float32)

    @torch.no_grad()
    def encode_texts(self, texts: list[str], batch_size: int = 128) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch_size):
            t = self.tokenizer(texts[i:i + batch_size]).to(self.device)
            out.append(self.model.encode_text(t, normalize=True).float().cpu().numpy())
        return np.concatenate(out)

    @property
    def dim(self) -> int:
        return int(self.encode_texts(["x"]).shape[1])


class TextEncoder:
    """Sentence embedding model for caption <-> query similarity (long text, no 64-token cap)."""

    def __init__(self, name: str, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(name, device=device)
        self.is_bge = "bge" in name.lower()

    def encode_docs(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True, batch_size=64).astype(np.float32)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        # BGE models are trained with an instruction prefix on the query side.
        if self.is_bge:
            texts = [f"Represent this sentence for searching relevant passages: {t}" for t in texts]
        return self.model.encode(texts, normalize_embeddings=True).astype(np.float32)


@lru_cache(maxsize=None)
def fashion_encoder(name: str, device: str) -> ClipEncoder:
    return ClipEncoder(name, device=device)


@lru_cache(maxsize=None)
def baseline_encoder(arch: str, pretrained: str, device: str) -> ClipEncoder:
    return ClipEncoder(arch, pretrained=pretrained, device=device)


@lru_cache(maxsize=None)
def text_encoder(name: str, device: str) -> TextEncoder:
    return TextEncoder(name, device=device)
