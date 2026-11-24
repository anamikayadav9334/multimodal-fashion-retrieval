"""Garment color naming: pixel evidence (LAB k-means + CIEDE2000) fused with zero-shot evidence.

Why two signals:
  * Pixels are precise for solid colors but fooled by shadows, skin and background in the box.
  * The crop-embedding ("a red tie" vs "a navy tie") is robust to lighting and patterns, but
    coarse between neighbouring shades.
Fusing both gives a calibrated distribution, e.g. {"navy": 0.55, "blue": 0.30}, which the
retriever uses as soft evidence rather than a hard label.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from skimage.color import deltaE_ciede2000, rgb2lab
from sklearn.cluster import KMeans

from ..common.vocab import COLOR_ANCHORS_RGB, COLORS

_ANCHOR_NAMES = [n for n, rgbs in COLOR_ANCHORS_RGB.items() for _ in rgbs]
_ANCHOR_LAB = rgb2lab(np.array([c for rgbs in COLOR_ANCHORS_RGB.values() for c in rgbs],
                               dtype=np.float64)[None] / 255.0)[0]


def _skin_mask(rgb: np.ndarray) -> np.ndarray:
    """Classic YCrCb skin-tone box. rgb: (N,3) uint8."""
    r, g, b = [rgb[:, i].astype(np.float32) for i in range(3)]
    cr = 0.5 * r - 0.4187 * g - 0.0813 * b + 128
    cb = -0.1687 * r - 0.3313 * g + 0.5 * b + 128
    return (cr > 135) & (cr < 175) & (cb > 85) & (cb < 130)


def _name_distribution(lab: np.ndarray, temperature: float) -> np.ndarray:
    """Softmax over palette names from CIEDE2000 distance to the closest anchor of each name."""
    d = deltaE_ciede2000(np.repeat(lab[None], len(_ANCHOR_LAB), 0), _ANCHOR_LAB)
    best = {n: np.inf for n in COLORS}
    for n, dist in zip(_ANCHOR_NAMES, d):
        best[n] = min(best[n], dist)
    v = -np.array([best[n] for n in COLORS]) / temperature
    e = np.exp(v - v.max())
    return e / e.sum()


def pixel_color_probs(image: Image.Image, box, shrink: float, k: int, temperature: float) -> np.ndarray:
    x0, y0, x1, y1 = box
    dx, dy = (x1 - x0) * shrink, (y1 - y0) * shrink
    patch = image.crop((x0 + dx, y0 + dy, x1 - dx, y1 - dy)).resize((48, 48))
    rgb = np.asarray(patch, dtype=np.uint8).reshape(-1, 3)
    skin = _skin_mask(rgb)
    if skin.mean() < 0.6:  # if the garment itself is skin-toned (beige/tan), keep all pixels
        rgb = rgb[~skin]
    if len(rgb) < 20:
        return np.full(len(COLORS), 1 / len(COLORS))
    lab = rgb2lab(rgb[None].astype(np.float64) / 255.0)[0]
    km = KMeans(n_clusters=min(k, len(lab)), n_init=3, random_state=0).fit(lab)
    shares = np.bincount(km.labels_, minlength=km.n_clusters) / len(lab)
    probs = np.zeros(len(COLORS))
    for c, s in zip(km.cluster_centers_, shares):
        probs += s * _name_distribution(c, temperature)
    return probs / probs.sum()


class ZeroShotColor:
    """P(color | crop) from crop embedding vs. 'a {color} {garment}' text embeddings."""

    def __init__(self, encoder, garment_classes: list[str]):
        self.scale = encoder.logit_scale
        self.text = {g: encoder.encode_texts([f"a photo of a {c} {g}" for c in COLORS]) for g in garment_classes}

    def __call__(self, crop_emb: np.ndarray, garment_cls: str) -> np.ndarray:
        sims = self.text[garment_cls] @ crop_emb
        z = sims * min(self.scale, 100.0)
        e = np.exp(z - z.max())
        return e / e.sum()


def fuse(pixel: np.ndarray, zeroshot: np.ndarray, pixel_weight: float, min_prob: float) -> dict[str, float]:
    p = pixel_weight * pixel + (1 - pixel_weight) * zeroshot
    order = np.argsort(-p)
    return {COLORS[i]: round(float(p[i]), 3) for i in order[:3] if p[i] >= min_prob} or {COLORS[order[0]]: round(float(p[order[0]]), 3)}
