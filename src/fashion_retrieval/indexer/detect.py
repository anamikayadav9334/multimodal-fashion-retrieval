"""Garment detection with YOLOS-Fashionpedia.

Keeps whole garments/accessories only (garment *parts* like sleeves and collars are dropped),
removes tiny boxes, and de-duplicates overlapping boxes of the same class.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from PIL import Image
from torchvision.ops import nms

from ..common.vocab import DETECTOR_CLASSES


@dataclass
class Region:
    cls: str                                # short detector class, e.g. "tie", "coat"
    score: float
    box: tuple[float, float, float, float]  # x0, y0, x1, y1 in pixels
    area_frac: float

    def box_norm(self, w: int, h: int) -> list[float]:
        x0, y0, x1, y1 = self.box
        return [round(x0 / w, 4), round(y0 / h, 4), round(x1 / w, 4), round(y1 / h, 4)]


class GarmentDetector:
    def __init__(self, model_id: str, device: str, threshold: float, min_area: float):
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        self.proc = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForObjectDetection.from_pretrained(model_id).to(device).eval()
        self.device, self.threshold, self.min_area = device, threshold, min_area
        self.id2label = self.model.config.id2label

    @torch.no_grad()
    def __call__(self, images: list[Image.Image]) -> list[list[Region]]:
        inputs = self.proc(images=images, return_tensors="pt")
        out = self.model(**{k: v.to(self.device) for k, v in inputs.items()})
        sizes = torch.tensor([im.size[::-1] for im in images])
        results = self.proc.post_process_object_detection(out, threshold=self.threshold, target_sizes=sizes)
        return [self._clean(r, im.size) for r, im in zip(results, images)]

    def _clean(self, r: dict, size: tuple[int, int]) -> list[Region]:
        w, h = size
        keep: list[Region] = []
        by_cls: dict[str, list[int]] = {}
        labels = [self.id2label[int(l)] for l in r["labels"]]
        for i, lab in enumerate(labels):
            if lab in DETECTOR_CLASSES:
                by_cls.setdefault(DETECTOR_CLASSES[lab], []).append(i)
        for cls, idx in by_cls.items():
            boxes, scores = r["boxes"][idx].float().cpu(), r["scores"][idx].float().cpu()
            for j in nms(boxes, scores, iou_threshold=0.5).tolist():
                x0, y0, x1, y1 = [float(v) for v in boxes[j].clamp(min=0)]
                x1, y1 = min(x1, w), min(y1, h)
                area = max(0.0, x1 - x0) * max(0.0, y1 - y0) / (w * h)
                if area >= self.min_area:
                    keep.append(Region(cls, round(float(scores[j]), 3), (x0, y0, x1, y1), round(area, 4)))
        return sorted(keep, key=lambda g: -g.score)


def crop(image: Image.Image, box: tuple[float, float, float, float], pad: float) -> Image.Image:
    """Crop with a little context padding (helps the encoder recognize e.g. a tie under a collar)."""
    w, h = image.size
    x0, y0, x1, y1 = box
    px, py = (x1 - x0) * pad, (y1 - y0) * pad
    return image.crop((max(0, x0 - px), max(0, y0 - py), min(w, x1 + px), min(h, y1 + py)))
