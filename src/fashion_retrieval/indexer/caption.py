"""Dense captions with Florence-2 (native transformers implementation).

Captions add what embeddings compress away: explicit color-garment binding ("white shirt,
black tie"), actions/poses ("sitting on a bench") and scene objects. They feed:
  * a dense text vector (bge-small) for semantic caption<->query matching
  * a sparse BM25 vector for exact-term matching
  * extracted (color, garment) pairs (common.attributes) as a second binding signal
"""

from __future__ import annotations

import torch
from PIL import Image


class Captioner:
    def __init__(self, model_id: str, device: str, task: str, max_new_tokens: int, num_beams: int):
        from transformers import AutoProcessor, Florence2ForConditionalGeneration

        self.dtype = torch.float16 if device in ("mps", "cuda") else torch.float32
        self.model = Florence2ForConditionalGeneration.from_pretrained(model_id, torch_dtype=self.dtype).to(device).eval()
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.device, self.task = device, task
        self.gen = dict(max_new_tokens=max_new_tokens, num_beams=num_beams)

    @torch.no_grad()
    def __call__(self, images: list[Image.Image]) -> list[str]:
        inputs = self.proc(text=[self.task] * len(images), images=images, return_tensors="pt").to(self.device, self.dtype)
        ids = self.model.generate(**inputs, **self.gen)
        texts = self.proc.batch_decode(ids, skip_special_tokens=False)
        return [
            self.proc.post_process_generation(t, task=self.task, image_size=im.size)[self.task].strip()
            for t, im in zip(texts, images)
        ]
