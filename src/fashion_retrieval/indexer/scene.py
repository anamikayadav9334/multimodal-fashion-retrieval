"""Zero-shot scene recognition with prompt ensembles.

Uses the general-domain CLIP (not the fashion-tuned model): scene understanding ("office",
"park bench") is general visual knowledge, and fashion fine-tuning narrows it.
Each scene's text embedding is the normalized mean over several prompts.
"""

from __future__ import annotations

import numpy as np

from ..common.vocab import SCENE_PROMPTS


class SceneClassifier:
    def __init__(self, encoder):
        self.names = list(SCENE_PROMPTS)
        protos = []
        for n in self.names:
            e = encoder.encode_texts(SCENE_PROMPTS[n]).mean(0)
            protos.append(e / np.linalg.norm(e))
        self.protos = np.stack(protos)
        self.scale = min(encoder.logit_scale, 100.0)

    def __call__(self, image_embs: np.ndarray) -> list[dict[str, float]]:
        z = image_embs @ self.protos.T * self.scale
        z = np.exp(z - z.max(1, keepdims=True))
        p = z / z.sum(1, keepdims=True)
        return [{n: round(float(v), 3) for n, v in zip(self.names, row)} for row in p]
