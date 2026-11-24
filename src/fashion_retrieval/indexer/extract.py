"""Stage 1 of the indexer: images -> feature records (model-heavy, resumable, DB-agnostic).

Separating extraction from loading means the vector DB can be rebuilt / re-weighted / swapped
without re-running any model.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PIL import Image

from ..common import encoders
from ..common.attributes import extract_garment_mentions
from ..common.vocab import DETECTOR_CLASSES
from .caption import Captioner
from .color import ZeroShotColor, fuse, pixel_color_probs
from .detect import GarmentDetector, crop
from .scene import SceneClassifier


class FeatureExtractor:
    def __init__(self, cfg: dict):
        m, ix = cfg["models"], cfg["indexer"]
        self.ix = ix
        dev = encoders.pick_device(m["device"])
        t = time.time()
        self.fashion = encoders.fashion_encoder(m["fashion"], dev)
        self.baseline = encoders.baseline_encoder(m["baseline"]["arch"], m["baseline"]["pretrained"], dev)
        self.text = encoders.text_encoder(m["text"], dev)
        self.detector = GarmentDetector(m["detector"], dev, ix["detector_threshold"], ix["min_region_area"])
        c = ix["caption"]
        self.captioner = Captioner(m["captioner"], dev, c["task"], c["max_new_tokens"], c["num_beams"])
        self.scene = SceneClassifier(self.baseline)
        self.zs_color = ZeroShotColor(self.fashion, sorted(set(DETECTOR_CLASSES.values())))
        print(f"[indexer] models ready on {dev} in {time.time() - t:.0f}s", flush=True)

    def dims(self) -> dict[str, int]:
        return {"global": self.fashion.dim, "crop": self.fashion.dim,
                "baseline": self.baseline.dim, "caption": int(self.text.encode_docs(["x"]).shape[1])}

    def __call__(self, rows: list[dict], images_dir: Path) -> list[dict]:
        images = [Image.open(images_dir / r["path"]).convert("RGB") for r in rows]
        g = self.fashion.encode_images(images)
        b = self.baseline.encode_images(images)
        scenes = self.scene(b)
        dets = self.detector(images)

        cb = self.ix["caption"]["batch_size"]
        captions = [c for i in range(0, len(images), cb) for c in self.captioner(images[i:i + cb])]
        cap_emb = self.text.encode_docs(captions)

        # all garment crops of the batch in one encoder pass
        crops, owners = [], []
        for i, (im, regs) in enumerate(zip(images, dets)):
            for reg in regs:
                crops.append(crop(im, reg.box, self.ix["crop_pad"]))
                owners.append((i, reg))
        crop_emb = self.fashion.encode_images(crops) if crops else np.zeros((0, g.shape[1]), np.float32)

        col = self.ix["color"]
        regions: list[list[dict]] = [[] for _ in images]
        for (i, reg), emb in zip(owners, crop_emb):
            pix = pixel_color_probs(images[i], reg.box, col["shrink"], col["kmeans_k"], col["temperature"])
            colors = fuse(pix, self.zs_color(emb, reg.cls), col["pixel_weight"], col["min_prob"])
            w, h = images[i].size
            regions[i].append({"garment": reg.cls, "score": reg.score, "box": reg.box_norm(w, h),
                               "area": reg.area_frac, "colors": colors, "emb": emb})

        out = []
        for i, r in enumerate(rows):
            sc = scenes[i]
            out.append({
                "id": r["id"], "path": r["path"], "source": r["source"],
                "width": images[i].size[0], "height": images[i].size[1],
                "global": g[i], "baseline": b[i], "caption_emb": cap_emb[i],
                "caption": captions[i],
                "caption_mentions": [m.as_dict() for m in extract_garment_mentions(captions[i])],
                "scene": sc, "scene_top": max(sc, key=sc.get),
                "regions": regions[i],
            })
        return out
