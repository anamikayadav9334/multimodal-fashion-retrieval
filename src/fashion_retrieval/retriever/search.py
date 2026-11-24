"""Part B - The Retriever.

    hybrid   : parse -> 3-channel recall (FashionSigLIP dense, caption dense, BM25 sparse)
               -> RRF fusion -> feature re-rank (global, caption, bm25, scene, binding)
    fashion  : FashionSigLIP text->image only (ablation: fashion encoder without our logic)
    baseline : vanilla CLIP ViT-B/32 text->image (the "vanilla CLIP" the assignment refers to)

Every result carries per-feature scores and the garment evidence (boxes) that explain it.
"""

from __future__ import annotations

import time
from collections import defaultdict

import os

import numpy as np
from qdrant_client import models

from ..common import sparse
from ..common.config import resolve
from ..common.store import get_client
from .query import QueryParser, QuerySpec
from .scoring import binding_score, garment_text, minmax

MODES = ("hybrid", "fashion", "baseline")


class Retriever:
    def __init__(self, cfg: dict, client=None, runtime: str | None = None):
        m = cfg["models"]
        self.cfg, self.rc = cfg, cfg["retriever"]
        self.images, self.regions = cfg["qdrant"]["images_collection"], cfg["qdrant"]["regions_collection"]
        # "onnx": torch-free int8 query encoders (deployment); "torch": original models (dev/indexing)
        self.runtime = runtime or os.environ.get("ENCODER_RUNTIME") or m.get("query_runtime", "torch")
        if self.runtime == "onnx":
            from ..common import onnx_encoders

            self.fashion, self.baseline, self.text = onnx_encoders.load(str(resolve(m["onnx_dir"])))
        else:
            from ..common import encoders

            dev = encoders.pick_device(m["device"])
            self.fashion = encoders.fashion_encoder(m["fashion"], dev)
            self.baseline = encoders.baseline_encoder(m["baseline"]["arch"], m["baseline"]["pretrained"], dev)
            self.text = encoders.text_encoder(m["text"], dev)
        self.client = client or get_client(cfg)
        self.parser = QueryParser(cfg)

    # ------------------------------------------------------------------ public
    def search(self, query: str, k: int = 12, mode: str = "hybrid", use_llm: bool = True,
               spec: QuerySpec | None = None, weights: dict | None = None) -> dict:
        t0 = time.time()
        if mode == "baseline":
            q = self.baseline.encode_texts([query])[0]
            return self._simple(query, q, "baseline", k, t0)
        if mode == "fashion":
            q = self.fashion.encode_texts([query])[0]
            return self._simple(query, q, "global", k, t0)
        spec = spec or self.parser.parse(query, use_llm=use_llm)
        t_parse = time.time() - t0
        out = self._hybrid(spec, k, weights or self.rc["weights"])
        out["timing"] = {"parse_s": round(t_parse, 2), "total_s": round(time.time() - t0, 2)}
        return out

    # ------------------------------------------------------------------ simple dense modes
    def _simple(self, query: str, qvec: np.ndarray, using: str, k: int, t0: float) -> dict:
        res = self.client.query_points(self.images, query=qvec.tolist(), using=using, limit=k, with_payload=True).points
        return {
            "query": query, "mode": using if using == "baseline" else "fashion",
            "results": [self._result(p.payload, float(p.score)) for p in res],
            "timing": {"total_s": round(time.time() - t0, 2)},
        }

    # ------------------------------------------------------------------ hybrid
    def _hybrid(self, spec: QuerySpec, k: int, w: dict) -> dict:
        rk = self.rc["recall_k"]
        texts = [spec.text] + ([spec.expanded] if spec.expanded and spec.expanded != spec.text else [])

        q_img = self.fashion.encode_texts(texts).mean(0)
        q_img /= np.linalg.norm(q_img)
        q_cap = self.text.encode_queries(texts).mean(0)
        q_cap /= np.linalg.norm(q_cap)
        bm_text = " ".join(texts + [" ".join(g.colors + [g.surface]) for g in spec.garments])
        bi, bv = sparse.query_vector(bm_text)

        # ---- 1. recall: three channels
        channels = {
            "global": self.client.query_points(self.images, query=q_img.tolist(), using="global", limit=rk).points,
            "caption": self.client.query_points(self.images, query=q_cap.tolist(), using="caption", limit=rk).points,
            "bm25": self.client.query_points(self.images, query=models.SparseVector(indices=bi, values=bv),
                                             using="bm25", limit=rk).points if bi else [],
        }
        bm25_raw = {p.id: float(p.score) for p in channels["bm25"]}

        # ---- 2. reciprocal rank fusion
        rrf: dict[str, float] = defaultdict(float)
        for pts in channels.values():
            for rank, p in enumerate(pts):
                rrf[p.id] += 1.0 / (self.rc["rrf_k"] + rank + 1)
        pool_ids = sorted(rrf, key=rrf.get, reverse=True)[: self.rc["pool_k"]]
        pts = self.client.retrieve(self.images, ids=pool_ids, with_payload=True, with_vectors=["global", "caption"])
        if not pts:
            return {"query": spec.text, "mode": "hybrid", "spec": spec.to_dict(), "results": []}

        # ---- 3. features
        G = np.array([p.vector["global"] for p in pts], np.float32)
        C = np.array([p.vector["caption"] for p in pts], np.float32)
        f_global, f_caption = G @ q_img, C @ q_cap
        f_bm25 = np.array([bm25_raw.get(p.id, 0.0) for p in pts])
        f_scene = np.array([p.payload["scene"].get(spec.scene, 0.0) if spec.scene else 0.0 for p in pts])

        regions_by_img, crop_pcts = self._regions(spec, [p.payload["image_id"] for p in pts])
        f_bind, evidence = [], []
        for p in pts:
            s, ev = binding_score(spec, p.payload, regions_by_img.get(p.payload["image_id"], []), crop_pcts,
                                  self.rc["inferred_garment_weight"], self.rc["related_garment_credit"])
            f_bind.append(s)
            evidence.append(ev)
        f_bind = np.array(f_bind)

        feats = {"global": minmax(f_global), "caption": minmax(f_caption), "bm25": minmax(f_bm25),
                 "scene": f_scene, "binding": f_bind}
        active = {n: w[n] for n in feats if w.get(n, 0) > 0
                  and not (n == "scene" and not spec.scene) and not (n == "binding" and not spec.garments)}
        total = sum(active[n] * feats[n] for n in active) / max(sum(active.values()), 1e-9)

        order = np.argsort(-total)[:k]
        results = []
        for i in order:
            r = self._result(pts[i].payload, float(total[i]))
            r["features"] = {n: round(float(feats[n][i]), 3) for n in feats}
            r["raw"] = {"global": round(float(f_global[i]), 4), "caption": round(float(f_caption[i]), 4),
                        "bm25": round(float(f_bm25[i]), 3)}
            r["evidence"] = [vars(e) for e in evidence[i]]
            results.append(r)
        return {"query": spec.text, "mode": "hybrid", "spec": spec.to_dict(), "weights": active,
                "pool": len(pts), "recall": {c: len(v) for c, v in channels.items()}, "results": results}

    def _regions(self, spec: QuerySpec, image_ids: list[str]):
        """Fetch pool regions; per query garment, rank crops by similarity to 'a <colors> <garment>'."""
        if not spec.garments:
            return {}, []
        regs, offset = [], None
        while True:
            batch, offset = self.client.scroll(
                self.regions, limit=2048, offset=offset, with_vectors=True, with_payload=True,
                scroll_filter=models.Filter(must=[models.FieldCondition(key="image_id", match=models.MatchAny(any=image_ids))]))
            regs += batch
            if offset is None:
                break
        by_img: dict[str, list[dict]] = defaultdict(list)
        for j, r in enumerate(regs):
            by_img[r.payload["image_id"]].append({**r.payload, "_rid": j})
        if not regs:
            return by_img, [{} for _ in spec.garments]
        E = np.array([r.vector["crop"] for r in regs], np.float32)
        T = self.fashion.encode_texts([garment_text(g) for g in spec.garments])
        sims = T @ E.T                                   # (garments, regions)
        pct = sims.argsort(1).argsort(1) / max(len(regs) - 1, 1)  # percentile rank within the pool
        return by_img, [{j: float(pct[gi, j]) for j in range(len(regs))} for gi in range(len(spec.garments))]

    @staticmethod
    def _result(payload: dict, score: float) -> dict:
        return {"id": payload["image_id"], "path": payload["path"], "score": round(score, 4),
                "caption": payload.get("caption", ""), "scene_top": payload.get("scene_top"),
                "source": payload.get("source"), "width": payload.get("width"), "height": payload.get("height")}
