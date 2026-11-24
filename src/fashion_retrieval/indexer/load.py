"""Stage 2 of the indexer: feature records -> Qdrant (fast, no models needed)."""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
from qdrant_client import models

from ..common import sparse
from ..common.store import create_collections, get_client, point_id


def iter_records(features_dir: Path):
    for part in sorted(features_dir.glob("part_*.pkl")):
        with open(part, "rb") as f:
            yield from pickle.load(f)


def image_payload(r: dict) -> dict:
    garments = sorted({g["garment"] for g in r["regions"]})
    region_pairs = {f"{c}:{g['garment']}" for g in r["regions"] for c in g["colors"]}
    caption_pairs = {f"{c}:{m['garment']}" for m in r["caption_mentions"] for c in m["colors"]}
    return {
        "image_id": r["id"], "path": r["path"], "source": r["source"],
        "width": r["width"], "height": r["height"],
        "caption": r["caption"], "caption_mentions": r["caption_mentions"],
        "scene": r["scene"], "scene_top": r["scene_top"],
        "garments": garments,
        "pairs": sorted(region_pairs | caption_pairs),
        "n_regions": len(r["regions"]),
    }


def load(cfg: dict, features_dir: Path, dims: dict[str, int] | None = None, batch: int = 128) -> dict:
    t = time.time()
    records = list(iter_records(features_dir))
    if not records:
        raise SystemExit(f"No feature parts in {features_dir}; run --stage extract first.")
    dims = dims or {
        "global": len(records[0]["global"]), "crop": len(records[0]["global"]),
        "baseline": len(records[0]["baseline"]), "caption": len(records[0]["caption_emb"]),
    }
    bm = cfg["indexer"]["bm25"]
    toks = [sparse.tokenize(r["caption"]) for r in records]
    avgdl = float(np.mean([len(t) for t in toks]))

    client = get_client(cfg)
    q = cfg["qdrant"]
    create_collections(client, cfg, dims)

    img_points, reg_points = [], []
    for r, tk in zip(records, toks):
        idx, val = sparse.doc_vector(tk, avgdl, bm["k1"], bm["b"])
        img_points.append(models.PointStruct(
            id=point_id(r["id"]),
            vector={"global": r["global"].tolist(), "baseline": r["baseline"].tolist(),
                    "caption": r["caption_emb"].tolist(),
                    "bm25": models.SparseVector(indices=idx, values=val)},
            payload=image_payload(r),
        ))
        for j, g in enumerate(r["regions"]):
            reg_points.append(models.PointStruct(
                id=point_id(f"{r['id']}#{j}"),
                vector={"crop": g["emb"].tolist()},
                payload={"image_id": r["id"], **{k: v for k, v in g.items() if k != "emb"}},
            ))

    for coll, pts in [(q["images_collection"], img_points), (q["regions_collection"], reg_points)]:
        for i in range(0, len(pts), batch):
            client.upsert(coll, pts[i:i + batch], wait=True)

    meta = {"images": len(img_points), "regions": len(reg_points), "dims": dims,
            "bm25_avgdl": round(avgdl, 2), "models": cfg["models"], "built_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    (features_dir.parent / "index_meta.json").write_text(json.dumps(meta, indent=2))
    client.close()
    print(f"[load] {meta['images']} images, {meta['regions']} regions -> Qdrant in {time.time() - t:.1f}s")
    return meta
