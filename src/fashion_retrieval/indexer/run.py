"""Part A - The Indexer.

    python -m fashion_retrieval.indexer.run                 # extract + load
    python -m fashion_retrieval.indexer.run --stage extract --limit 32   # quick trial
    python -m fashion_retrieval.indexer.run --stage load    # rebuild DB from cached features

Extraction is resumable: finished batches are cached in artifacts/features/part_*.pkl.
"""

from __future__ import annotations

import argparse
import pickle
import time
from collections import Counter

from ..common.config import load_config, resolve
from ..common.io import read_jsonl
from .load import iter_records, load


def extract(cfg: dict, limit: int | None) -> None:
    from .extract import FeatureExtractor

    p = cfg["paths"]
    rows = list(read_jsonl(resolve(p["manifest"])))[: limit or None]
    feat_dir = resolve(p["features_dir"])
    feat_dir.mkdir(parents=True, exist_ok=True)
    bs = cfg["indexer"]["batch_size"]
    todo = [i for i in range(0, len(rows), bs) if not (feat_dir / f"part_{i // bs:05d}.pkl").exists()]
    print(f"[extract] {len(rows)} images, {len(todo)} of {-(-len(rows) // bs)} batches to do", flush=True)
    if not todo:
        return

    fx = FeatureExtractor(cfg)
    t0 = time.time()
    for n, i in enumerate(todo, 1):
        recs = fx(rows[i:i + bs], resolve(p["images_dir"]))
        tmp = feat_dir / f"part_{i // bs:05d}.tmp"
        with open(tmp, "wb") as f:
            pickle.dump(recs, f)
        tmp.rename(feat_dir / f"part_{i // bs:05d}.pkl")  # atomic: no half-written parts
        rate = (time.time() - t0) / (n * bs)
        print(f"[extract] batch {n}/{len(todo)}  {rate:.2f}s/img  eta {rate * bs * (len(todo) - n) / 60:.1f} min", flush=True)


def summary(cfg: dict) -> None:
    recs = list(iter_records(resolve(cfg["paths"]["features_dir"])))
    regs = [g for r in recs for g in r["regions"]]
    print(f"\n[summary] {len(recs)} images, {len(regs)} garment regions "
          f"({len(regs) / max(len(recs), 1):.1f}/img), {sum(not r['regions'] for r in recs)} images without regions")
    print("  scenes  :", dict(Counter(r["scene_top"] for r in recs)))
    print("  garments:", Counter(g["garment"] for g in regs).most_common(12))
    print("  colors  :", Counter(max(g["colors"], key=g["colors"].get) for g in regs).most_common(12))
    print("  example caption:", recs[0]["caption"][:200])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["extract", "load", "all"], default="all")
    ap.add_argument("--limit", type=int, default=None, help="only first N images (trial runs)")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.stage in ("extract", "all"):
        extract(cfg, args.limit)
        summary(cfg)
    if args.stage in ("load", "all"):
        load(cfg, resolve(cfg["paths"]["features_dir"]))


if __name__ == "__main__":
    main()
