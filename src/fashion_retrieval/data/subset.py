"""Shrink the manifest to a smaller balanced subset (e.g. 600 images for a laptop-friendly demo).

    python -m fashion_retrieval.data.subset --size 600 --keep-first 192

* The full manifest is preserved as data/manifest_full.jsonl.
* --keep-first N keeps the first N rows unchanged (already-indexed batches stay valid).
* Remaining Flickr30k slots are filled round-robin over scene x style cells (concept-quota
  images first); Fashionpedia keeps half of each per-class minimum, then random full outfits.
"""

from __future__ import annotations

import argparse
import random
import shutil
from collections import defaultdict

from ..common.config import load_config, resolve
from ..common.io import read_jsonl, write_jsonl
from .build import stats


def round_robin(rows: list[dict], n: int, key) -> list[dict]:
    cells: dict = defaultdict(list)
    for r in rows:
        cells[key(r)].append(r)
    out, queues = [], list(cells.values())
    while len(out) < n and any(queues):
        for q in queues:
            if q and len(out) < n:
                out.append(q.pop(0))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", type=int, default=600)
    ap.add_argument("--keep-first", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config()
    manifest = resolve(cfg["paths"]["manifest"])
    full = manifest.with_name("manifest_full.jsonl")
    if not full.exists():
        shutil.copy(manifest, full)
    rows = list(read_jsonl(full))
    rng = random.Random(cfg["dataset"]["seed"])

    kept = rows[: args.keep_first]
    kept_ids = {r["id"] for r in kept}
    half = args.size // 2

    # Flickr30k: concept-quota images first, then balance scene x style
    fk_rest = [r for r in rows if r["source"] == "flickr30k" and r["id"] not in kept_ids]
    need_fk = max(0, half - sum(r["source"] == "flickr30k" for r in kept))
    concept = [r for r in fk_rest if "concept" in r["tags"]]
    grid = [r for r in fk_rest if "concept" not in r["tags"]]
    fk = round_robin(concept, need_fk // 3, key=lambda r: r["tags"]["concept"])
    fk += round_robin(grid, need_fk - len(fk), key=lambda r: (r["tags"]["scene"], r["tags"]["style"]))

    # Fashionpedia: half of each per-class minimum, then random fill
    fp_rows = [r for r in rows if r["source"] == "fashionpedia" and r["id"] not in kept_ids]
    need_fp = args.size - len(kept) - len(fk)
    mins = {c: n // 2 for c, n in cfg["dataset"]["fashionpedia"]["min_per_class"].items()}
    fp, seen = [], set()
    for cls, n in sorted(mins.items(), key=lambda kv: kv[1]):
        pool = [r for r in fp_rows if cls in r["tags"]["garments"] and r["id"] not in seen]
        rng.shuffle(pool)
        have = sum(cls in r["tags"]["garments"] for r in fp)
        for r in pool[: max(0, n - have)]:
            fp.append(r); seen.add(r["id"])
    rest = [r for r in fp_rows if r["id"] not in seen]
    rng.shuffle(rest)
    fp += rest[: max(0, need_fp - len(fp))]

    subset = kept + fk + fp[:need_fp]
    write_jsonl(manifest, subset)
    stats(subset)
    print(f"\n{len(subset)} rows -> {manifest}  (full manifest kept at {full.name})")


if __name__ == "__main__":
    main()
