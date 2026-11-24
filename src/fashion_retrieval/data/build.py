"""Build the curated dataset + manifest.

    python -m fashion_retrieval.data.build --source all
    python -m fashion_retrieval.data.build --source flickr30k      # resumable; skips existing images
"""

from __future__ import annotations

import argparse
from collections import Counter

from ..common.config import load_config, resolve
from ..common.io import read_jsonl, write_jsonl
from . import fashionpedia, flickr30k

SOURCES = {"flickr30k": flickr30k, "fashionpedia": fashionpedia}


def stats(rows: list[dict]) -> None:
    print(f"\nTotal images: {len(rows)}  {dict(Counter(r['source'] for r in rows))}")
    fk = [r for r in rows if r["source"] == "flickr30k"]
    if fk:
        print("Flickr30k scene :", dict(Counter(r["tags"]["scene"] for r in fk)))
        print("Flickr30k style :", dict(Counter(r["tags"]["style"] for r in fk)))
        print("Flickr30k colors:", Counter(c for r in fk for c in r["tags"]["colors"]).most_common(12))
    fp = [r for r in rows if r["source"] == "fashionpedia"]
    if fp:
        print("Fashionpedia garments:", Counter(g for r in fp for g in r["tags"]["garments"]).most_common(20))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=[*SOURCES, "all"], default="all")
    ap.add_argument("--config", default=None)
    ap.add_argument("--keep-archive", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    p, d = cfg["paths"], cfg["dataset"]
    raw_dir, images_dir, manifest = resolve(p["raw_dir"]), resolve(p["images_dir"]), resolve(p["manifest"])
    common = dict(raw_dir=raw_dir, images_dir=images_dir, max_side=d["max_side"], quality=d["jpeg_quality"], seed=d["seed"])

    names = list(SOURCES) if args.source == "all" else [args.source]
    for name in names:
        rows = SOURCES[name].build(cfg, **common, keep_archive=args.keep_archive)
        write_jsonl(manifest.with_name(f"manifest_{name}.jsonl"), rows)

    # Merge whatever per-source manifests exist into the single manifest the indexer reads.
    merged = []
    for name in SOURCES:
        part = manifest.with_name(f"manifest_{name}.jsonl")
        if part.exists():
            merged += list(read_jsonl(part))
    write_jsonl(manifest, merged)
    stats(merged)
    print(f"\nManifest -> {manifest}")


if __name__ == "__main__":
    main()
