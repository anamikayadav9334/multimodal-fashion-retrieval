"""Fashionpedia curation: ~600 images with guaranteed coverage of key garment classes.

The Kaggle mirror is in YOLO format (images/{train,test}/*.jpg + labels/{train,test}/*.txt,
29 classes in data.yaml). We read the archive *without extracting it*: parse every label file,
select images, then pull only the chosen JPEGs out of the zip.

Selection: per-class minimums first (rarest class first, so ties/capes/hoods aren't crowded out),
then fill with random full-outfit images (>= N distinct garment classes).
Garment boxes are kept under `gt` for detector/retrieval evaluation only.
"""

from __future__ import annotations

import random
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath

import yaml
from tqdm import tqdm

from ..common.io import save_normalized_image
from .kaggle_source import download_archive


def _short(name: str) -> str:
    """'top, t-shirt, sweatshirt' -> 'top'."""
    return name.split(",")[0].strip()


def parse_archive(zf: zipfile.ZipFile):
    names = zf.namelist()
    yaml_name = next(n for n in names if n.endswith("data.yaml"))
    classes = {int(k): _short(v) for k, v in yaml.safe_load(zf.read(yaml_name))["names"].items()}

    images = {PurePosixPath(n).stem: n for n in names
              if "/images/" in f"/{n}" and n.lower().endswith((".jpg", ".jpeg", ".png"))}
    labels: dict[str, list[dict]] = {}
    for n in names:
        if "/labels/" in f"/{n}" and n.endswith(".txt"):
            stem = PurePosixPath(n).stem
            if stem not in images:
                continue
            boxes = []
            for line in zf.read(n).decode().splitlines():
                parts = line.split()
                if len(parts) >= 5:
                    c, x, y, w, h = int(parts[0]), *map(float, parts[1:5])
                    boxes.append({"label": classes.get(c, str(c)), "xywhn": [round(v, 4) for v in (x, y, w, h)]})
            if boxes:
                labels[stem] = boxes
    return classes, images, labels


def select(labels: dict[str, list[dict]], fc: dict, seed: int) -> list[str]:
    rng = random.Random(seed)
    by_class: dict[str, list[str]] = defaultdict(list)
    for stem, boxes in labels.items():
        for c in {b["label"] for b in boxes}:
            by_class[c].append(stem)

    chosen: list[str] = []
    seen: set[str] = set()
    mins: dict[str, int] = fc.get("min_per_class", {})
    for cls in sorted(mins, key=lambda c: len(by_class.get(c, []))):  # rarest first
        pool = by_class.get(cls, [])[:]
        rng.shuffle(pool)
        have = sum(1 for s in chosen if any(b["label"] == cls for b in labels[s]))
        for s in pool:
            if have >= mins[cls] or len(chosen) >= fc["target"]:
                break
            if s not in seen:
                chosen.append(s); seen.add(s); have += 1

    min_k = fc.get("min_garment_classes", 1)
    rest = [s for s, b in labels.items() if s not in seen and len({x["label"] for x in b}) >= min_k]
    rng.shuffle(rest)
    chosen += rest[: max(0, fc["target"] - len(chosen))]
    return chosen


def build(cfg: dict, raw_dir: Path, images_dir: Path, max_side: int, quality: int, seed: int,
          keep_archive: bool = False) -> list[dict]:
    fc = cfg["dataset"]["fashionpedia"]
    ref = fc["kaggle_ref"]
    archive = download_archive(ref, raw_dir / "fashionpedia")
    rows: list[dict] = []
    with zipfile.ZipFile(archive) as zf:
        classes, images, labels = parse_archive(zf)
        chosen = select(labels, fc, seed)
        print(f"[fashionpedia] {len(labels)} labelled images, {len(classes)} classes -> {len(chosen)} selected")
        for stem in tqdm(chosen, desc="fashionpedia images"):
            rid = f"fp_{stem}"
            out = images_dir / f"{rid}.jpg"
            w, h = save_normalized_image(zf.read(images[stem]), out, max_side, quality)
            garments = sorted({b["label"] for b in labels[stem]})
            rows.append({
                "id": rid, "source": "fashionpedia", "path": out.name, "width": w, "height": h,
                "license": fc["license"], "origin": f"kaggle:{ref}/{images[stem]}",
                "tags": {"garments": garments},
                "gt": {"boxes": labels[stem]},
            })
    if not keep_archive:
        archive.unlink()
        print(f"[fashionpedia] deleted archive {archive.name}")
    return rows
