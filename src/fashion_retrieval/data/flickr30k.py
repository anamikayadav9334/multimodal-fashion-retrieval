"""Flickr30k curation: pick ~500 'person in a context' images, stratified by scene x style.

Why Flickr30k: Fashionpedia is mostly street/runway, but the assignment needs offices, parks and
homes. Flickr30k has real people in everyday contexts plus 5 human captions per image.

Selection strategy (cheap, transparent, reproducible):
  1. Tag each image with a scene (office/street/park/home) and a style (formal/casual/outerwear)
     from its 5 captions. A scene needs >= 2 of 5 annotators to agree (robust to one noisy caption).
  2. Score: annotator agreement + color mentions + single-subject bonus - crowd penalty
     + bonus for evaluation-relevant concepts (raincoat, tie, bench, office).
  3. Fill a 4x3 scene x style grid with equal quotas, back-fill with the best leftovers.
  4. Download the archive once (Kaggle rate-limits per-file downloads with HTTP 429), read only
     the chosen JPEGs straight out of the zip, then delete the archive.

The human captions are stored under `gt` for EVALUATION ONLY; the indexer generates its own
captions so the system behaves the same on uncaptioned images.
"""

from __future__ import annotations

import csv
import io
import random
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from tqdm import tqdm

from ..common import vocab
from ..common.io import save_normalized_image
from .kaggle_source import download_archive

PRIORITY = vocab.keyword_pattern(["raincoat", "rain jacket", "tie", "necktie", "bench", "office", "blazer", "suit"])


def parse_captions(lines: Iterable[str]) -> dict[str, list[str]]:
    caps: dict[str, list[str]] = defaultdict(list)
    reader = csv.reader(lines)
    next(reader)  # header: image_name,comment_number,comment
    for row in reader:
        if len(row) >= 3:
            caps[row[0]].append(",".join(row[2:]).strip())
    return caps


def load_captions(path: Path) -> dict[str, list[str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return parse_captions(f)


def tag_image(captions: list[str], min_scene_votes: int = 2) -> dict[str, Any] | None:
    """Return curation tags + score, or None if the image lacks a clear scene and outfit.

    min_scene_votes=0 is used for concept quotas: the concept itself is the reason to keep the
    image, so an unclear scene is tagged "other" instead of rejecting it.
    """
    def votes(patterns: dict) -> dict[str, int]:
        return {k: sum(bool(p.search(c)) for c in captions) for k, p in patterns.items()}

    scene_votes, style_votes = votes(vocab.SCENE_PATTERNS), votes(vocab.STYLE_PATTERNS)
    scene, s_n = max(scene_votes.items(), key=lambda kv: kv[1])
    style, t_n = max(style_votes.items(), key=lambda kv: kv[1])
    if s_n < min_scene_votes or t_n < 1:
        return None
    if s_n == 0:
        scene = "other"

    colors = sorted({vocab.normalize_color(m.group(1)) for c in captions for m in vocab.COLOR_PATTERN.finditer(c)})
    single = sum(bool(vocab.SINGLE_SUBJECT_PREFIX.match(c)) for c in captions)
    crowd = sum(bool(vocab.CROWD_PATTERN.search(c)) for c in captions)
    priority = sum(bool(PRIORITY.search(c)) for c in captions)
    if crowd >= 3:  # mostly-crowd scenes make "the person in X" ambiguous
        return None

    score = s_n + t_n + 0.5 * min(len(colors), 3) + 0.6 * single - 1.0 * crowd + 0.8 * min(priority, 3)
    return {"scene": scene, "style": style, "colors": colors, "score": round(score, 2)}


def select(caps: dict[str, list[str]], target: int, seed: int,
           concept_quotas: dict[str, int] | None = None) -> list[tuple[str, dict]]:
    rng = random.Random(seed)
    items = list(caps.items())
    rng.shuffle(items)  # random tie-break before score sorts

    # 1) concept quotas (relaxed scene rule)
    chosen: list[tuple[str, dict]] = []
    taken: set[str] = set()
    for concept, n in (concept_quotas or {}).items():
        pat = vocab.keyword_pattern([concept])
        pool = []
        for img, c in items:
            if img in taken:
                continue
            hits = sum(bool(pat.search(x)) for x in c)
            t = tag_image(c, min_scene_votes=0) if hits else None
            if t:
                t = {**t, "concept": concept}
                pool.append((hits + t["score"], img, t))
        pool.sort(key=lambda x: x[0], reverse=True)
        for _, img, t in pool[:n]:
            chosen.append((img, t)); taken.add(img)

    # 2) balanced scene x style grid over the remaining budget
    cells: dict[tuple[str, str], list[tuple[str, dict]]] = defaultdict(list)
    for img, c in items:
        if img not in taken and (t := tag_image(c)):
            cells[(t["scene"], t["style"])].append((img, t))
    for v in cells.values():
        v.sort(key=lambda x: x[1]["score"], reverse=True)

    quota = (target - len(chosen)) // max(len(cells), 1)
    chosen += [x for v in cells.values() for x in v[:quota]]
    taken = {img for img, _ in chosen}
    leftovers = sorted((x for v in cells.values() for x in v[quota:] if x[0] not in taken),
                       key=lambda x: x[1]["score"], reverse=True)
    chosen += leftovers[: target - len(chosen)]
    return chosen


def build(cfg: dict, raw_dir: Path, images_dir: Path, max_side: int, quality: int, seed: int,
          keep_archive: bool = False) -> list[dict]:
    fc = cfg["dataset"]["flickr30k"]
    ref = fc["kaggle_ref"]
    archive = download_archive(ref, raw_dir / "flickr30k")
    rows: list[dict] = []
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
        cap_member = next(n for n in names if n.endswith(fc["captions_file"]))
        caps = parse_captions(io.TextIOWrapper(zf.open(cap_member), encoding="utf-8", newline=""))
        by_file = {PurePosixPath(n).name: n for n in names if n.lower().endswith(".jpg")}
        chosen = select(caps, fc["target"], seed, fc.get("concept_quotas"))
        print(f"[flickr30k] {len(caps)} images captioned -> {len(chosen)} selected")
        for img, tags in tqdm(chosen, desc="flickr30k images"):
            if img not in by_file:
                print(f"  ! missing in archive: {img}")
                continue
            rid = f"fk_{Path(img).stem}"
            out = images_dir / f"{rid}.jpg"
            w, h = save_normalized_image(zf.read(by_file[img]), out, max_side, quality)
            rows.append({
                "id": rid, "source": "flickr30k", "path": out.name, "width": w, "height": h,
                "license": fc["license"], "origin": f"kaggle:{ref}/{by_file[img]}",
                "tags": tags, "gt": {"captions": caps[img]},
            })
    if not keep_archive:
        archive.unlink()
        print(f"[flickr30k] deleted archive {archive.name}")
    return rows
