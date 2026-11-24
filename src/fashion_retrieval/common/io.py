"""Small I/O helpers: JSONL manifests and normalized image writing."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Iterable, Iterator

from PIL import Image, ImageOps


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def save_normalized_image(data: bytes | Path, out: Path, max_side: int, quality: int) -> tuple[int, int]:
    """EXIF-rotate, convert to RGB, downscale so the longest side <= max_side, save as JPEG."""
    img = Image.open(io.BytesIO(data) if isinstance(data, bytes) else data)
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, "JPEG", quality=quality, optimize=True)
    return img.size
