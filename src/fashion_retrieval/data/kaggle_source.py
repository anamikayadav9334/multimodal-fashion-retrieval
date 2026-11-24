"""Thin wrapper around the Kaggle API (auth via ~/.kaggle/access_token or kaggle.json)."""

from __future__ import annotations

import threading
import time
import zipfile
from pathlib import Path

_local = threading.local()


def _api():
    """One authenticated client per thread (the client holds a requests session)."""
    if not hasattr(_local, "api"):
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        _local.api = api
    return _local.api


def download_file(dataset: str, file_name: str, dest_dir: Path) -> Path:
    """Download a single file from a dataset. Kaggle sometimes wraps it in a .zip; unwrap it."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    base = Path(file_name).name
    out = dest_dir / base
    if out.exists():
        return out
    _api().dataset_download_file(dataset, file_name, path=str(dest_dir), quiet=True)
    zipped = dest_dir / f"{base}.zip"
    if zipped.exists():
        with zipfile.ZipFile(zipped) as zf:
            zf.extract(zf.namelist()[0], dest_dir)
        zipped.unlink()
    if not out.exists():
        raise FileNotFoundError(f"Kaggle download produced no file for {dataset}:{file_name}")
    return out


def _with_backoff(fn, what: str, attempts: int = 10, base: float = 60, cap: float = 600):
    """Retry on HTTP 429 (Kaggle rate limit) with exponential backoff: 1, 2, 4 ... <=10 min."""
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:  # requests.HTTPError, but avoid importing requests here
            if "429" not in str(e) or i == attempts - 1:
                raise
            wait = min(base * 2**i, cap)
            print(f"[kaggle] rate-limited on {what}; retry {i + 1}/{attempts - 1} in {wait:.0f}s", flush=True)
            time.sleep(wait)


def download_archive(dataset: str, dest_dir: Path) -> Path:
    """Download the full dataset archive (kept zipped; members are read selectively)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / f"{dataset.split('/')[1]}.zip"
    if not out.exists():
        _with_backoff(lambda: _api().dataset_download_files(dataset, path=str(dest_dir), quiet=False, unzip=False),
                      dataset)
    return out
