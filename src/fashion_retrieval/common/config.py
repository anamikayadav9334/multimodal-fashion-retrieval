"""Config loading. Logic never hard-codes paths or model names; everything comes from YAML."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    """Walk up from this file until we find pyproject.toml."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Could not locate project root (pyproject.toml).")


@lru_cache(maxsize=None)
def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else project_root() / "configs" / "default.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def resolve(rel: str | Path) -> Path:
    """Resolve a config-relative path against the project root."""
    p = Path(rel)
    return p if p.is_absolute() else project_root() / p
