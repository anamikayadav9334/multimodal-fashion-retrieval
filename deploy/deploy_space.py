"""Assemble and upload the Hugging Face Space (backend).

    huggingface-cli login            # once, with a WRITE token
    .venv/bin/python deploy/deploy_space.py --space <your-hf-username>/fashion-retrieval-api

Builds deploy/build/space/ with only what the API needs (no detector/captioner, only the
indexed images), then uploads it. Your OPENROUTER key is NOT uploaded - add it as a Space
secret in the Space settings if you want the LLM parser.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "deploy" / "build" / "space"


def assemble() -> Path:
    if BUILD.exists():
        shutil.rmtree(BUILD)
    BUILD.mkdir(parents=True)
    shutil.copy(ROOT / "deploy" / "Dockerfile", BUILD / "Dockerfile")
    shutil.copy(ROOT / "deploy" / "space_README.md", BUILD / "README.md")
    shutil.copy(ROOT / "pyproject.toml", BUILD)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    for d in ("src", "configs", "api"):
        shutil.copytree(ROOT / d, BUILD / d, ignore=ignore)
    (BUILD / "eval").mkdir()
    for f in ("queries.yaml", "results.json"):
        shutil.copy(ROOT / "eval" / f, BUILD / "eval" / f)

    # only images that are actually indexed
    ids = [json.loads(l)["path"] for l in open(ROOT / "data" / "manifest.jsonl") if l.strip()]
    (BUILD / "data" / "images").mkdir(parents=True)
    for p in ids:
        shutil.copy(ROOT / "data" / "images" / p, BUILD / "data" / "images" / p)

    art = BUILD / "artifacts"
    shutil.copytree(ROOT / "artifacts" / "qdrant", art / "qdrant", ignore=shutil.ignore_patterns(".lock"))
    for f in ("index_meta.json", "query_cache.json"):
        if (ROOT / "artifacts" / f).exists():
            shutil.copy(ROOT / "artifacts" / f, art / f)

    # safety: never ship secrets
    for f in BUILD.rglob("*"):
        if f.is_file() and f.suffix in {".py", ".yaml", ".json", ".md", ".toml"} and "sk-or-v1" in f.read_text(errors="ignore"):
            sys.exit(f"refusing to upload: secret found in {f}")
    print(f"assembled {sum(1 for _ in BUILD.rglob('*') if _.is_file())} files, {len(ids)} images -> {BUILD}")
    return BUILD


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", required=False, help="e.g. yourname/fashion-retrieval-api")
    ap.add_argument("--build-only", action="store_true")
    args = ap.parse_args()
    folder = assemble()
    if args.build_only or not args.space:
        return
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.space, repo_type="space", space_sdk="docker", exist_ok=True)
    api.upload_folder(folder_path=str(folder), repo_id=args.space, repo_type="space",
                      commit_message="Deploy retrieval API")
    print(f"uploaded -> https://huggingface.co/spaces/{args.space}")
    print(f"API base URL -> https://{args.space.replace('/', '-').replace('_', '-').lower()}.hf.space")


if __name__ == "__main__":
    main()
