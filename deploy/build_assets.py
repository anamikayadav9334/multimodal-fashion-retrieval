"""Precompute exact CLIP baseline query vectors and upload deploy assets to a HF *model* repo.

    .venv/bin/python deploy/build_assets.py                 # build deploy/build/assets
    .venv/bin/python deploy/build_assets.py --upload YellowBlue12/fashion-retrieval-assets

Assets (downloaded by the Render Docker build): qdrant index, index_meta, query cache,
int8 ONNX encoders (fashion, text) and the baseline CLIP vector cache.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "deploy" / "build" / "assets"
EXTRA = ["A white tie and a red shirt."]  # web UI example chips not in queries.yaml


def baseline_cache() -> dict:
    from fashion_retrieval.common import encoders
    from fashion_retrieval.common.config import load_config
    from fashion_retrieval.common.onnx_encoders import cache_key

    m = load_config()["models"]
    enc = encoders.baseline_encoder(m["baseline"]["arch"], m["baseline"]["pretrained"], "cpu")
    q = yaml.safe_load(open(ROOT / "eval" / "queries.yaml"))
    texts = [x["text"] for x in q["test"] + q["dev"]] + EXTRA
    vecs = enc.encode_texts(texts)
    return {"logit_scale": enc.logit_scale, "vectors": {cache_key(t): v.round(6).tolist() for t, v in zip(texts, vecs)}}


def build() -> Path:
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "onnx").mkdir(parents=True)
    shutil.copytree(ROOT / "artifacts" / "qdrant", OUT / "qdrant", ignore=shutil.ignore_patterns(".lock"))
    for f in ("index_meta.json", "query_cache.json"):
        shutil.copy(ROOT / "artifacts" / f, OUT / f)
    for d in ("fashion", "text"):
        shutil.copytree(ROOT / "artifacts" / "onnx" / d, OUT / "onnx" / d)
    (OUT / "onnx" / "baseline_cache.json").write_text(json.dumps(baseline_cache()))
    secret = re.compile(r"sk-or-v1-[a-f0-9]{20,}|hf_[A-Za-z0-9]{30,}|gh[pousr]_[A-Za-z0-9]{30,}")
    for f in OUT.rglob("*.json"):
        if secret.search(f.read_text(errors="ignore")):
            sys.exit(f"refusing: secret in {f}")
    print(f"assets -> {OUT}")
    return OUT


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--upload", help="HF model repo id")
    a = ap.parse_args()
    out = build()
    if a.upload:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(a.upload, repo_type="model", exist_ok=True)
        api.upload_folder(folder_path=str(out), repo_id=a.upload, repo_type="model", commit_message="deploy assets")
        print(f"uploaded -> https://huggingface.co/{a.upload}")
