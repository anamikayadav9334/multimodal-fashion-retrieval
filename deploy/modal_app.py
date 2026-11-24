"""Serverless deployment of the retrieval API on Modal (free monthly credits).

    .venv/bin/python deploy/deploy_space.py --build-only      # assemble deploy/build/space
    .venv/bin/modal deploy deploy/modal_app.py                # -> https://<workspace>--fashion-retrieval-api-api.modal.run

Auth: MODAL_TOKEN_ID / MODAL_TOKEN_SECRET env vars (or `modal token new`).
OPENROUTER_API_KEY is read from the local environment at deploy time and stored as a Modal
secret (never baked into the image or the repo). Without it the rule-based parser is used.
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

BUNDLE = Path(__file__).resolve().parent / "build" / "space"


def warm_models() -> None:
    """Download query-time encoders into the image so cold starts don't hit the network."""
    import open_clip
    from sentence_transformers import SentenceTransformer

    open_clip.create_model_and_transforms("hf-hub:Marqo/marqo-fashionSigLIP")
    open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    SentenceTransformer("BAAI/bge-small-en-v1.5")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "torchvision", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install(
        "open_clip_torch>=2.26", "transformers>=4.56", "timm", "einops", "qdrant-client>=1.12",
        "scikit-learn>=1.4", "scikit-image>=0.22", "sentence-transformers>=3.0", "fastapi>=0.110",
        "pyyaml>=6.0", "pillow>=10.0", "tqdm", "python-dotenv", "numpy>=1.26", "requests",
    )
    .env({"HF_HOME": "/root/hf", "TOKENIZERS_PARALLELISM": "false"})
    .run_function(warm_models)
    .add_local_dir(BUNDLE, "/bundle")  # src, api, configs, eval, data/images, artifacts (index)
)

_secret = {"OPENROUTER_API_KEY": os.environ["OPENROUTER_API_KEY"]} if os.environ.get("OPENROUTER_API_KEY") else {}

app = modal.App("fashion-retrieval-api")


@app.function(
    image=image,
    cpu=2.0,
    memory=4096,
    timeout=120,
    scaledown_window=600,  # stay warm 10 min after the last request
    secrets=[modal.Secret.from_dict(_secret)],
)
@modal.concurrent(max_inputs=8)
@modal.asgi_app()
def api():
    import shutil
    import sys

    # embedded Qdrant needs a writable directory (lock file) -> work on a copy in /tmp
    work = Path("/tmp/app")
    if not work.exists():
        shutil.copytree("/bundle", work)
    os.chdir(work)
    sys.path[:0] = [str(work), str(work / "src")]
    from api.main import app as fastapi_app

    return fastapi_app
