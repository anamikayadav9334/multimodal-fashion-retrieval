"""The systems compared in evaluation (all share one loaded Retriever).

baseline        vanilla CLIP ViT-B/32 text->image                      (the assignment's "vanilla CLIP")
fashion         Marqo-FashionSigLIP text->image                         (better encoder, no ML logic)
hybrid          full system: LLM parser + 3-channel recall + re-rank    (ours)
Ablations of `hybrid` (one component removed at a time):
  -binding      no color<->garment binding feature
  -text         no caption dense + BM25 features (image vectors + structure only)
  -scene        no scene feature
  rules         rule-based parser instead of the LLM
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fashion_retrieval.common.config import load_config  # noqa: E402
from fashion_retrieval.retriever.search import Retriever  # noqa: E402

MAIN = ["baseline", "fashion", "hybrid"]
ABLATIONS = ["-binding", "-text", "-scene", "rules"]


def make_runner(weights: dict | None = None):
    cfg = load_config()
    R = Retriever(cfg)
    base_w = dict(weights or cfg["retriever"]["weights"])

    def run(system: str, query: str, k: int = 10) -> list[str]:
        if system in ("baseline", "fashion"):
            out = R.search(query, k=k, mode=system)
        elif system == "hybrid":
            out = R.search(query, k=k, weights=base_w)
        elif system == "rules":
            out = R.search(query, k=k, weights=base_w, use_llm=False)
        else:
            drop = {"-binding": ["binding"], "-text": ["caption", "bm25"], "-scene": ["scene"]}[system]
            out = R.search(query, k=k, weights={n: (0.0 if n in drop else v) for n, v in base_w.items()})
        return [r["id"] for r in out["results"]]

    return R, run
