"""Verify the int8 ONNX query encoders against the original torch models.

    .venv/bin/python deploy/check_onnx.py   -> prints token equality, embedding cosine, retrieval impact
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fashion_retrieval.common.config import load_config  # noqa: E402
from fashion_retrieval.common.store import get_client  # noqa: E402
from fashion_retrieval.retriever.search import Retriever  # noqa: E402


def dcg(g):
    return sum((2 ** x - 1) / math.log2(i + 2) for i, x in enumerate(g))


def main() -> None:
    cfg = load_config()
    q = yaml.safe_load(open(ROOT / "eval" / "queries.yaml"))
    texts = [x["text"] for x in q["test"] + q["dev"]] + ["a photo of a bright yellow raincoat", "a photo of a red tie",
                                                          "Someone's \"casual\" look_test &amp; more!!"]
    client = get_client(cfg)
    T = Retriever(cfg, client=client, runtime="torch")
    O = Retriever(cfg, client=client, runtime="onnx")

    print("1) tokenization identical to open_clip:")
    for name in ("fashion", "baseline"):
        ref = getattr(T, name).tokenizer(texts).numpy()
        mine = getattr(O, name).tokenize(texts)
        print(f"   {name:9s} {np.array_equal(ref, mine)}")

    print("2) embedding cosine (torch vs int8 onnx), min / mean:")
    for name, a, b in [("fashion", T.fashion.encode_texts(texts), O.fashion.encode_texts(texts)),
                       ("baseline", T.baseline.encode_texts(texts), O.baseline.encode_texts(texts)),
                       ("text", T.text.encode_queries(texts), O.text.encode_queries(texts))]:
        c = (a * b).sum(1)
        print(f"   {name:9s} {c.min():.4f} / {c.mean():.4f}")

    print("3) retrieval on the 5 assignment queries (hybrid, same parsed queries):")
    pools = json.loads((ROOT / "eval" / "pools.json").read_text())
    grades = yaml.safe_load(open(ROOT / "eval" / "grades.yaml"))
    tot = {"torch": [], "onnx": []}
    for item in q["test"]:
        g = dict(zip(pools[item["id"]]["ids"], grades[item["id"]]))
        ideal = sorted(grades[item["id"]], reverse=True)[:10]
        runs = {k: [r["id"] for r in R.search(item["text"], k=10)["results"]] for k, R in (("torch", T), ("onnx", O))}
        overlap = len(set(runs["torch"]) & set(runs["onnx"]))
        line = f"   {item['id']} overlap@10={overlap}/10"
        for k, ids in runs.items():
            gs = [g.get(i, 0) for i in ids]
            nd, p5 = dcg(gs) / dcg(ideal), sum(x == 2 for x in gs[:5]) / 5
            tot[k].append((nd, p5))
            line += f"  {k}: nDCG@10={nd:.2f} P@5={p5:.2f}"
        unj = sum(i not in g for i in runs["onnx"])
        print(line + (f"  ({unj} onnx results unjudged -> counted 0)" if unj else ""))
    for k, v in tot.items():
        print(f"   mean {k:5s}: nDCG@10={np.mean([x[0] for x in v]):.3f}  P@5={np.mean([x[1] for x in v]):.3f}")
    client.close()


if __name__ == "__main__":
    main()
