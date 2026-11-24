"""Compute retrieval metrics from pooled runs + image-based grades.

    python eval/metrics.py        -> eval/results.json, eval/results.md

P@k     fraction of top-k with grade 2 (strict: fully matches)
Rel@k   fraction of top-k with grade >= 1 (at least partially relevant)
nDCG@k  graded (gain 2^g - 1), ideal ranking from all judged images of the query
Main systems are judged to depth 10, ablations to depth 5 (pool depth).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import yaml

HERE = Path(__file__).parent
SYSTEMS = [("baseline", 10), ("fashion", 10), ("hybrid", 10),
           ("-binding", 5), ("-text", 5), ("-scene", 5), ("rules", 5)]
NAMES = {"baseline": "Vanilla CLIP ViT-B/32", "fashion": "FashionSigLIP only", "hybrid": "Ours (hybrid)",
         "-binding": "Ours − binding", "-text": "Ours − caption/BM25", "-scene": "Ours − scene",
         "rules": "Ours, rule parser (no LLM)"}


def dcg(gains: list[int]) -> float:
    return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(gains))


def main() -> None:
    pools = json.loads((HERE / "pools.json").read_text())
    grades_raw = yaml.safe_load(open(HERE / "grades.yaml"))
    res: dict = {}
    for qid, gl in grades_raw.items():
        ids = pools[qid]["ids"]
        assert len(ids) == len(gl), f"{qid}: {len(ids)} pooled vs {len(gl)} grades"
        g = dict(zip(ids, gl))
        ideal = sorted(gl, reverse=True)
        res[qid] = {"query": pools[qid]["query"], "type": pools[qid]["type"]}
        for s, depth in SYSTEMS:
            run = [g.get(i, 0) for i in pools[qid]["runs"][s][:depth]]
            m = {}
            for k in (5, 10):
                if k <= depth:
                    m[f"P@{k}"] = sum(x == 2 for x in run[:k]) / k
                    m[f"Rel@{k}"] = sum(x >= 1 for x in run[:k]) / k
                    m[f"nDCG@{k}"] = dcg(run[:k]) / (dcg(ideal[:k]) or 1)
            res[qid][s] = m

    qids = list(grades_raw)
    mean = {s: {m: sum(res[q][s][m] for q in qids) / len(qids) for m in res[qids[0]][s]} for s, _ in SYSTEMS}
    (HERE / "results.json").write_text(json.dumps({"per_query": res, "mean": mean}, indent=1))

    lines = ["# Evaluation results (5 assignment queries, image-judged)", "",
             "| System | P@5 | P@10 | Rel@5 | nDCG@5 | nDCG@10 |", "|---|---|---|---|---|---|"]
    for s, _ in SYSTEMS:
        m = mean[s]
        f = lambda k: f"{m[k]:.2f}" if k in m else "–"  # noqa: E731
        lines.append(f"| {NAMES[s]} | {f('P@5')} | {f('P@10')} | {f('Rel@5')} | {f('nDCG@5')} | {f('nDCG@10')} |")
    lines += ["", "## Per query (nDCG@10 / P@5)", "", "| Query | Vanilla CLIP | FashionSigLIP | Ours |", "|---|---|---|---|"]
    for q in qids:
        r = res[q]
        cell = lambda s: f"{r[s]['nDCG@10']:.2f} / {r[s]['P@5']:.2f}"  # noqa: E731
        lines.append(f"| {q} {r['type']}: {r['query']} | {cell('baseline')} | {cell('fashion')} | {cell('hybrid')} |")
    (HERE / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
