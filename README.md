# Multimodal Fashion & Context Retrieval

Natural-language search over fashion images that understands **what** someone wears, **where** they are, and the **vibe** of the outfit — with region-level color↔garment binding to fix CLIP's compositionality failures ("red tie + white shirt" ≠ "white tie + red shirt").

```
src/fashion_retrieval/
  common/     config, shared vocabulary (scenes / styles / colors), I/O
  data/       dataset sourcing from Kaggle (Fashionpedia + Flickr30k) -> data/manifest.jsonl
  indexer/    Part A — feature extraction + vector storage
  retriever/  Part B — query parsing, recall, compositional re-ranking
api/          FastAPI service
web/          Next.js UI
eval/         evaluation queries, labels, metrics (baseline CLIP vs ours)
configs/      all tunables (paths, models, weights) — logic is separated from data
```

## Setup

```bash
uv venv --python 3.11 && uv pip install -e ".[data]"
cp .env.example .env            # optional: OPENROUTER_API_KEY for the LLM query parser
# Kaggle auth: ~/.kaggle/access_token (or kaggle.json)
```

## 1. Build the dataset

```bash
.venv/bin/python -m fashion_retrieval.data.build --source all
```

- **Fashionpedia** (~600, CC BY 4.0): garment-rich images; per-class minimums guarantee ties, coats, hoods, etc.
- **Flickr30k** (~500, non-commercial research use): people in office / street / park / home scenes, stratified
  on a 4×3 scene×style grid using caption keywords (≥2 of 5 annotators must agree on the scene).

Ground-truth captions/boxes are stored in the manifest **for evaluation only** — the indexer never reads them.
