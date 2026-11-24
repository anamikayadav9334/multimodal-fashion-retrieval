# Multimodal Fashion & Context Retrieval

Natural-language search over fashion images that understands **what** someone wears, **where** they are,
and the **vibe** of the outfit — and binds colors to the right garment, fixing CLIP's compositionality
failure ("a red tie and a white shirt" ≠ "a white tie and a red shirt").

| | Vanilla CLIP | FashionSigLIP only | **Ours** |
|---|---|---|---|
| P@5 (full matches, 5 assignment queries) | 0.12 | 0.20 | **0.48** |
| nDCG@10 (image-judged, graded) | 0.47 | 0.60 | **0.80** |

See [eval/results.md](eval/results.md) and [docs/report.pdf](docs/report.pdf).

## Architecture

```
            ┌──────────────── Part A: INDEXER (offline, src/fashion_retrieval/indexer) ───────────────┐
 images ──► │ FashionSigLIP global vector     ── style / vibe                                         │
            │ YOLOS-Fashionpedia garments ──► crop vectors + colors (LAB k-means ⊕ zero-shot)         │
            │ CLIP prompt-ensemble scenes     ── office / street / park / home / studio / other        │
            │ Florence-2 captions ──► bge dense vector + BM25 sparse vector + (color,garment) pairs    │
            └────────────► Qdrant: `images` (3 dense + 1 sparse vectors) + `regions` (crop vectors) ───┘

            ┌──────────────── Part B: RETRIEVER (online, src/fashion_retrieval/retriever) ────────────┐
 query ───► │ parse: OpenRouter free LLM ⟶ fallback rule parser  → garments{color}, scene, style       │
            │ recall: FashionSigLIP ∥ caption-dense ∥ BM25  → RRF → 200 candidates                     │
            │ re-rank: global + caption + bm25 + scene + BINDING(garment ∧ its color, region|caption)  │
            └──────────────► top-k with per-feature scores and garment boxes ("why it matched") ──────┘
```

## Repository layout

```
src/fashion_retrieval/
  common/     config, shared vocabulary, color↔garment binding parser, encoders, Qdrant schema, BM25
  data/       Kaggle sourcing (Fashionpedia + Flickr30k), stratified curation, subset
  indexer/    Part A — detect, color, scene, caption, extract (stage 1), load (stage 2), run
  retriever/  Part B — query parsing, scoring (binding), hybrid search, CLI
api/          FastAPI service           web/   Next.js UI (search, compare, evaluation)
eval/         queries, pooled runs, contact sheets, grades, metrics
configs/      every model name, path, threshold and weight (logic is separated from data)
deploy/       Hugging Face Space Dockerfile + upload script
```

## Run it

```bash
uv venv --python 3.11 && uv pip install -e ".[data,ml,api]"
cp .env.example .env                    # optional OPENROUTER_API_KEY (free models)

python -m fashion_retrieval.data.build --source all          # Kaggle auth: ~/.kaggle/access_token
python -m fashion_retrieval.data.subset --size 600           # optional laptop-sized subset
python -m fashion_retrieval.indexer.run                      # Part A (resumable)
python -m fashion_retrieval.retriever.cli "a red tie and a white shirt in a formal setting" -k 5   # Part B
python -m fashion_retrieval.retriever.cli "..." --mode baseline                                  # vanilla CLIP

uvicorn api.main:app --port 8000                             # API
cd web && npm install && npm run dev                         # UI on http://localhost:3000
python eval/pool.py && python eval/metrics.py                # evaluation
```

## Data

- **Fashionpedia** (CC BY 4.0) — garment-rich images; per-class minimums guarantee ties, coats, hoods, etc.
- **Flickr30k** (non-commercial research use) — people in office / street / park / home scenes, stratified
  on a scene × style grid from captions, plus quotas for evaluation-critical concepts (raincoat, office, tie).

Human captions / boxes are stored **for evaluation only**; the indexer never reads them (true zero-shot).
Images are not committed — rebuild with the data scripts.
