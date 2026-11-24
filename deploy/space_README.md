---
title: Multimodal Fashion & Context Retrieval API
emoji: 👔
colorFrom: yellow
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

FastAPI backend for **Multimodal Fashion & Context Retrieval** — hybrid, compositional
text-to-image fashion search (FashionSigLIP + garment regions + captions + BM25, Qdrant).

- `GET /api/health` · `POST /api/search` · `GET /api/eval` · `GET /images/<file>`
- Optional secret `OPENROUTER_API_KEY` enables the LLM query parser (free models);
  without it the rule-based parser is used.
