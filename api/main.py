"""FastAPI service exposing the retriever.

    uvicorn api.main:app --port 8000          (from the project root)

GET  /api/health          status + index size
GET  /api/examples        the 5 assignment queries + dev queries
GET  /api/eval            evaluation results (eval/results.json)
POST /api/search          {"query", "k", "mode": hybrid|fashion|baseline, "compare": bool}
GET  /images/<file>       dataset images
"""

from __future__ import annotations

import json
import os
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from fashion_retrieval.common.config import load_config, resolve  # noqa: E402
from fashion_retrieval.retriever.search import MODES, Retriever  # noqa: E402

STATE: dict = {}
LOCK = threading.Lock()  # embedded Qdrant + MPS are not re-entrant; requests are fast anyway


@asynccontextmanager
async def lifespan(_: FastAPI):
    cfg = load_config()
    STATE["retriever"] = Retriever(cfg)
    meta = resolve(cfg["paths"]["artifacts_dir"]) / "index_meta.json"
    STATE["meta"] = json.loads(meta.read_text()) if meta.exists() else {}
    yield
    STATE["retriever"].client.close()


app = FastAPI(title="Multimodal Fashion & Context Retrieval", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.mount("/images", StaticFiles(directory=ROOT / "data" / "images"), name="images")


class SearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=300)
    k: int = Field(default=12, ge=1, le=48)
    mode: str = "hybrid"
    compare: bool = False
    use_llm: bool = True


@app.get("/api/health")
def health():
    m = STATE.get("meta", {})
    return {"status": "ok", "images": m.get("images"), "regions": m.get("regions"), "built_at": m.get("built_at")}


@app.get("/api/examples")
def examples():
    q = yaml.safe_load(open(ROOT / "eval" / "queries.yaml"))
    return {"test": q["test"], "dev": q["dev"]}


@app.get("/api/eval")
def evaluation():
    p = ROOT / "eval" / "results.json"
    if not p.exists():
        raise HTTPException(404, "run eval/metrics.py first")
    return json.loads(p.read_text())


@app.post("/api/search")
def search(req: SearchRequest):
    if req.mode not in MODES:
        raise HTTPException(400, f"mode must be one of {MODES}")
    R: Retriever = STATE["retriever"]
    with LOCK:
        out = R.search(req.query, k=req.k, mode=req.mode, use_llm=req.use_llm)
        if req.compare:
            other = "baseline" if req.mode != "baseline" else "hybrid"
            out = {"primary": out, "compare": R.search(req.query, k=req.k, mode=other)}
    return out
