"""Qdrant schema + client. One place defines how vectors are laid out.

Collection `images` (one point per image)
    vectors:  global   (FashionSigLIP image, 768)   - overall look / vibe
              baseline (vanilla CLIP image, 512)    - for the baseline comparison only
              caption  (bge-small, 384)             - dense caption semantics
    sparse:   bm25     (Modifier.IDF)               - exact-term caption matching
    payload:  image_id, path, source, caption, scene{...}, scene_top, garments[], pairs[], ...

Collection `regions` (one point per detected garment)
    vectors:  crop (FashionSigLIP crop, 768)
    payload:  image_id, garment, score, box, area, colors{...}

Embedded mode (path) for local dev; set qdrant.url for a server with HNSW at large scale.
"""

from __future__ import annotations

import uuid

from qdrant_client import QdrantClient, models

from .config import resolve

_NS = uuid.UUID("6f1c9d3e-2b7a-4c55-9e1a-2d3b4c5d6e7f")


def point_id(key: str) -> str:
    """Deterministic UUID per image/region id -> idempotent upserts."""
    return str(uuid.uuid5(_NS, key))


def get_client(cfg: dict) -> QdrantClient:
    url = cfg["qdrant"].get("url")
    return QdrantClient(url=url) if url else QdrantClient(path=str(resolve(cfg["paths"]["qdrant_path"])))


def create_collections(client: QdrantClient, cfg: dict, dims: dict[str, int]) -> None:
    q = cfg["qdrant"]
    cos = models.Distance.COSINE
    for name in (q["images_collection"], q["regions_collection"]):
        if client.collection_exists(name):
            client.delete_collection(name)

    client.create_collection(
        q["images_collection"],
        vectors_config={
            "global": models.VectorParams(size=dims["global"], distance=cos),
            "baseline": models.VectorParams(size=dims["baseline"], distance=cos),
            "caption": models.VectorParams(size=dims["caption"], distance=cos),
        },
        sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )
    client.create_collection(
        q["regions_collection"],
        vectors_config={"crop": models.VectorParams(size=dims["crop"], distance=cos)},
    )
    # Payload indexes: used for filtered search in server mode (no-op in embedded mode).
    for coll, field in [(q["regions_collection"], "image_id"), (q["regions_collection"], "garment"),
                        (q["images_collection"], "scene_top"), (q["images_collection"], "garments"),
                        (q["images_collection"], "pairs")]:
        try:
            client.create_payload_index(coll, field, models.PayloadSchemaType.KEYWORD)
        except Exception:
            pass
