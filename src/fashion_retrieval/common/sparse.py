"""BM25 as Qdrant sparse vectors.

Document side stores the BM25 term-frequency part tf*(k1+1) / (tf + k1*(1-b+b*dl/avgdl));
Qdrant's `Modifier.IDF` multiplies in the IDF at query time; the query side is a bag of
unique terms with weight 1. The dot product is then exactly BM25 — computed inside the vector
DB with an inverted index, so it scales to millions of docs (no in-memory rank_bm25 object).
"""

from __future__ import annotations

import re
import zlib
from collections import Counter

_WORD = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_STOP = set("""a an the and or of in on at to with for from by is are was were be been being this that these
those it its he she they his her their him them there here as into over under while who which what image
shows picture photo appears also can seen background front behind""".split())


def _stem(w: str) -> str:
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def tokenize(text: str) -> list[str]:
    return [_stem(w) for w in _WORD.findall(text.lower()) if w not in _STOP]


def term_id(term: str) -> int:
    return zlib.crc32(term.encode()) & 0x7FFFFFFF


def doc_vector(tokens: list[str], avgdl: float, k1: float, b: float) -> tuple[list[int], list[float]]:
    tf = Counter(tokens)
    dl = len(tokens) or 1
    norm = k1 * (1 - b + b * dl / avgdl)
    idx, val = [], []
    for t, f in tf.items():
        idx.append(term_id(t))
        val.append(round(f * (k1 + 1) / (f + norm), 4))
    return idx, val


def query_vector(text: str) -> tuple[list[int], list[float]]:
    terms = sorted(set(tokenize(text)))
    return [term_id(t) for t in terms], [1.0] * len(terms)
