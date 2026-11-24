"""Torch-free query encoders (int8 ONNX + HF `tokenizers`) for low-memory serving.

Same interface as common.encoders (encode_texts / encode_queries, logit_scale), so the
retriever is unchanged. Tokenization reproduces open_clip exactly (verified in
deploy/check_onnx.py): SigLIP = canonicalize + T5 sentencepiece, ctx 64, pad 1;
CLIP = whitespace-clean + lowercase BPE, ctx 77, pad 0.
"""

from __future__ import annotations

import html
import json
import string
from functools import lru_cache
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

_PUNCT = str.maketrans("", "", string.punctuation)


def _basic_clean(t: str) -> str:
    try:
        import ftfy

        t = ftfy.fix_text(t)
    except ImportError:
        pass
    return html.unescape(html.unescape(t)).strip()


def _clean(t: str, mode: str) -> str:
    t = _basic_clean(t)
    if mode == "canonicalize":
        t = t.replace("_", " ").translate(_PUNCT).lower()
    elif mode == "clip":
        t = t.lower()
    return " ".join(t.split()).strip()


def _session(path: Path) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    so.enable_mem_pattern = False
    so.enable_cpu_mem_arena = False  # keeps peak RSS ~400 MB (fits 512 MB free hosting)
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def cache_key(text: str) -> str:
    return " ".join(text.lower().split())


class CachedClipText:
    """Baseline CLIP query vectors precomputed with the ORIGINAL torch model (exact, not quantized).
    Used on 512 MB hosting where the fp32 CLIP text tower does not fit. Unknown queries raise KeyError."""

    def __init__(self, path: Path):
        d = json.loads(path.read_text())
        self.logit_scale = d["logit_scale"]
        self.vecs = {k: np.array(v, np.float32) for k, v in d["vectors"].items()}

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        return np.stack([self.vecs[cache_key(t)] for t in texts])


class OnnxClipText:
    def __init__(self, folder: Path):
        self.meta = json.loads((folder / "meta.json").read_text())
        self.tok = Tokenizer.from_file(str(folder / "tokenizer.json"))
        self.tok.no_padding()
        self.tok.no_truncation()
        self.sess = _session(folder / "model.onnx")
        self.logit_scale = self.meta["logit_scale"]
        self.ctx, self.pad = self.meta["context_length"], self.meta["pad_id"]

    def tokenize(self, texts: list[str]) -> np.ndarray:
        out = np.full((len(texts), self.ctx), self.pad, dtype=np.int64)
        for i, t in enumerate(texts):
            ids = self.tok.encode(_clean(t, self.meta["clean"])).ids
            if len(ids) > self.ctx:  # keep the end-of-text token, as open_clip does
                ids = ids[: self.ctx - 1] + [ids[-1]]
            out[i, : len(ids)] = ids
        return out

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        return self.sess.run(None, {"input_ids": self.tokenize(texts)})[0].astype(np.float32)


class OnnxTextEncoder:
    def __init__(self, folder: Path):
        self.meta = json.loads((folder / "meta.json").read_text())
        self.tok = Tokenizer.from_file(str(folder / "tokenizer.json"))
        self.tok.enable_truncation(self.meta["max_length"])
        self.tok.enable_padding(pad_id=0)
        self.sess = _session(folder / "model.onnx")

    def _run(self, texts: list[str]) -> np.ndarray:
        enc = self.tok.encode_batch(texts)
        ids = np.array([e.ids for e in enc], np.int64)
        mask = np.array([e.attention_mask for e in enc], np.int64)
        return self.sess.run(None, {"input_ids": ids, "attention_mask": mask,
                                    "token_type_ids": np.zeros_like(ids)})[0].astype(np.float32)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._run([self.meta["query_prefix"] + t for t in texts])

    def encode_docs(self, texts: list[str]) -> np.ndarray:
        return self._run(texts)


@lru_cache(maxsize=None)
def load(root: str):
    r = Path(root)
    baseline = (OnnxClipText(r / "baseline") if (r / "baseline" / "model.onnx").exists()
                else CachedClipText(r / "baseline_cache.json"))
    return OnnxClipText(r / "fashion"), baseline, OnnxTextEncoder(r / "text")
