"""Export the three QUERY-side text encoders to int8 ONNX for a torch-free, low-memory API.

    .venv/bin/python deploy/export_onnx.py          -> artifacts/onnx/{fashion,baseline,text}/

Only text towers are exported: image vectors are already in the index, so the server never
needs the image encoders, the detector or the captioner. Each folder holds
model.onnx (int8 dynamic-quantized), tokenizer.json and meta.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from onnxruntime.quantization import QuantType, quantize_dynamic

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fashion_retrieval.common.config import load_config  # noqa: E402

OUT = ROOT / "artifacts" / "onnx"


class ClipText(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self, input_ids):
        return F.normalize(self.m.encode_text(input_ids), dim=-1)


class BertCLS(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self, input_ids, attention_mask, token_type_ids):
        h = self.m(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).last_hidden_state
        return F.normalize(h[:, 0], dim=-1)  # bge uses CLS pooling


def export(module, args, names, dyn, out_dir: Path, quantize: bool = True) -> None:
    """quantize=False keeps fp32: CLIP's text tower has activation outliers and int8 (even per-channel,
    even MatMul-only) drops its cosine to ~0.75-0.86 vs the original, which would make the baseline
    look worse than it is. SigLIP/bge quantize faithfully (retrieval unchanged, see check_onnx.py)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fp32 = out_dir / "model.fp32.onnx"
    torch.onnx.export(module, args, str(fp32), input_names=names, output_names=["emb"],
                      dynamic_axes={**dyn, "emb": {0: "batch"}}, opset_version=17, dynamo=False)
    if quantize:
        quantize_dynamic(str(fp32), str(out_dir / "model.onnx"), weight_type=QuantType.QInt8, per_channel=True)
        fp32.unlink()
    else:
        fp32.rename(out_dir / "model.onnx")
    for extra in out_dir.glob("*.data"):
        extra.unlink()
    print(f"  {out_dir.name}: {(out_dir / 'model.onnx').stat().st_size / 1e6:.0f} MB")


@torch.no_grad()
def main() -> None:
    import open_clip
    from transformers import AutoModel, AutoTokenizer

    m = load_config()["models"]

    # 1) Marqo-FashionSigLIP text tower (T5 sentencepiece tokenizer, ctx 64, pad id 1)
    model, _, _ = open_clip.create_model_and_transforms(m["fashion"])
    tok = open_clip.get_tokenizer(m["fashion"])
    d = OUT / "fashion"
    ids = tok(["a red tie and a white shirt"])
    export(ClipText(model.eval()), (ids,), ["input_ids"], {"input_ids": {0: "batch"}}, d)
    tok.tokenizer.backend_tokenizer.save(str(d / "tokenizer.json"))
    (d / "meta.json").write_text(json.dumps({"kind": "siglip", "context_length": tok.context_length,
                                             "pad_id": tok.pad_token_id, "clean": "canonicalize",
                                             "logit_scale": float(model.logit_scale.exp())}))
    del model

    # 2) Vanilla CLIP ViT-B/32 text tower (BPE, ctx 77, pad 0) - tokenizer.json from the HF CLIP repo
    model, _, _ = open_clip.create_model_and_transforms(m["baseline"]["arch"], pretrained=m["baseline"]["pretrained"])
    tok = open_clip.get_tokenizer(m["baseline"]["arch"])
    d = OUT / "baseline"
    export(ClipText(model.eval()), (tok(["a red tie"]),), ["input_ids"], {"input_ids": {0: "batch"}}, d, quantize=False)
    AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32").backend_tokenizer.save(str(d / "tokenizer.json"))
    (d / "meta.json").write_text(json.dumps({"kind": "clip", "context_length": 77, "pad_id": 0, "clean": "clip",
                                             "logit_scale": float(model.logit_scale.exp())}))
    del model

    # 3) bge-small (BERT, CLS pooling)
    d = OUT / "text"
    bt = AutoTokenizer.from_pretrained(m["text"])
    bm = AutoModel.from_pretrained(m["text"]).eval()
    enc = bt(["a red tie"], return_tensors="pt")
    export(BertCLS(bm), (enc["input_ids"], enc["attention_mask"], enc["token_type_ids"]),
           ["input_ids", "attention_mask", "token_type_ids"],
           {k: {0: "batch", 1: "seq"} for k in ("input_ids", "attention_mask", "token_type_ids")}, d)
    bt.backend_tokenizer.save(str(d / "tokenizer.json"))
    (d / "meta.json").write_text(json.dumps({"kind": "bge", "max_length": 128,
                                             "query_prefix": "Represent this sentence for searching relevant passages: "}))


if __name__ == "__main__":
    main()
