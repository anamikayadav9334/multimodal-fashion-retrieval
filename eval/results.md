# Evaluation results (5 assignment queries, image-judged)

| System | P@5 | P@10 | Rel@5 | nDCG@5 | nDCG@10 |
|---|---|---|---|---|---|
| Vanilla CLIP ViT-B/32 | 0.12 | 0.16 | 0.80 | 0.41 | 0.47 |
| FashionSigLIP only | 0.20 | 0.18 | 0.92 | 0.54 | 0.60 |
| Ours (hybrid) | 0.48 | 0.38 | 0.88 | 0.77 | 0.80 |
| Ours − binding | 0.44 | – | 0.92 | 0.69 | – |
| Ours − caption/BM25 | 0.44 | – | 0.92 | 0.74 | – |
| Ours − scene | 0.36 | – | 0.88 | 0.68 | – |
| Ours, rule parser (no LLM) | 0.56 | – | 0.88 | 0.80 | – |

## Per query (nDCG@10 / P@5)

| Query | Vanilla CLIP | FashionSigLIP | Ours |
|---|---|---|---|
| T1 attribute: A person in a bright yellow raincoat. | 0.31 / 0.20 | 0.73 / 0.40 | 0.84 / 0.60 |
| T2 contextual: Professional business attire inside a modern office. | 0.47 / 0.00 | 0.55 / 0.20 | 0.89 / 0.80 |
| T3 complex_semantic: Someone wearing a blue shirt sitting on a park bench. | 0.54 / 0.00 | 0.62 / 0.00 | 0.70 / 0.20 |
| T4 style_inference: Casual weekend outfit for a city walk. | 0.50 / 0.40 | 0.58 / 0.40 | 0.77 / 0.60 |
| T5 compositional: A red tie and a white shirt in a formal setting. | 0.54 / 0.00 | 0.51 / 0.00 | 0.80 / 0.20 |
