"""Dataset sourcing: Kaggle -> curated, stratified, normalized image set + JSONL manifest.

Manifest row schema (data/manifest.jsonl):
    id          unique id, e.g. "fp_10003" / "fk_1000092795"
    source      "fashionpedia" | "flickr30k"
    path        image path relative to the images dir
    width,height
    license, origin
    tags        curation tags (how the image was selected) - used for dataset stats only
    gt          ground truth (captions / garment boxes) - used ONLY for evaluation,
                never by the indexer, so retrieval stays genuinely zero-shot.
"""
