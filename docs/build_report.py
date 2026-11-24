"""Build the submission PDF (docs/report.pdf) from real evaluation outputs.

    python docs/build_report.py --github <url> --demo <url>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image as RLImage, PageBreak, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

ROOT = Path(__file__).resolve().parents[1]
DOCS, EVAL, IMAGES = ROOT / "docs", ROOT / "eval", ROOT / "data" / "images"
FIG = DOCS / "figures"
ACCENT = colors.HexColor("#c2410c")


def font(size: int):
    try:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
    except OSError:
        return ImageFont.load_default()


# ------------------------------------------------------------------ figures
def architecture_png(out: Path) -> None:
    W, H = 1800, 860
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)

    def box(x, y, w, h, title, lines, fill):
        d.rounded_rectangle([x, y, x + w, y + h], 18, fill=fill, outline="#9a8f7d", width=2)
        d.text((x + 18, y + 12), title, fill="#16130f", font=font(28))
        for i, t in enumerate(lines):
            d.text((x + 18, y + 54 + i * 31), t, fill="#3b352d", font=font(22))

    def arrow(x0, y0, x1, y1):
        d.line([x0, y0, x1, y1], fill="#c2410c", width=5)
        d.polygon([(x1, y1), (x1 - 16, y1 - 10), (x1 - 16, y1 + 10)], fill="#c2410c")

    d.text((30, 20), "Part A - Indexer (offline)", fill="#c2410c", font=font(32))
    box(30, 70, 250, 150, "Images", ["Fashionpedia 300", "Flickr30k 300"], "#f3efe7")
    box(360, 70, 600, 330, "Feature extraction", [
        "FashionSigLIP global vector (style/vibe)",
        "YOLOS-Fashionpedia garment boxes",
        "  -> crop vectors (FashionSigLIP)",
        "  -> color: LAB k-means + CIEDE2000",
        "     fused with zero-shot crop color",
        "CLIP prompt-ensemble scene scores",
        "Florence-2 detailed captions",
        "  -> bge vector, BM25, (color,garment)"], "#fdf6ee")
    box(1040, 70, 720, 330, "Qdrant", [
        "images: global, caption, baseline (dense)",
        "        bm25 (sparse, IDF modifier)",
        "        payload: scene, pairs, caption",
        "regions: crop vector + garment, box,",
        "         color distribution, image_id",
        "", "Embedded locally; server + HNSW at scale"], "#eef4f8")
    arrow(280, 145, 355, 145)
    arrow(960, 235, 1035, 235)

    d.text((30, 440), "Part B - Retriever (online)", fill="#c2410c", font=font(32))
    box(30, 490, 380, 330, "1. Parse query", [
        "OpenRouter free LLM (JSON)", "fallback: rule parser", "same vocab as indexer",
        "-> garments{colors}", "-> scene, style", "-> visual expansion"], "#fdf6ee")
    box(470, 490, 420, 330, "2. Hybrid recall", [
        "FashionSigLIP text->image", "bge query->caption", "BM25 over captions",
        "each top-200", "-> reciprocal rank fusion", "-> 200 candidates"], "#eef4f8")
    box(950, 490, 480, 330, "3. Re-rank", [
        "global + caption + bm25", "+ P(scene given image)", "+ BINDING per garment:",
        "  region(class & its color)", "  or caption pair (color,garment)",
        "weighted sum (configurable)"], "#fdf6ee")
    box(1490, 490, 280, 330, "4. Results", ["top-k images", "per-feature scores", "garment boxes", "'why it matched'"], "#f3efe7")
    arrow(410, 655, 465, 655)
    arrow(890, 655, 945, 655)
    arrow(1430, 655, 1485, 655)
    im.save(out)


def strip_png(qid: str, pools: dict, grades: dict, out: Path, k: int = 5) -> None:
    g = dict(zip(pools[qid]["ids"], grades[qid]))
    T, pad, lab = 230, 10, 230
    rows = [("Vanilla CLIP", pools[qid]["runs"]["baseline"][:k]), ("Ours", pools[qid]["runs"]["hybrid"][:k])]
    im = Image.new("RGB", (lab + k * (T + pad), len(rows) * (T + pad) + 50), "white")
    d = ImageDraw.Draw(im)
    d.text((10, 10), f"{qid}: {pools[qid]['query']}", fill="black", font=font(24))
    col = {2: "#15803d", 1: "#ca8a04", 0: "#b91c1c"}
    for r, (name, ids) in enumerate(rows):
        y = 50 + r * (T + pad)
        d.text((10, y + T // 2 - 14), name, fill="#c2410c" if name == "Ours" else "#555", font=font(26))
        for i, iid in enumerate(ids):
            t = Image.open(IMAGES / f"{iid}.jpg").convert("RGB")
            t.thumbnail((T, T))
            x = lab + i * (T + pad)
            im.paste(t, (x + (T - t.width) // 2, y + (T - t.height) // 2))
            gr = g.get(iid, 0)
            d.rectangle([x, y, x + T - 1, y + T - 1], outline=col[gr], width=6)
            d.rectangle([x, y, x + 34, y + 30], fill=col[gr])
            d.text((x + 10, y + 2), str(gr), fill="white", font=font(24))
    im.save(out, quality=85)


# ------------------------------------------------------------------ pdf
def build(github: str, demo: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    pools = json.loads((EVAL / "pools.json").read_text())
    grades = yaml.safe_load(open(EVAL / "grades.yaml"))
    res = json.loads((EVAL / "results.json").read_text())
    architecture_png(FIG / "architecture.png")
    for q in ("T1", "T3", "T5"):
        strip_png(q, pools, grades, FIG / f"strip_{q}.jpg")

    ss = getSampleStyleSheet()
    H1 = ParagraphStyle("H1", parent=ss["Heading1"], textColor=ACCENT, fontSize=17, spaceAfter=6)
    H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=12.5, spaceBefore=8, spaceAfter=4)
    B = ParagraphStyle("B", parent=ss["BodyText"], fontSize=9.6, leading=13.4, spaceAfter=5)
    S = ParagraphStyle("S", parent=B, fontSize=8.4, leading=11)
    P = lambda t, s=B: Paragraph(t, s)  # noqa: E731

    def table(data, widths, header=True):
        t = Table([[P(str(c), S) for c in row] for row in data], colWidths=widths, repeatRows=1)
        st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d8d0c2")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
              ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
        if header:
            st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f3efe7")))
        t.setStyle(TableStyle(st))
        return t

    m = res["mean"]
    s = []
    s += [P("Multimodal Fashion &amp; Context Retrieval", ParagraphStyle("T", parent=H1, fontSize=22)),
          P("Glance ML Internship Assignment — region-aware, compositional text-to-image fashion search"),
          P(f"<b>Code:</b> {github} &nbsp;&nbsp; <b>Live demo:</b> {demo}"),
          Spacer(1, 4),
          table([["", "Vanilla CLIP", "FashionSigLIP only", "Ours"],
                 ["P@5 (full matches)", f"{m['baseline']['P@5']:.2f}", f"{m['fashion']['P@5']:.2f}", f"<b>{m['hybrid']['P@5']:.2f}</b>"],
                 ["nDCG@10 (graded)", f"{m['baseline']['nDCG@10']:.2f}", f"{m['fashion']['nDCG@10']:.2f}", f"<b>{m['hybrid']['nDCG@10']:.2f}</b>"]],
                [5 * cm, 3.5 * cm, 4 * cm, 3 * cm]),
          P("Mean over the 5 assignment queries; relevance judged by looking at the images (not the captions the system uses).", S)]

    # 1. approaches
    s += [P("1. Approaches and trade-offs", H1)]
    s += [table([
        ["Approach", "Good at", "Weak at", "When to use"],
        ["Vanilla CLIP zero-shot (one vector per image)", "Fast, open-vocabulary, trivial to scale (ANN)", "Bag-of-concepts: colors not bound to garments; weak fine-grained fashion", "Baseline; generic image search"],
        ["Domain-tuned encoder (FashionCLIP, Marqo-FashionSigLIP)", "Fine-grained garment/style semantics", "Still one vector: same compositionality failure; weaker on scenes", "Fashion catalogs, product search"],
        ["Caption-then-retrieve (VLM captions + BM25/dense text)", "Explicit relations ('white shirt, black tie'), actions, scenes", "Caption errors/omissions; costly indexing", "When text-side reasoning matters"],
        ["Structured attributes (detector + color + scene tags)", "Precise, filterable, explainable", "Closed vocabulary, detector misses, brittle alone", "Faceted search, hard filters"],
        ["Fine-tune with hard negatives (swapped attributes)", "Fixes compositionality inside the embedding", "Needs labeled pairs + GPU training", "With product data and budget"],
        ["Cross-encoder / VLM re-ranker (BLIP-2 ITM, VLM judge)", "Highest precision per pair", "Slow: O(k) model calls per query", "Re-rank top-20..50 only"],
        ["Multi-vector / region-level retrieval (late interaction)", "Matches parts of images to parts of queries", "Storage x regions; more complex scoring", "Compositional queries at scale"],
    ], [4.2 * cm, 4.1 * cm, 4.3 * cm, 4.1 * cm])]
    s += [P("<b>Choice:</b> a hybrid of a domain-tuned encoder (recall, vibe), caption retrieval (relations, scenes) and "
            "region-level structured attributes (binding), combined with rank fusion and an explainable re-ranker. "
            "It needs no training labels (zero-shot), runs on a laptop, and each part covers another's failure mode.")]

    # 2. chosen approach
    s += [PageBreak(), P("2. Chosen architecture", H1), RLImage(str(FIG / "architecture.png"), width=17 * cm, height=8.1 * cm)]
    s += [P("<b>Indexer (Part A).</b> Each image gets (i) a Marqo-FashionSigLIP vector for overall look; (ii) garment regions from "
            "YOLOS-Fashionpedia (garment parts such as sleeves are dropped), each with a crop vector and a color distribution: "
            "LAB k-means pixels (skin removed) named by CIEDE2000 distance, fused 50/50 with a zero-shot 'a {color} {garment}' "
            "score so shadows and patterns do not fool it; (iii) zero-shot scene probabilities from CLIP with prompt ensembles; "
            "(iv) a Florence-2 caption stored as a bge vector, as BM25 sparse vector (Qdrant IDF modifier — exact BM25 computed in "
            "the DB), and as bound (color, garment) pairs. Extraction and loading are separate stages, so the DB can be rebuilt "
            "without re-running models.")]
    s += [P("<b>Retriever (Part B).</b> A free OpenRouter LLM (with a deterministic rule-parser fallback built on the same "
            "vocabulary) converts the query to structure: garments with their own colors, scene, style, plus a visual expansion. "
            "Three channels recall 200 candidates each (FashionSigLIP text-to-image, caption dense, BM25), merged by reciprocal "
            "rank fusion. The re-ranker scores global similarity, caption similarity, BM25, P(scene|image) and <b>binding</b>: for "
            "every requested garment, the best evidence that <i>this garment has this color</i> — a detected region of a compatible "
            "class whose crop matches 'a red tie' and whose own color distribution has red, or a caption phrase pairing them.")]
    s += [P("<b>How the five query types are handled.</b> <i>Attribute</i> (yellow raincoat): binding on region color + caption "
            "pair; 'raincoat' vs 'jacket' separated by crop-text similarity. <i>Contextual</i> (business attire in office): scene "
            "probability + inferred formal garments. <i>Complex semantic</i> (blue shirt on park bench): caption channels capture "
            "'sitting on a bench'; scene=park; binding for blue shirt. <i>Style inference</i> (casual weekend city walk): the parser "
            "infers t-shirt / jeans / sneakers (down-weighted as inferred) + scene=street. <i>Compositional</i> (red tie + white shirt): "
            "colors are bound per garment, so a white tie on a red shirt scores low although CLIP sees the same bag of concepts.")]

    # 3. results
    s += [PageBreak(), P("3. Evaluation", H1)]
    s += [P("Protocol: pooled the top-10 of vanilla CLIP, FashionSigLIP-only and ours (plus top-5 of 4 ablations) for each "
            "assignment query; every pooled image was graded 0/1/2 from contact sheets of the <b>images</b>. P@k counts full "
            "matches; nDCG uses graded gains. Weights were hand-set and <b>not tuned on these queries</b>.")]
    names = {"baseline": "Vanilla CLIP ViT-B/32", "fashion": "FashionSigLIP only", "hybrid": "Ours (hybrid)",
             "-binding": "Ours − binding", "-text": "Ours − caption/BM25", "-scene": "Ours − scene", "rules": "Ours, rule parser (no LLM)"}
    rows = [["System", "P@5", "Rel@5", "nDCG@5", "nDCG@10"]]
    for k, v in names.items():
        mm = m[k]
        rows.append([v, f"{mm['P@5']:.2f}", f"{mm['Rel@5']:.2f}", f"{mm['nDCG@5']:.2f}", f"{mm['nDCG@10']:.2f}" if "nDCG@10" in mm else "–"])
    s += [table(rows, [6 * cm, 2.4 * cm, 2.4 * cm, 2.4 * cm, 2.4 * cm])]
    pq = res["per_query"]
    rows = [["Query", "CLIP", "FashionSigLIP", "Ours"]] + [
        [f"{q} ({pq[q]['type']}): {pq[q]['query']}", *(f"{pq[q][s_]['nDCG@10']:.2f}" for s_ in ("baseline", "fashion", "hybrid"))]
        for q in pq]
    s += [Spacer(1, 6), P("nDCG@10 per query", H2), table(rows, [10 * cm, 2.2 * cm, 2.6 * cm, 2 * cm])]
    s += [P("<b>Reading the numbers.</b> Ours improves every query (mean nDCG@10 0.47 → 0.80; P@5 0.12 → 0.48). Each ablation "
            "lowers nDCG@5 (binding 0.77→0.69, scene →0.68, captions →0.74), so every component contributes. The rule parser "
            "slightly beat the LLM parser here (0.80 vs 0.77): on short, explicit queries deterministic parsing is enough, while "
            "the LLM adds value for vague, style-type phrasing and long-tail wording.")]
    s += [P("Top-5, vanilla CLIP vs ours (frame = grade: green 2, amber 1, red 0)", H2)]
    for q in ("T5", "T1", "T3"):
        s += [RLImage(str(FIG / f"strip_{q}.jpg"), width=17 * cm, height=17 * cm * 530 / 1380), Spacer(1, 3)]

    # limitations
    s += [P("Limitations (honest)", H2)]
    s += [P("• The garment detector is trained on Fashionpedia's fashion photos and misses or mislabels garments in everyday "
            "Flickr scenes (≈10% of images have no region); binding therefore falls back on caption pairs. "
            "• Captions can hallucinate or omit details; region evidence and caption evidence are combined with max, not trusted alone. "
            "• Multi-person images: attributes of different people can combine (no person-level grouping yet). "
            "• Small judged pool (5 queries, one annotator) — numbers indicate direction, not a benchmark. "
            "• The 600-image subset contains few perfect matches for some queries (e.g. blue shirt + sitting on a park bench), "
            "which caps achievable P@5.")]

    # 4. codebase + scale + zero-shot
    s += [PageBreak(), P("4. Codebase", H1)]
    s += [P(f"GitHub: {github}. Modules: <font face='Courier'>data/</font> (Kaggle sourcing, stratified curation), "
            "<font face='Courier'>indexer/</font> (Part A: detect, color, scene, caption, extract, load), "
            "<font face='Courier'>retriever/</font> (Part B: query, scoring, search, CLI), <font face='Courier'>common/</font> "
            "(shared vocabulary and binding parser used on both sides, encoders, Qdrant schema, BM25), "
            "<font face='Courier'>api/</font> (FastAPI), <font face='Courier'>web/</font> (Next.js), "
            "<font face='Courier'>eval/</font>. All model names, paths, thresholds and weights live in "
            "<font face='Courier'>configs/default.yaml</font> — logic is separated from data.")]
    s += [P("Scalability to 1M images", H2),
          P("Query cost is independent of corpus size beyond ANN: three HNSW/inverted-index lookups (sub-linear), then re-ranking "
            "a fixed 200 candidates with a payload-filtered fetch of their regions (indexed on image_id). For 1M images: Qdrant "
            "server/cluster with HNSW and scalar or binary quantization (768-d float → ~0.2–1 KB/vector), sparse BM25 stays an "
            "inverted index in Qdrant, regions (~3/image) in their own collection. Indexing is embarrassingly parallel and "
            "resumable; on one GPU the extractor runs roughly 10–20 img/s, i.e. a day for 1M images, or hours across workers. "
            "Captioning is the dominant cost and can be restricted to new / popular images.")]
    s += [P("Zero-shot capability", H2),
          P("Nothing is trained on our labels. Image–text matching uses open-vocabulary encoders, so unseen phrasing still lands "
            "near the right images; the LLM maps long-tail words (e.g. 'slicker', 'poncho') onto the garment taxonomy and the "
            "crop–text similarity keeps the fine-grained surface word. The closed-set part is the detector (46 classes); "
            "unknown garments still work via global and caption channels.")]

    # 5. future
    s += [P("5. Future work", H1)]
    s += [P("a) Locations (cities, places) and weather", H2),
          P("• Extend the scene head from 6 to Places365-style categories, and add landmark / city recognition (retrieval against "
            "a landmark gallery) stored as payload. • Use EXIF GPS + timestamp when available: reverse-geocode to city, and look up "
            "historical weather for that place and time. • Zero-shot weather/lighting attributes from the image (rain, snow, sunny, "
            "night, wet ground) with prompt ensembles; umbrellas and raincoats as supporting evidence. • Extend the query schema "
            "with city, place_type and weather fields so they become payload filters or soft re-rank features, and expand "
            "implications ('rainy day' → raincoat, umbrella, boots).")]
    s += [P("b) Improving precision", H2),
          P("• Cross-encoder re-rank of the top-20 with a VLM (BLIP-2 ITM or a small VLM judge answering 'is the tie red?'). "
            "• Fine-tune FashionSigLIP with hard negatives that swap attributes (red tie/white shirt vs white tie/red shirt). "
            "• Replace boxes with segmentation masks (SAM) for cleaner garment colors; an open-vocabulary detector "
            "(Grounding DINO / OWLv2) fine-tuned on in-the-wild photos. • Person-level grouping so attributes are bound per "
            "person in multi-person images. • Learn re-rank weights (learning-to-rank) from a larger judged set or click logs, "
            "and calibrate per query type. • Grow evaluation to 100+ queries with multiple annotators.")]

    doc = SimpleDocTemplate(str(DOCS / "report.pdf"), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=1.6 * cm, bottomMargin=1.6 * cm, title="Multimodal Fashion & Context Retrieval")
    doc.build(s)
    print(f"-> {DOCS / 'report.pdf'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--github", default="(link added after publishing)")
    ap.add_argument("--demo", default="(link added after deployment)")
    a = ap.parse_args()
    build(a.github, a.demo)
