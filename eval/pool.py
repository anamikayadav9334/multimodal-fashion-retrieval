"""Build judgment pools + numbered contact sheets for image-based relevance labeling.

    python eval/pool.py

Pools the top-10 of the main systems and the top-5 of each ablation, per query, then renders
eval/sheets/<qid>.jpg: numbered thumbnails so a human (or reviewer) labels relevance by
LOOKING AT THE IMAGES - not by reading the captions our own system uses.
Writes eval/pools.json  {qid: {"query":..., "ids":[...], "runs": {system: [ids]}}}
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageFont

from systems import ABLATIONS, MAIN, make_runner

HERE = Path(__file__).parent
IMAGES = HERE.parent / "data" / "images"
THUMB, COLS = 170, 7


def contact_sheet(ids: list[str], title: str, out: Path) -> None:
    rows = -(-len(ids) // COLS)
    sheet = Image.new("RGB", (COLS * THUMB, rows * THUMB + 28), "white")
    d = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        big = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 15)
    except OSError:
        font = big = ImageFont.load_default()
    d.text((6, 5), title, fill="black", font=big)
    for n, iid in enumerate(ids):
        im = Image.open(IMAGES / f"{iid}.jpg").convert("RGB")
        im.thumbnail((THUMB - 4, THUMB - 4))
        x, y = (n % COLS) * THUMB, (n // COLS) * THUMB + 28
        sheet.paste(im, (x + (THUMB - im.width) // 2, y + (THUMB - im.height) // 2))
        d.rectangle([x + 2, y + 2, x + 30, y + 22], fill="black")
        d.text((x + 5, y + 3), str(n), fill="yellow", font=font)
    sheet.save(out, quality=82)


def main() -> None:
    qs = yaml.safe_load(open(HERE / "queries.yaml"))
    queries = qs["test"] + qs["dev"]
    _, run = make_runner()
    (HERE / "sheets").mkdir(exist_ok=True)
    pools = {}
    for q in queries:
        runs = {s: run(s, q["text"], 10) for s in MAIN}
        runs |= {s: run(s, q["text"], 5) for s in ABLATIONS}
        ids = list(dict.fromkeys(i for s in MAIN + ABLATIONS for i in runs[s]))
        pools[q["id"]] = {"query": q["text"], "type": q["type"], "ids": ids, "runs": runs}
        contact_sheet(ids, f"{q['id']}: {q['text']}", HERE / "sheets" / f"{q['id']}.jpg")
        print(f"{q['id']}: {len(ids)} pooled images")
    (HERE / "pools.json").write_text(json.dumps(pools, indent=1))


if __name__ == "__main__":
    main()
