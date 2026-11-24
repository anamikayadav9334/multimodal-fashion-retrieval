"""Search from the command line.

    python -m fashion_retrieval.retriever.cli "a red tie and a white shirt in a formal setting" -k 5
    python -m fashion_retrieval.retriever.cli "..." --mode baseline     # vanilla CLIP
    python -m fashion_retrieval.retriever.cli "..." --no-llm --json
"""

from __future__ import annotations

import argparse
import json

from ..common.config import load_config
from .search import MODES, Retriever


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("query")
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--mode", choices=MODES, default="hybrid")
    ap.add_argument("--no-llm", action="store_true", help="rule-based query parser only")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    out = Retriever(load_config()).search(args.query, k=args.k, mode=args.mode, use_llm=not args.no_llm)
    if args.json:
        print(json.dumps(out, indent=2, default=str))
        return
    if "spec" in out:
        s = out["spec"]
        print(f"parser={s['parser']} scene={s['scene']} style={s['style']}")
        print("garments:", [(g["garment"], g["colors"], "inferred" if g["inferred"] else "") for g in s["garments"]])
        print("expanded:", s["expanded"])
    print(f"\n{out['mode']} - {out['timing']}")
    for i, r in enumerate(out["results"], 1):
        print(f"{i:2d}. {r['id']:<16} {r['score']:.3f}  [{r['scene_top']}]  {r['caption'][:95]}")
        if "features" in r:
            f = r["features"]
            ev = ", ".join(f"{e['garment']}:{'/'.join(e['colors']) or '-'}<-{e['source']}({e['score']:.2f})" for e in r["evidence"])
            print(f"      g={f['global']:.2f} c={f['caption']:.2f} bm={f['bm25']:.2f} sc={f['scene']:.2f} "
                  f"bind={f['binding']:.2f}  {ev}")


if __name__ == "__main__":
    main()
