"""Re-ranking features. Pure functions over candidate payloads/vectors - no I/O, easy to test.

The key feature is `binding`: for every garment in the query we look for evidence that THIS
garment has THIS color, from two independent sources:
  1. detected regions  - a region whose detector class fits the garment, scored by
                         (crop <-> "a red tie" similarity percentile) x (that region's P(red)) x conf
  2. caption mentions  - Florence-2 caption contains the bound pair, e.g. ("red", "tie")
A garment's score is the max over its evidence; the query score averages garments (inferred
garments weighted lower). "white tie + red shirt" therefore scores low for "red tie + white
shirt" even though a global embedding sees the same bag of {red, white, tie, shirt}.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..common import vocab
from .query import GarmentQuery, QuerySpec


@dataclass
class Evidence:
    garment: str
    colors: list[str]
    score: float
    source: str                      # "region" | "caption" | "none"
    detail: dict = field(default_factory=dict)


def minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(x)), float(np.max(x))
    return np.zeros_like(x) if hi - lo < 1e-9 else (x - lo) / (hi - lo)


def garment_text(g: GarmentQuery) -> str:
    return "a photo of a " + " ".join([*g.modifiers, *g.colors, g.surface])


def _related(a: str, b: str) -> bool:
    """Garments that share a detector class are 'related' (raincoat ~ jacket ~ coat)."""
    da, db = set(vocab.GARMENTS.get(a, {}).get("det", [])), set(vocab.GARMENTS.get(b, {}).get("det", []))
    return bool(da & db)


def color_match(colors_wanted: list[str], colors_have: dict[str, float] | list[str]) -> float:
    if not colors_wanted:
        return 1.0
    if isinstance(colors_have, list):
        return 1.0 if set(colors_wanted) & set(colors_have) else 0.0
    # P(any wanted color); navy/blue & maroon/red are near-misses, not zero
    near = {"navy": "blue", "blue": "navy", "maroon": "red", "red": "maroon", "teal": "green", "beige": "white"}
    p = sum(colors_have.get(c, 0.0) + 0.5 * colors_have.get(near.get(c, ""), 0.0) for c in colors_wanted)
    return float(min(1.0, p / max(len(colors_wanted), 1) * 1.6))  # 1.6: a 0.6-prob color counts ~fully


def region_evidence(g: GarmentQuery, regions: list[dict], crop_pct: dict[int, float]) -> Evidence:
    ok_classes = set(vocab.GARMENTS.get(g.garment, {}).get("det", []))
    best = Evidence(g.garment, g.colors, 0.0, "none")
    for r in regions:
        if r["garment"] not in ok_classes:
            continue
        s = (0.4 + 0.6 * crop_pct.get(r["_rid"], 0.0)) * color_match(g.colors, r["colors"]) * min(1.0, 0.5 + r["score"])
        if s > best.score:
            best = Evidence(g.garment, g.colors, s, "region",
                            {"class": r["garment"], "box": r["box"], "colors": r["colors"], "det": r["score"]})
    return best


def caption_evidence(g: GarmentQuery, mentions: list[dict], related_credit: float) -> Evidence:
    best = Evidence(g.garment, g.colors, 0.0, "none")
    for m in mentions:
        if m["garment"] == g.garment:
            base = 1.0
        elif _related(g.garment, m["garment"]):
            base = related_credit
        else:
            continue
        if g.colors and not m["colors"]:
            base *= 0.5                       # garment named, color unknown
        else:
            base *= color_match(g.colors, m["colors"])
        if base > best.score:
            best = Evidence(g.garment, g.colors, base, "caption", {"mention": m})
    return best


def binding_score(spec: QuerySpec, payload: dict, regions: list[dict], crop_pcts: list[dict[int, float]],
                  inferred_w: float, related_credit: float) -> tuple[float, list[Evidence]]:
    """crop_pcts[i]: region id -> percentile of crop similarity to garment i's text, within the pool."""
    if not spec.garments:
        return 0.0, []
    evid, num, den = [], 0.0, 0.0
    for g, crop_pct in zip(spec.garments, crop_pcts):
        e_r = region_evidence(g, regions, crop_pct)
        e_c = caption_evidence(g, payload.get("caption_mentions", []), related_credit)
        e = e_r if e_r.score >= e_c.score else e_c
        if e is e_c and e_r.score > 0:  # caption won, but keep the region box for visualization
            e.detail = {**e.detail, "box": e_r.detail.get("box"), "class": e_r.detail.get("class")}
        w = inferred_w if g.inferred else 1.0
        num, den = num + w * e.score, den + w
        evid.append(e)
    return num / den, evid
