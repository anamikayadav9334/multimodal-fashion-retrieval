"""Extract bound (color, garment) pairs from free text.

Used on BOTH sides so they speak the same language:
  * indexer:   Florence-2 captions -> "white shirt, black tie" -> [(white, shirt), (black, tie)]
  * retriever: rule-based query parser fallback

Binding rule: colors attach to the NEAREST following garment within a short window, so
"a red tie and a white shirt" yields (red, tie), (white, shirt) - never (red, shirt).
"black and white striped shirt" yields (black, shirt), (white, shirt).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import vocab

_TOKEN = re.compile(r"[a-z]+(?:-[a-z]+)*")
_MODIFIERS = {"bright", "dark", "light", "pale", "deep", "neon", "pastel", "vivid"}
_WINDOW = 4  # max tokens between a color and its garment


@dataclass
class GarmentMention:
    garment: str                        # canonical garment (vocab.GARMENTS key)
    surface: str                        # word as written, e.g. "raincoat", "blazer"
    colors: list[str] = field(default_factory=list)
    modifiers: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"garment": self.garment, "surface": self.surface,
                "colors": self.colors, "modifiers": self.modifiers}


def _garment_spans(text: str) -> list[tuple[int, int, str]]:
    return [(m.start(), m.end(), m.group(1).lower()) for m in vocab.GARMENT_PATTERN.finditer(text)]


def extract_garment_mentions(text: str) -> list[GarmentMention]:
    low = text.lower()
    spans = _garment_spans(low)
    if not spans:
        return []
    tokens = [(m.start(), m.end(), m.group(0)) for m in _TOKEN.finditer(low)]

    mentions: list[GarmentMention] = []
    prev_end = 0
    for start, end, surface in spans:
        # tokens strictly between the previous garment and this one, closest first
        between = [t for t in tokens if prev_end <= t[0] and t[1] <= start]
        colors, mods = [], []
        for _, _, tok in reversed(between[-_WINDOW * 2:]):
            if vocab.COLOR_PATTERN.fullmatch(tok):
                colors.append(vocab.normalize_color(tok))
            elif tok in _MODIFIERS:
                mods.append(tok)
            elif tok in {"and", "or", "with", "in", "a", "an", "the", "striped", "plaid", "checkered"} | _MODIFIERS:
                continue
            elif colors:  # a non-connector word after we found colors ends the color run
                break
        mentions.append(GarmentMention(
            garment=vocab.GARMENT_SYNONYM_TO_CANONICAL.get(surface, surface.rstrip("s")),
            surface=surface,
            colors=list(dict.fromkeys(reversed(colors))),
            modifiers=list(dict.fromkeys(mods)),
        ))
        prev_end = end
    return mentions


def color_garment_pairs(text: str) -> list[str]:
    """Flat 'color:garment' tokens - handy as a Qdrant keyword payload for filtering."""
    return sorted({f"{c}:{m.garment}" for m in extract_garment_mentions(text) for c in m.colors})
