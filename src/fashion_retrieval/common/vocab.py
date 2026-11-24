"""Canonical fashion / scene / color vocabulary.

Single source of truth shared by:
  * dataset curation (keyword-based stratified sampling of Flickr30k captions)
  * the indexer (zero-shot scene labels, color naming)
  * the rule-based query parser fallback in the retriever

Keeping this in one place means "office" means the same thing at index time and at query time.
"""

from __future__ import annotations

import re

# --- Environments (assignment axis 1) -------------------------------------------------------
SCENE_KEYWORDS: dict[str, list[str]] = {
    "office": ["office", "desk", "cubicle", "conference room", "meeting", "workplace",
               "boardroom", "laptop", "computer"],
    "street": ["street", "sidewalk", "city", "crosswalk", "downtown", "urban", "alley",
               "intersection", "storefront"],
    "park": ["park", "bench", "grass", "garden", "lawn", "trees", "picnic", "fountain"],
    "home": ["living room", "kitchen", "couch", "sofa", "bedroom", "home", "dining room",
             "apartment"],
}

# --- Clothing style (assignment axis 2) ----------------------------------------------------
STYLE_KEYWORDS: dict[str, list[str]] = {
    "formal": ["suit", "blazer", "tie", "necktie", "button-down", "button down", "dress shirt",
               "tuxedo", "sport coat", "business attire", "bow tie"],
    "casual": ["t-shirt", "tshirt", "hoodie", "sweatshirt", "jeans", "shorts", "tank top",
               "tee", "sneakers", "flip-flops"],
    "outerwear": ["jacket", "coat", "raincoat", "parka", "windbreaker", "poncho",
                  "trench coat", "rain jacket"],
}

# --- Colors (assignment axis 3) ------------------------------------------------------------
# Base color names used for both caption parsing and pixel-level color naming (LAB centroids
# are defined in the indexer). Modifiers like "bright"/"dark"/"light" are parsed separately.
COLORS: list[str] = [
    "red", "orange", "yellow", "green", "blue", "navy", "purple", "pink", "brown", "beige",
    "black", "white", "gray", "maroon", "teal",
]
COLOR_SYNONYMS: dict[str, str] = {
    "grey": "gray", "crimson": "red", "scarlet": "red", "burgundy": "maroon",
    "violet": "purple", "lavender": "purple", "tan": "beige", "khaki": "beige",
    "cream": "beige", "olive": "green", "turquoise": "teal", "cyan": "teal",
    "gold": "yellow", "golden": "yellow", "silver": "gray", "magenta": "pink",
}

# Crowd / multi-subject cues: we prefer images with one clear subject.
CROWD_CUES = ["crowd", "group of", "many people", "team", "parade", "audience", "several people"]
SINGLE_SUBJECT_PREFIX = re.compile(
    r"^(a|an|one)\s+(young\s+|old\s+|older\s+|little\s+|smiling\s+)?"
    r"(man|woman|person|girl|boy|lady|guy|gentleman|businessman|businesswoman)\b",
    re.IGNORECASE,
)


def keyword_pattern(words: list[str]) -> re.Pattern[str]:
    """Word-boundary regex so 'tie' doesn't match 'tied'."""
    alts = sorted((re.escape(w) for w in words), key=len, reverse=True)
    return re.compile(r"\b(" + "|".join(alts) + r")s?\b", re.IGNORECASE)


SCENE_PATTERNS = {k: keyword_pattern(v) for k, v in SCENE_KEYWORDS.items()}
STYLE_PATTERNS = {k: keyword_pattern(v) for k, v in STYLE_KEYWORDS.items()}
COLOR_PATTERN = keyword_pattern(COLORS + list(COLOR_SYNONYMS))
CROWD_PATTERN = keyword_pattern(CROWD_CUES)


def normalize_color(word: str) -> str:
    w = word.lower()
    return COLOR_SYNONYMS.get(w, w)
