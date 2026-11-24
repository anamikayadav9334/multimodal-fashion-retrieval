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


# --- Garment taxonomy ------------------------------------------------------------------------
# canonical query garment -> detector classes that may contain it + surface forms in text.
# The detector is coarse (e.g. "coat"); the fine-grained distinction (raincoat vs trench coat)
# is made by comparing the garment CROP embedding against the query text - see retriever.
GARMENTS: dict[str, dict[str, list[str]]] = {
    "shirt":    {"det": ["shirt"], "syn": ["shirt", "blouse", "button-down", "button down", "dress shirt", "polo", "collared shirt"]},
    "t-shirt":  {"det": ["top"], "syn": ["t-shirt", "tshirt", "tee", "tank top", "top", "jersey"]},
    "hoodie":   {"det": ["top", "sweater", "jacket", "hood"], "syn": ["hoodie", "hooded sweatshirt", "sweatshirt"]},
    "sweater":  {"det": ["sweater", "cardigan"], "syn": ["sweater", "jumper", "pullover", "cardigan", "knit"]},
    "blazer":   {"det": ["jacket", "coat"], "syn": ["blazer", "sport coat", "suit jacket"]},
    "jacket":   {"det": ["jacket", "coat"], "syn": ["jacket", "windbreaker", "bomber", "denim jacket", "leather jacket"]},
    "coat":     {"det": ["coat", "jacket"], "syn": ["coat", "overcoat", "trench coat", "parka", "peacoat"]},
    "raincoat": {"det": ["coat", "jacket", "hood"], "syn": ["raincoat", "rain jacket", "poncho", "slicker"]},
    "suit":     {"det": ["jacket", "pants", "vest"], "syn": ["suit", "tuxedo"]},
    "vest":     {"det": ["vest"], "syn": ["vest", "waistcoat"]},
    "pants":    {"det": ["pants"], "syn": ["pants", "trousers", "jeans", "slacks", "chinos", "leggings"]},
    "shorts":   {"det": ["shorts"], "syn": ["shorts"]},
    "skirt":    {"det": ["skirt"], "syn": ["skirt"]},
    "dress":    {"det": ["dress"], "syn": ["dress", "gown"]},
    "jumpsuit": {"det": ["jumpsuit"], "syn": ["jumpsuit", "overalls"]},
    "tie":      {"det": ["tie"], "syn": ["tie", "necktie", "bow tie"]},
    "hat":      {"det": ["hat", "headband"], "syn": ["hat", "cap", "beanie", "beret"]},
    "scarf":    {"det": ["scarf"], "syn": ["scarf"]},
    "glasses":  {"det": ["glasses"], "syn": ["glasses", "sunglasses"]},
    "bag":      {"det": ["bag"], "syn": ["bag", "backpack", "purse", "handbag", "wallet", "tote"]},
    "shoes":    {"det": ["shoe"], "syn": ["shoe", "shoes", "sneakers", "boots", "heels", "sandals", "loafers"]},
    "belt":     {"det": ["belt"], "syn": ["belt"]},
    "watch":    {"det": ["watch"], "syn": ["watch"]},
    "gloves":   {"det": ["glove"], "syn": ["glove", "gloves", "mittens"]},
    "umbrella": {"det": ["umbrella"], "syn": ["umbrella"]},
}

# Detector label (YOLOS-Fashionpedia) -> short class. Garment *parts* (sleeve, collar, pocket,
# zipper, ...) are deliberately absent: they are not things people search for by color.
DETECTOR_CLASSES: dict[str, str] = {
    "shirt, blouse": "shirt", "top, t-shirt, sweatshirt": "top", "sweater": "sweater",
    "cardigan": "cardigan", "jacket": "jacket", "vest": "vest", "pants": "pants",
    "shorts": "shorts", "skirt": "skirt", "coat": "coat", "dress": "dress",
    "jumpsuit": "jumpsuit", "cape": "cape", "glasses": "glasses", "hat": "hat",
    "headband, head covering, hair accessory": "headband", "tie": "tie", "glove": "glove",
    "watch": "watch", "belt": "belt", "tights, stockings": "tights", "sock": "sock",
    "shoe": "shoe", "bag, wallet": "bag", "scarf": "scarf", "umbrella": "umbrella", "hood": "hood",
}

GARMENT_SYNONYM_TO_CANONICAL: dict[str, str] = {
    s: g for g, spec in GARMENTS.items() for s in spec["syn"]
}
GARMENT_PATTERN = keyword_pattern(list(GARMENT_SYNONYM_TO_CANONICAL))

# --- Color palette for pixel-level naming (sRGB anchors; several shades per name) -----------
COLOR_ANCHORS_RGB: dict[str, list[tuple[int, int, int]]] = {
    "red":    [(200, 30, 40), (230, 50, 50), (170, 20, 30)],
    "maroon": [(110, 20, 35), (128, 0, 32)],
    "orange": [(240, 130, 30), (220, 100, 40)],
    "yellow": [(245, 215, 40), (250, 235, 110), (230, 190, 30)],
    "green":  [(40, 140, 60), (110, 140, 60), (30, 90, 40), (140, 200, 120)],
    "teal":   [(0, 128, 128), (40, 170, 170)],
    "blue":   [(40, 90, 200), (100, 150, 220), (70, 130, 180), (150, 190, 230)],
    "navy":   [(20, 30, 80), (35, 45, 100)],
    "purple": [(110, 50, 150), (170, 130, 200), (80, 40, 100)],
    "pink":   [(240, 130, 170), (250, 190, 210), (220, 80, 140)],
    "brown":  [(110, 70, 40), (150, 100, 60), (80, 50, 30)],
    "beige":  [(220, 200, 160), (200, 180, 140), (235, 225, 200)],
    "black":  [(20, 20, 20), (45, 45, 50)],
    "white":  [(245, 245, 245), (225, 225, 220)],
    "gray":   [(128, 128, 128), (90, 90, 95), (180, 180, 180)],
}

# --- Zero-shot scene prompts (prompt ensembles are more robust than single labels) ----------
SCENE_PROMPTS: dict[str, list[str]] = {
    "office": ["a photo taken inside a modern office", "a person at work in an office",
               "an office interior with desks and computers", "a corporate meeting room"],
    "street": ["a photo taken on a city street", "a person walking on an urban sidewalk",
               "a busy downtown street with buildings", "a street scene in a city"],
    "park":   ["a photo taken in a park", "a person sitting on a park bench",
               "trees and grass in a public park", "an outdoor garden or lawn"],
    "home":   ["a photo taken inside a home", "a person in a living room",
               "a cozy home interior with a sofa", "a person in their kitchen or bedroom"],
    "studio": ["a fashion photo shoot in a studio", "a model on a plain studio background",
               "a runway fashion show", "a professional fashion editorial photo"],
    "other":  ["a photo taken outdoors in nature", "a photo at a beach or in the mountains",
               "a photo at a sports event", "a photo at a concert or party"],
}
