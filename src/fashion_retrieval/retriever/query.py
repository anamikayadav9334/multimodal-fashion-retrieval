"""Query understanding: natural language -> structured QuerySpec.

Two interchangeable parsers with the same output:
  * LLMParser  - OpenRouter free models (JSON output). Best at *style inference*
                 ("casual weekend outfit" -> t-shirt, jeans, sneakers) and vague phrasing.
  * RuleParser - offline fallback built on the SAME vocabulary + binding logic as the indexer.

The LLM never sees images and never ranks anything: it only rewrites the query into the
attribute schema, so its failures degrade gracefully to the rule parser.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..common import vocab
from ..common.attributes import extract_garment_mentions
from ..common.config import resolve


@dataclass
class GarmentQuery:
    garment: str                 # canonical (vocab.GARMENTS key)
    surface: str                 # as phrased, used for crop-text similarity ("raincoat")
    colors: list[str] = field(default_factory=list)
    modifiers: list[str] = field(default_factory=list)
    inferred: bool = False       # guessed from style rather than stated


@dataclass
class QuerySpec:
    text: str
    garments: list[GarmentQuery] = field(default_factory=list)
    scene: str | None = None     # office | street | park | home | None
    style: str | None = None     # formal | casual | outerwear | sporty | None
    expanded: str = ""           # visual description used for dense recall
    parser: str = "rules"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "QuerySpec":
        return cls(**{**d, "garments": [GarmentQuery(**g) for g in d.get("garments", [])]})


# --------------------------------------------------------------------------- rule-based parser
_STYLE_CUES = {
    "formal": re.compile(r"\b(formal|professional|business|corporate|office wear|elegant|smart)\b", re.I),
    "casual": re.compile(r"\b(casual|weekend|relaxed|laid[- ]back|everyday|streetwear|comfortable)\b", re.I),
    "outerwear": re.compile(r"\b(rain|rainy|cold|winter|chilly|layered)\b", re.I),
    "sporty": re.compile(r"\b(sport|sporty|athletic|gym|workout|running)\b", re.I),
}
_STYLE_GARMENTS = {  # used only when the query names no garment at all
    "formal": [("blazer", "blazer"), ("shirt", "dress shirt"), ("tie", "tie")],
    "casual": [("t-shirt", "t-shirt"), ("pants", "jeans"), ("shoes", "sneakers")],
    "outerwear": [("coat", "coat"), ("jacket", "jacket")],
    "sporty": [("t-shirt", "athletic shirt"), ("shorts", "shorts"), ("shoes", "running shoes")],
}


class RuleParser:
    def parse(self, text: str) -> QuerySpec:
        garments = [GarmentQuery(m.garment, m.surface, m.colors, m.modifiers)
                    for m in extract_garment_mentions(text)]
        scene_hits = {k: len(p.findall(text)) for k, p in vocab.SCENE_PATTERNS.items()}
        scene = max(scene_hits, key=scene_hits.get) if any(scene_hits.values()) else None
        style = next((s for s, p in _STYLE_CUES.items() if p.search(text)), None)
        if style is None:
            style = next((s for s, p in vocab.STYLE_PATTERNS.items() if p.search(text)), None)
        if not garments and style in _STYLE_GARMENTS:
            garments = [GarmentQuery(g, s, inferred=True) for g, s in _STYLE_GARMENTS[style]]
        return QuerySpec(text=text, garments=garments, scene=scene, style=style,
                         expanded=_describe(text, garments, scene, style), parser="rules")


def _describe(text: str, garments: list[GarmentQuery], scene: str | None, style: str | None) -> str:
    parts = [" ".join([*g.modifiers, *g.colors, g.surface]) for g in garments]
    desc = "a person wearing " + ", ".join(parts) if parts else text
    if style and style not in ("outerwear",):
        desc = f"{desc}, {style} style"
    if scene:
        desc += f", {SCENE_PHRASE[scene]}"
    return desc


SCENE_PHRASE = {"office": "inside an office", "street": "on a city street",
                "park": "in a park", "home": "at home"}


# --------------------------------------------------------------------------- LLM parser
_SYSTEM = """You convert fashion image-search queries into JSON. Output ONLY a JSON object:
{
 "garments": [{"garment": <one of GARMENTS>, "surface": <the word used or implied, e.g. "raincoat">,
               "colors": [<COLORS>], "modifiers": [<e.g. "bright","dark","light">], "inferred": <bool>}],
 "scene": <one of SCENES or null>,
 "style": <"formal"|"casual"|"outerwear"|"sporty"|null>,
 "expanded": <one vivid visual sentence describing what a matching photo shows>
}
Rules:
- Bind each color to the garment it describes ("red tie and white shirt" -> tie:red, shirt:white).
- Only use colors that are stated; never invent colors.
- If the query names NO garment but implies a style/occasion, add 2-3 typical garments with "inferred": true.
- scene: only if a place is stated or clearly implied ("city walk" -> street, "modern office" -> office,
  "park bench" -> park). "formal setting" alone is NOT a scene.
GARMENTS: %s
COLORS: %s
SCENES: office, street, park, home"""


class LLMParser:
    def __init__(self, cfg: dict):
        c = cfg["retriever"]["llm"]
        self.base_url, self.models, self.timeout = c["base_url"], c["models"], c["timeout"]
        self.budget = c.get("total_budget", 12)
        self.key = os.environ.get(c["api_key_env"], "")
        self.system = _SYSTEM % (", ".join(vocab.GARMENTS), ", ".join(vocab.COLORS))

    @property
    def available(self) -> bool:
        return bool(self.key)

    def parse(self, text: str) -> QuerySpec:
        """Try each free model in turn; any failure (504, 429, empty/invalid JSON) -> next model."""
        import requests

        import time

        errors, t0 = [], time.time()
        for model in self.models:
            if time.time() - t0 > self.budget:
                errors.append("time budget exhausted")
                break
            try:
                r = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.key}", "X-Title": "Multimodal Fashion Retrieval"},
                    json={"model": model,
                          "messages": [{"role": "system", "content": self.system},
                                       {"role": "user", "content": text}],
                          "temperature": 0, "max_tokens": 1200,
                          # reasoning models otherwise burn the budget thinking and return no content
                          "reasoning": {"effort": "low", "exclude": True},
                          "response_format": {"type": "json_object"}},
                    timeout=self.timeout,
                )
                r.raise_for_status()
                content = (r.json()["choices"][0]["message"].get("content") or "")
                m = re.search(r"\{.*\}", content, re.S)
                if not m:
                    raise ValueError("empty or non-JSON content")
                return self._validate(text, json.loads(m.group(0)), model)
            except Exception as e:
                errors.append(f"{model.split('/')[-1]}: {type(e).__name__}")
        raise RuntimeError("all free models failed -> " + "; ".join(errors))

    def _validate(self, text: str, d: dict, model: str) -> QuerySpec:
        """Never trust LLM output blindly: snap everything onto the known vocabulary."""
        garments = []
        for g in d.get("garments") or []:
            name = str(g.get("garment", "")).lower()
            name = name if name in vocab.GARMENTS else vocab.GARMENT_SYNONYM_TO_CANONICAL.get(name)
            if not name:
                continue
            colors = [vocab.normalize_color(c) for c in g.get("colors") or []]
            garments.append(GarmentQuery(
                garment=name, surface=str(g.get("surface") or name).lower(),
                colors=[c for c in colors if c in vocab.COLORS],
                modifiers=[str(m).lower() for m in g.get("modifiers") or []][:3],
                inferred=bool(g.get("inferred", False)),
            ))
        scene = d.get("scene") if d.get("scene") in vocab.SCENE_KEYWORDS else None
        style = d.get("style") if d.get("style") in _STYLE_GARMENTS else None
        expanded = str(d.get("expanded") or "").strip() or _describe(text, garments, scene, style)
        return QuerySpec(text=text, garments=garments, scene=scene, style=style,
                         expanded=expanded, parser=f"llm:{model}")


# --------------------------------------------------------------------------- facade with cache
class QueryParser:
    def __init__(self, cfg: dict):
        from dotenv import load_dotenv

        load_dotenv(resolve(".env"))
        self.rules = RuleParser()
        self.llm = LLMParser(cfg) if cfg["retriever"]["llm"]["enabled"] else None
        self.cache_path: Path = resolve(cfg["retriever"]["llm"]["cache"])
        self.cache: dict[str, dict] = json.loads(self.cache_path.read_text()) if self.cache_path.exists() else {}

    def parse(self, text: str, use_llm: bool = True) -> QuerySpec:
        text = " ".join(text.split())
        key = text.lower()
        if use_llm and self.llm and self.llm.available:
            if key in self.cache:
                return QuerySpec.from_dict(self.cache[key])
            try:
                spec = self.llm.parse(text)
                self.cache[key] = spec.to_dict()
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                self.cache_path.write_text(json.dumps(self.cache, indent=1))
                return spec
            except Exception as e:  # rate limit, timeout, bad JSON -> deterministic fallback
                print(f"[query] LLM parser failed ({type(e).__name__}: {str(e)[:120]}); using rules")
        return self.rules.parse(text)
