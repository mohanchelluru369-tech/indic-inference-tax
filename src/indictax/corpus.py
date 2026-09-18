"""Parallel prompt corpus.

One JSONL line per item. Every item carries the *same content* in several
variants, so every comparison in this project is paired by item id:

    {"id": "agri-01", "domain": "agriculture", "kind": "short",
     "variants": {"en": "...", "hi": "...", "te": "...",
                  "hi_rom": "...", "te_rom": "...", "hi_cm": "...", "te_cm": "..."},
     "gold": "B",                # optional, for scored tasks
     "review": "pending"}        # native-speaker review status
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# variant -> (language, how it is written, script the *answer* is expected in)
VARIANTS: dict[str, dict[str, str]] = {
    "en": {"language": "English", "form": "native", "script": "latin"},
    "hi": {"language": "Hindi", "form": "native", "script": "devanagari"},
    "te": {"language": "Telugu", "form": "native", "script": "telugu"},
    "hi_rom": {"language": "Hindi", "form": "romanized", "script": "latin"},
    "te_rom": {"language": "Telugu", "form": "romanized", "script": "latin"},
    "hi_cm": {"language": "Hindi-English", "form": "code-mixed", "script": "latin"},
    "te_cm": {"language": "Telugu-English", "form": "code-mixed", "script": "latin"},
}

BASELINE = "en"


@dataclass
class Item:
    id: str
    domain: str
    kind: str
    variants: dict[str, str]
    gold: str | None = None
    review: str = "pending"
    meta: dict = field(default_factory=dict)


def expected_script(variant: str) -> str | None:
    info = VARIANTS.get(variant)
    return info["script"] if info else None


def load_corpus(path: str | Path) -> list[Item]:
    items: list[Item] = []
    seen: set[str] = set()
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            row = json.loads(line)
            if row["id"] in seen:
                raise ValueError(f"{path}:{n}: duplicate id {row['id']!r}")
            seen.add(row["id"])
            if BASELINE not in row["variants"]:
                raise ValueError(f"{path}:{n}: item {row['id']!r} has no '{BASELINE}' variant")
            items.append(
                Item(
                    id=row["id"],
                    domain=row.get("domain", ""),
                    kind=row.get("kind", "short"),
                    variants=row["variants"],
                    gold=row.get("gold"),
                    review=row.get("review", "pending"),
                    meta=row.get("meta", {}),
                )
            )
    if not items:
        raise ValueError(f"{path}: corpus is empty")
    return items


def variants_in(items: list[Item]) -> list[str]:
    """All variants present, baseline first, then in VARIANTS order, then extras."""
    present = {v for it in items for v in it.variants}
    ordered = [v for v in VARIANTS if v in present]
    return ordered + sorted(present - set(ordered))
