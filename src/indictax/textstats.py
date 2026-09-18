"""Language-neutral text statistics.

Tokens are model-specific, bytes are biased against Indic scripts (3 bytes per
code point in UTF-8), so content is counted in units a reader would recognise:
whitespace words and grapheme clusters (one Telugu akshara = one unit).
"""

from __future__ import annotations

import regex

_GRAPHEME = regex.compile(r"\X")

# Unicode blocks for the scripts this project cares about.
SCRIPT_RANGES: dict[str, list[tuple[int, int]]] = {
    "latin": [(0x41, 0x5A), (0x61, 0x7A), (0xC0, 0x24F)],
    "devanagari": [(0x900, 0x97F), (0xA8E0, 0xA8FF)],
    "bengali": [(0x980, 0x9FF)],
    "gurmukhi": [(0xA00, 0xA7F)],
    "gujarati": [(0xA80, 0xAFF)],
    "odia": [(0xB00, 0xB7F)],
    "tamil": [(0xB80, 0xBFF)],
    "telugu": [(0xC00, 0xC7F)],
    "kannada": [(0xC80, 0xCFF)],
    "malayalam": [(0xD00, 0xD7F)],
}


def word_count(text: str) -> int:
    return len(text.split())


def grapheme_count(text: str) -> int:
    """Grapheme clusters, ignoring whitespace."""
    return sum(1 for g in _GRAPHEME.findall(text) if not g.isspace())


def utf8_bytes(text: str) -> int:
    return len(text.encode("utf-8"))


def _script_of(ch: str) -> str | None:
    cp = ord(ch)
    for name, ranges in SCRIPT_RANGES.items():
        for lo, hi in ranges:
            if lo <= cp <= hi:
                return name
    return None


def script_profile(text: str) -> dict[str, float]:
    """Share of script-bearing characters per script. Digits, punctuation and
    whitespace are ignored. Returns {} when the text has no script characters."""
    counts: dict[str, int] = {}
    for ch in text:
        s = _script_of(ch)
        if s:
            counts[s] = counts.get(s, 0) + 1
    total = sum(counts.values())
    if not total:
        return {}
    return {k: v / total for k, v in sorted(counts.items(), key=lambda kv: -kv[1])}


def script_share(text: str, script: str) -> float:
    return script_profile(text).get(script, 0.0)
