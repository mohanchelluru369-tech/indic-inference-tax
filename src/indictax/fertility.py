"""Experiment 01: the tokenizer tax, measured on parallel content.

For each tokenizer and variant, over the items where both the variant and the
English baseline exist:

    tax              total tokens(variant) / total tokens(en)   same content, so this
                     is the multiplier on prefill compute, KV-cache memory and API cost
    context_share    1 / tax   how much of an English context window you effectively get
    tokens_per_word  classic "fertility"
    tokens_per_grapheme
    fragment_rate    share of tokens that are lone bytes / partial UTF-8
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

from .corpus import BASELINE, Item, variants_in
from .textstats import grapheme_count, word_count
from .tok import TokenizerAdapter


@dataclass
class FertilityRow:
    tokenizer: str
    vocab_size: int
    variant: str
    items: int
    tokens: int
    baseline_tokens: int
    tax: float
    context_share: float
    tokens_per_word: float
    tokens_per_grapheme: float
    fragment_rate: float


def measure(tokenizer: TokenizerAdapter, items: list[Item]) -> list[FertilityRow]:
    rows: list[FertilityRow] = []
    encoded: dict[tuple[str, str], list[int]] = {}
    for it in items:
        for v, text in it.variants.items():
            encoded[(it.id, v)] = tokenizer.encode(text)

    for v in variants_in(items):
        paired = [it for it in items if v in it.variants]
        toks = sum(len(encoded[(it.id, v)]) for it in paired)
        base = sum(len(encoded[(it.id, BASELINE)]) for it in paired)
        words = sum(word_count(it.variants[v]) for it in paired)
        graphs = sum(grapheme_count(it.variants[v]) for it in paired)
        frags = sum(
            1 for it in paired for tid in encoded[(it.id, v)] if tokenizer.is_fragment(tid)
        )
        tax = toks / base if base else float("nan")
        rows.append(
            FertilityRow(
                tokenizer=tokenizer.name,
                vocab_size=tokenizer.vocab_size,
                variant=v,
                items=len(paired),
                tokens=toks,
                baseline_tokens=base,
                tax=round(tax, 3),
                context_share=round(1 / tax, 3) if tax else float("nan"),
                tokens_per_word=round(toks / words, 3) if words else float("nan"),
                tokens_per_grapheme=round(toks / graphs, 3) if graphs else float("nan"),
                fragment_rate=round(frags / toks, 4) if toks else 0.0,
            )
        )
    return rows


def write_csv(rows: list[FertilityRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(asdict(rows[0]).keys()))
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


def to_markdown(rows: list[FertilityRow]) -> str:
    """One table: tokenizers down, variants across, cell = tax vs English."""
    variants = list(dict.fromkeys(r.variant for r in rows))
    by_tok: dict[str, dict[str, FertilityRow]] = {}
    for r in rows:
        by_tok.setdefault(r.tokenizer, {})[r.variant] = r

    out = ["## Tokenizer tax vs English (same content; 1.00 = parity)", ""]
    out.append("| tokenizer | vocab | " + " | ".join(variants) + " |")
    out.append("|---|---:|" + "---:|" * len(variants))
    for name, cells in by_tok.items():
        vocab = next(iter(cells.values())).vocab_size
        vals = [f"{cells[v].tax:.2f}" if v in cells else "-" for v in variants]
        out.append(f"| {name} | {vocab:,} | " + " | ".join(vals) + " |")

    out += ["", "## Fragment rate (tokens that are lone bytes / partial characters)", ""]
    out.append("| tokenizer | " + " | ".join(variants) + " |")
    out.append("|---|" + "---:|" * len(variants))
    for name, cells in by_tok.items():
        vals = [f"{cells[v].fragment_rate:.1%}" if v in cells else "-" for v in variants]
        out.append(f"| {name} | " + " | ".join(vals) + " |")
    return "\n".join(out) + "\n"
