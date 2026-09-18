"""Build a scored, parallel corpus from Belebele (facebook/belebele on Hugging Face).

Belebele is the same 900 reading-comprehension questions in 122 language
variants, which is exactly the paired design this project needs: it turns
"energy per answer" into "energy per *correct* answer".

Needs the optional dependency:  pip install 'indictax[hf]'

NOTE: written against the published dataset card (configs such as eng_Latn /
hin_Deva / tel_Telu / hin_Latn, split "test", fields flores_passage, question,
mc_answer1..4, correct_answer_num, link, question_number). It could not be
exercised against the live dataset from the sandbox it was written in; the
row-to-item transform is unit-tested, the download call is not.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

# our variant name -> Belebele config
DEFAULT_LANGS = {"en": "eng_Latn", "hi": "hin_Deva", "te": "tel_Telu", "hi_rom": "hin_Latn"}

LETTERS = "ABCD"
# The instruction line is deliberately identical (English) in every variant: a
# constant token overhead, so differences between variants come from the content.
INSTRUCTION = "Answer with only the letter A, B, C or D."


def format_prompt(row: dict) -> str:
    options = "\n".join(f"{LETTERS[i]}) {row[f'mc_answer{i + 1}']}" for i in range(4))
    return f"{row['flores_passage']}\n\n{row['question']}\n{options}\n\n{INSTRUCTION}"


def key_of(row: dict) -> str:
    return f"{row['link']}#{row['question_number']}"


def build_items(rows_by_variant: dict[str, list[dict]], n: int, seed: int = 0) -> list[dict]:
    """Keep only questions present in every variant, sample n, emit corpus rows."""
    indexed = {v: {key_of(r): r for r in rows} for v, rows in rows_by_variant.items()}
    shared = sorted(set.intersection(*(set(ix) for ix in indexed.values())))
    if not shared:
        raise ValueError("no question is present in every requested language")
    random.Random(seed).shuffle(shared)
    items = []
    for i, k in enumerate(shared[:n]):
        base = indexed["en"][k] if "en" in indexed else next(iter(indexed.values()))[k]
        items.append({
            "id": f"bele-{i:04d}",
            "domain": "reading-comprehension",
            "kind": "context",
            "variants": {v: format_prompt(ix[k]) for v, ix in indexed.items()},
            "gold": LETTERS[int(base["correct_answer_num"]) - 1],
            "review": "source:belebele",
            "meta": {"belebele_key": k},
        })
    return items


def make(out_path: str | Path, n: int = 100, langs: dict[str, str] | None = None, seed: int = 0) -> Path:
    try:
        from datasets import load_dataset
    except ImportError as e:  # pragma: no cover
        raise SystemExit("Needs the 'datasets' package:  uv pip install -e '.[hf]'") from e

    langs = langs or DEFAULT_LANGS
    if "en" not in langs:
        raise SystemExit("the 'en' baseline variant is required")
    rows_by_variant = {}
    for variant, config in langs.items():
        print(f"loading facebook/belebele [{config}] ...")
        rows_by_variant[variant] = list(load_dataset("facebook/belebele", config, split="test"))

    items = build_items(rows_by_variant, n=n, seed=seed)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"wrote {len(items)} items x {len(langs)} variants -> {out_path}")
    return out_path
