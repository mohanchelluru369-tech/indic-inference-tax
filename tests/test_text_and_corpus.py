from pathlib import Path

import pytest

from indictax.corpus import VARIANTS, expected_script, load_corpus, variants_in
from indictax.textstats import grapheme_count, script_profile, script_share, word_count

SEED = Path(__file__).parent.parent / "data/prompts/seed_v0.jsonl"


def test_grapheme_count_treats_an_akshara_as_one_unit():
    # "క్షి" is 4 code points (ka + virama + ssa + vowel sign) but one written unit.
    assert len("క్షి") == 4
    assert grapheme_count("క్షి") == 1
    assert grapheme_count("hello world") == 10  # whitespace ignored


def test_word_count():
    assert word_count("నా ఫోన్ బ్యాటరీ") == 3


def test_script_profile_ignores_digits_and_punctuation():
    prof = script_profile("1857 తిరుగుబాటు!")
    assert prof == {"telugu": 1.0}
    assert script_profile("123 ... !!") == {}


def test_script_share_mixed():
    share = script_share("hello నమస్తే", "telugu")
    assert 0.4 < share < 0.7


def test_seed_corpus_loads_and_is_complete():
    items = load_corpus(SEED)
    assert len(items) >= 16
    assert variants_in(items)[0] == "en"
    for it in items:
        expected = set(VARIANTS) if it.kind == "short" else {"en", "hi", "te"}
        assert set(it.variants) == expected, it.id


def test_seed_corpus_variants_are_in_the_script_they_claim():
    # Guards against a Devanagari line pasted into the Telugu slot, or a
    # "romanized" variant that still contains native script.
    for it in load_corpus(SEED):
        for v, text in it.variants.items():
            assert script_share(text, expected_script(v)) >= 0.85, (it.id, v)


def test_duplicate_ids_rejected(tmp_path):
    p = tmp_path / "c.jsonl"
    row = '{"id": "a", "variants": {"en": "x"}}\n'
    p.write_text(row + row, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_corpus(p)


def test_missing_baseline_rejected(tmp_path):
    p = tmp_path / "c.jsonl"
    p.write_text('{"id": "a", "variants": {"te": "x"}}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="no 'en'"):
        load_corpus(p)
