"""Uses a byte-level BPE trained on English only, built locally in the test.
That is a fair miniature of the real problem: a tokenizer that has never seen
Telugu has to spell it out byte by byte."""

from pathlib import Path

import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from indictax import fertility
from indictax.corpus import load_corpus
from indictax.tok import load_tokenizer

SEED = Path(__file__).parent.parent / "data/prompts/seed_v0.jsonl"


@pytest.fixture(scope="module")
def english_only_tokenizer(tmp_path_factory):
    items = load_corpus(SEED)
    english = [it.variants["en"] for it in items] * 20
    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=600, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                  show_progress=False)
    tk.train_from_iterator(english, trainer)
    path = tmp_path_factory.mktemp("tok") / "tokenizer.json"
    tk.save(str(path))
    return load_tokenizer(f"file:{path}")


def test_tax_and_fragments(english_only_tokenizer):
    rows = {r.variant: r for r in fertility.measure(english_only_tokenizer, load_corpus(SEED))}

    assert rows["en"].tax == 1.0
    assert rows["en"].fragment_rate < 0.01

    # Telugu is 3 bytes per code point and nothing merges: several x English, almost all fragments.
    assert rows["te"].tax > 3
    assert rows["te"].fragment_rate > 0.9
    assert rows["te"].context_share == pytest.approx(1 / rows["te"].tax, abs=0.01)

    # Romanized Telugu is Latin script: far cheaper than native script under this tokenizer.
    assert rows["te_rom"].tax < rows["te"].tax / 2
    assert rows["te_rom"].fragment_rate < 0.01


def test_pairing_uses_only_items_that_have_the_variant(english_only_tokenizer):
    items = load_corpus(SEED)
    rows = {r.variant: r for r in fertility.measure(english_only_tokenizer, items)}
    n_short = sum(1 for it in items if it.kind == "short")
    assert rows["te"].items == len(items)
    assert rows["te_cm"].items == n_short
    # baseline tokens for te_cm exclude the long context items, so they must be smaller
    assert rows["te_cm"].baseline_tokens < rows["te"].baseline_tokens


def test_outputs(english_only_tokenizer, tmp_path):
    rows = fertility.measure(english_only_tokenizer, load_corpus(SEED))
    fertility.write_csv(rows, tmp_path / "f.csv")
    assert (tmp_path / "f.csv").read_text(encoding="utf-8").count("\n") == len(rows) + 1
    md = fertility.to_markdown(rows)
    assert "| tokenizer | vocab | en | hi | te |" in md
