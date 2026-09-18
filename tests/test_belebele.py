from indictax.belebele import INSTRUCTION, build_items
from indictax.corpus import load_corpus
import json


def _row(link, qn, passage, answer="2"):
    return {"link": link, "question_number": qn, "flores_passage": passage, "question": "Q?",
            "mc_answer1": "w", "mc_answer2": "x", "mc_answer3": "y", "mc_answer4": "z",
            "correct_answer_num": answer}


def test_only_questions_present_in_every_language_are_kept(tmp_path):
    rows = {
        "en": [_row("u1", 1, "English one"), _row("u1", 2, "English two", "4"), _row("u2", 1, "English only")],
        "te": [_row("u1", 2, "తెలుగు రెండు", "4"), _row("u1", 1, "తెలుగు ఒకటి")],
    }
    items = build_items(rows, n=10)
    assert len(items) == 2
    golds = {it["meta"]["belebele_key"]: it["gold"] for it in items}
    assert golds == {"u1#1": "B", "u1#2": "D"}
    for it in items:
        assert set(it["variants"]) == {"en", "te"}
        assert it["variants"]["te"].endswith(INSTRUCTION) and "B) x" in it["variants"]["te"]

    p = tmp_path / "b.jsonl"
    p.write_text("\n".join(json.dumps(i, ensure_ascii=False) for i in items) + "\n", encoding="utf-8")
    assert load_corpus(p)[0].gold in "ABCD"
