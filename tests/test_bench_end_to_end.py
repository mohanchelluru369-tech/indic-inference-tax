import json

import pytest

from indictax import bench, report
from indictax.bench import BenchConfig, score_mcq
from indictax.client import ChatClient


def _corpus(tmp_path, rows):
    p = tmp_path / "corpus.jsonl"
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    return str(p)


def _cfg(tmp_path, base_url, corpus, **kw):
    defaults = dict(name="t", corpus=corpus, base_url=base_url, model="fake", label="fake@test",
                    max_tokens=32, repeats=2, warmup=1, cooldown_s=0.0, power_sampler="none",
                    extra_body={"cache_prompt": False})
    return BenchConfig(**{**defaults, **kw})


def test_client_measures_a_stream(fake_server):
    _, url = fake_server
    c = ChatClient(url)
    assert c.ping() == (True, "fake-model")
    r = c.complete("fake", [{"role": "user", "content": "నమస్తే"}], max_tokens=32)
    c.close()
    assert r.ok and r.text == "ఇది ఒక పరీక్ష సమాధానం."
    assert r.prompt_tokens == len("నమస్తే".encode()) and r.completion_tokens == 5
    assert r.finish_reason == "stop"
    assert 0 < r.ttft_s < r.total_s
    assert r.decode_tok_s and r.decode_tok_s > 0
    assert len(r.inter_chunk_gaps()) == 4
    assert r.server_timings["predicted_per_second"] == 250.0


def test_client_reports_http_errors_as_data(fake_server):
    _, url = fake_server
    c = ChatClient(url)
    r = c.complete("fake", [{"role": "user", "content": "FAIL please"}], max_tokens=8)
    c.close()
    assert not r.ok and "HTTP 500" in r.error


def test_unreachable_endpoint_stops_the_run_with_instructions(tmp_path):
    corpus = _corpus(tmp_path, [{"id": "a", "variants": {"en": "hi"}}])
    with pytest.raises(SystemExit, match="serve_llamacpp"):
        bench.run(_cfg(tmp_path, "http://127.0.0.1:9/v1", corpus), out_root=tmp_path / "results")


def test_full_run_and_report(fake_server, tmp_path):
    srv, url = fake_server
    rows = [
        {"id": "q1", "domain": "d", "variants": {"en": "What is rice?", "te": "వరి అంటే ఏమిటి?", "hi": "चावल क्या है?"}},
        {"id": "q2", "domain": "d", "variants": {"en": "Plan a trip.", "te": "ఒక యాత్రను ప్లాన్ చేయండి."}},
    ]
    cfg = _cfg(tmp_path, url, _corpus(tmp_path, rows), system_prompt="Reply in the user's language.")
    run_dir = bench.run(cfg, out_root=tmp_path / "results")

    raw = report.load_rows(run_dir)
    assert len(raw) == (3 + 2) * 2  # every present (item, variant) x repeats; the warm-up is not recorded
    assert all(r["ok"] for r in raw)
    assert all(r["expected_script_share"] == 1.0 for r in raw)

    # request hygiene: caching off, deterministic decode, system prompt first
    sent = srv.requests[-1]
    assert sent["cache_prompt"] is False and sent["temperature"] == 0.0 and sent["stream"] is True
    assert sent["messages"][0]["role"] == "system"

    # interleaving: the shuffled order must not be grouped by variant
    order = [r["variant"] for r in raw]
    assert order != sorted(order)

    taxes = {(t["variant"], t["metric"]): t for t in report.tax(raw)}
    te_in = taxes[("te", "prompt_tokens")]
    assert te_in["items"] == 2 and te_in["tax_median"] > 2          # 3 bytes per Telugu code point
    assert te_in["ci_lo"] <= te_in["tax_median"] <= te_in["ci_hi"]
    assert taxes[("hi", "prompt_tokens")]["items"] == 1             # q2 has no Hindi: paired on q1 only
    assert ("te", "energy") not in taxes                            # no sampler -> no invented energy

    summary = report.write(run_dir).read_text(encoding="utf-8")
    assert "fake@test" in summary and "| te |" in summary
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["n_requests"] == 10 and "finished" in meta and meta["system"]["python"]


def test_truncation_and_scoring(fake_server, tmp_path):
    _, url = fake_server
    rows = [{"id": "m1", "gold": "B", "variants": {"en": "ANSWER_B passage", "te": "ANSWER_B గద్యం"}},
            {"id": "m2", "gold": "C", "variants": {"en": "ANSWER_B other", "te": "ANSWER_B ఇంకొకటి"}}]
    run_dir = bench.run(_cfg(tmp_path, url, _corpus(tmp_path, rows), max_tokens=1, repeats=1),
                        out_root=tmp_path / "results")
    raw = report.load_rows(run_dir)
    assert all(r["truncated"] for r in raw)  # one-token cap reached
    by_variant = {a["variant"]: a for a in report.absolute(raw)}
    assert by_variant["en"]["accuracy"] == 0.5 and by_variant["te"]["truncated_rate"] == 1.0


@pytest.mark.parametrize("text,gold,ok", [
    ("B", "B", True), ("The answer is B.", "b", True), ("**C**", "C", True),
    ("A) because ...", "A", True), ("Because", "B", False), ("", "A", False), ("D", "A", False),
])
def test_score_mcq(text, gold, ok):
    assert score_mcq(text, gold) is ok


def test_config_from_yaml():
    cfg = BenchConfig.from_yaml("configs/exp02_gemma3_4b.yaml")
    assert cfg.extra_body == {"cache_prompt": False} and cfg.temperature == 0.0 and cfg.repeats == 3


def test_in_stream_error_and_dropped_stream_are_failures_not_results(fake_server):
    _, url = fake_server
    c = ChatClient(url)
    err = c.complete("fake", [{"role": "user", "content": "STREAM_ERROR"}], max_tokens=8)
    dropped = c.complete("fake", [{"role": "user", "content": "DROP"}], max_tokens=8)
    c.close()
    assert not err.ok and "context size exceeded" in err.error
    assert not dropped.ok and "without finish_reason" in dropped.error and dropped.text == "partial "


@pytest.mark.parametrize("text,expected", [
    ("A farmer must rotate crops, so the answer is C.", "C"),   # the article "A" is not an answer
    ("Option A is wrong. The correct answer is B", "B"),
    ("b", "B"), ("(b)", "B"), ("B) 1945", "B"), ("Answer: C", "C"), ("The answer is **D**", "D"),
    ("उत्तर: B", "B"), ("सही उत्तर B है, A नहीं", "B"), ("సమాధానం: C", "C"),
    ("According to the passage, Vitamin C helps. D", "D"),
    ("I cannot determine the answer from passage A or B.", None),
    ("A farmer must rotate crops.", None), ("", None),
])
def test_parse_mcq(text, expected):
    assert bench.parse_mcq(text) == expected
