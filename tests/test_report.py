from indictax import report


def _r(item, variant, **kw):
    base = dict(label="m", item_id=item, variant=variant, ok=True, prompt_tokens=10, completion_tokens=10,
                ttft_s=0.1, total_s=1.0, decode_tok_s=10.0, out_graphemes=10, truncated=False,
                expected_script_share=1.0, energy_j=None, net_energy_j=None, correct=None, parsed_answer=None)
    return {**base, **kw}


def test_joules_per_correct_uses_the_same_rows_top_and_bottom():
    # 4 correct answers at 100 J each, but only 2 of them have an energy reading.
    rows = [_r(f"i{k}", "te", correct=True, parsed_answer="A", net_energy_j=100.0 if k < 2 else None)
            for k in range(4)]
    te = report.absolute(rows)[0]
    assert te["energy_j_per_correct"] == 100.0       # not 200 J / 4 = 50
    assert te["accuracy"] == 1.0 and te["unparsed_rate"] == 0.0


def test_wrong_answers_still_cost_energy():
    rows = [_r("i0", "en", correct=True, parsed_answer="A", net_energy_j=10.0),
            _r("i1", "en", correct=False, parsed_answer="B", net_energy_j=10.0),
            _r("i2", "en", correct=False, parsed_answer=None, net_energy_j=10.0)]
    en = report.absolute(rows)[0]
    assert en["energy_j_per_correct"] == 30.0 and en["unparsed_rate"] == 1 / 3


def test_failures_are_counted_and_called_out():
    rows = [_r("i0", "en"), _r("i0", "te"), _r("i1", "te", ok=False)]
    table = {a["variant"]: a for a in report.absolute(rows)}
    assert table["te"]["failed"] == 1 and table["te"]["n"] == 1
    md = report.to_markdown({"system": {}}, report.absolute(rows), report.tax(rows))
    assert "| failed |" in md and "Failed requests are excluded" in md


def test_interval_for_one_variant_does_not_depend_on_the_others():
    def rows_for(variants):
        out = []
        for k in range(12):
            out.append(_r(f"i{k}", "en", prompt_tokens=10))
            for v in variants:
                out.append(_r(f"i{k}", v, prompt_tokens=20 + 3 * k))
        return out

    def te_ci(rows):
        hit = [t for t in report.tax(rows) if t["variant"] == "te" and t["metric"] == "prompt_tokens"][0]
        return hit["ci_lo"], hit["ci_hi"]

    assert te_ci(rows_for(["te"])) == te_ci(rows_for(["hi", "te"]))


def test_server_decode_rate_preferred_over_client_estimate():
    rows = [_r("i0", "te", decode_tok_s=35.5, server_decode_tok_s=33.3)]
    te = report.absolute(rows)[0]
    assert te["decode_tok_s_p50"] == 33.3 and te["decode_source"] == "server"
