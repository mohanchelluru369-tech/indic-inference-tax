"""Turn a run's raw.jsonl into the numbers the paper reports.

Two views:
  absolute   per variant: what a user of that language experiences
  tax        per variant, paired with English on the same item: the multiplier
             on tokens, time-to-first-token, total time and energy

The tax is a median of per-item ratios with a 95% bootstrap interval that
resamples *items*, not requests: repeats of one prompt are not independent
observations, and pretending they are would make the intervals look tighter
than the evidence supports.
"""

from __future__ import annotations

import csv
import json
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np

from .corpus import BASELINE, VARIANTS

TAX_METRICS = [
    ("prompt_tokens", "input tokens"),
    ("completion_tokens", "output tokens"),
    ("ttft_s", "time to first token"),
    ("total_s", "time per answer"),
    ("energy", "energy per answer"),
]


def _variant_order(v: str) -> tuple[int, str]:
    order = list(VARIANTS)
    return (order.index(v) if v in order else len(order), v)


def load_rows(run_dir: str | Path) -> list[dict]:
    rows = []
    with open(Path(run_dir) / "raw.jsonl", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _energy(row: dict) -> float | None:
    return row.get("net_energy_j") if row.get("net_energy_j") is not None else row.get("energy_j")


def _value(row: dict, metric: str) -> float | None:
    return _energy(row) if metric == "energy" else row.get(metric)


def _vals(rows: list[dict], metric: str) -> list[float]:
    return [v for r in rows if (v := _value(r, metric)) is not None]


def _med(xs: list[float]) -> float | None:
    return float(np.median(xs)) if xs else None


def _mean(xs: list[float]) -> float | None:
    return float(np.mean(xs)) if xs else None


def absolute(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    failed: dict[tuple[str, str], int] = defaultdict(int)
    for r in rows:
        key = (r["label"], r["variant"])
        if r["ok"]:
            groups[key].append(r)
        else:
            failed[key] += 1
    out = []
    for (label, variant), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], _variant_order(kv[0][1]))):
        scored = [r for r in rs if r.get("correct") is not None]
        n_correct = sum(1 for r in scored if r["correct"])
        energies = _vals(rs, "energy")
        # joules per correct answer: numerator and denominator over the SAME rows
        # (scored and with an energy reading), or the figure is meaningless.
        both = [r for r in scored if _energy(r) is not None]
        both_correct = sum(1 for r in both if r["correct"])
        decode = _vals(rs, "server_decode_tok_s") or _vals(rs, "decode_tok_s")
        out.append({
            "label": label,
            "variant": variant,
            "n": len(rs),
            "failed": failed[(label, variant)],
            "prompt_tokens_mean": _mean(_vals(rs, "prompt_tokens")),
            "completion_tokens_mean": _mean(_vals(rs, "completion_tokens")),
            "ttft_s_p50": _med(_vals(rs, "ttft_s")),
            "total_s_p50": _med(_vals(rs, "total_s")),
            "total_s_p95": float(np.percentile(_vals(rs, "total_s"), 95)) if rs else None,
            "decode_tok_s_p50": _med(decode),
            "decode_source": "server" if _vals(rs, "server_decode_tok_s") else "client",
            "graphemes_per_s_p50": _med([r["out_graphemes"] / r["total_s"] for r in rs if r["total_s"]]),
            "truncated_rate": _mean([1.0 if r.get("truncated") else 0.0 for r in rs]),
            "expected_script_share": _mean(_vals(rs, "expected_script_share")),
            "energy_j_p50": _med(energies),
            "accuracy": (n_correct / len(scored)) if scored else None,
            "unparsed_rate": _mean([1.0 if r.get("parsed_answer") is None else 0.0 for r in scored]) if scored else None,
            "energy_j_per_correct": (sum(_energy(r) for r in both) / both_correct) if both_correct else None,
        })
    return out


def _per_item_means(rows: list[dict], label: str, variant: str, metric: str) -> dict[str, float]:
    acc: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r["ok"] and r["label"] == label and r["variant"] == variant:
            v = _value(r, metric)
            if v is not None:
                acc[r["item_id"]].append(v)
    return {k: float(np.mean(v)) for k, v in acc.items()}


def tax(rows: list[dict], n_boot: int = 2000, seed: int = 0) -> list[dict]:
    labels = list(dict.fromkeys(r["label"] for r in rows))
    variants = sorted({r["variant"] for r in rows} - {BASELINE}, key=_variant_order)
    out = []
    for label in labels:
        for variant in variants:
            for metric, title in TAX_METRICS:
                base = _per_item_means(rows, label, BASELINE, metric)
                var = _per_item_means(rows, label, variant, metric)
                ratios = np.array([var[i] / base[i] for i in var if i in base and base[i] > 0])
                if ratios.size == 0:
                    continue
                if ratios.size > 1:
                    # own stream per cell: an interval must not depend on which other
                    # variants or models happen to be in the same run
                    rng = np.random.default_rng([seed, zlib.crc32(f"{label}|{variant}|{metric}".encode())])
                    idx = rng.integers(0, ratios.size, size=(n_boot, ratios.size))
                    boots = np.median(ratios[idx], axis=1)
                    lo, hi = np.percentile(boots, [2.5, 97.5])
                else:
                    lo = hi = ratios[0]
                out.append({
                    "label": label, "variant": variant, "metric": metric, "title": title,
                    "items": int(ratios.size), "tax_median": float(np.median(ratios)),
                    "ci_lo": float(lo), "ci_hi": float(hi),
                })
    return out


def _f(x, spec=".2f", dash="-") -> str:
    return dash if x is None else format(x, spec)


def to_markdown(meta: dict, abs_rows: list[dict], tax_rows: list[dict]) -> str:
    sysm = meta.get("system", {})
    lines = [
        f"# Run {meta.get('run_id', '')}",
        "",
        f"- machine: {sysm.get('chip') or sysm.get('machine')} / {sysm.get('ram_gb')} GB / {sysm.get('platform')}",
        f"- gpu: {sysm.get('gpu') or 'n/a'}",
        f"- engine: llama-server={sysm.get('tool:llama-server')}  ollama={sysm.get('tool:ollama')}",
        f"- power: {meta.get('power_sampler')} ({meta.get('power_channel')}), "
        f"idle baseline {_f(meta.get('idle_baseline_w'))} W; energy below is net of idle when a baseline exists",
        f"- requests: {meta.get('n_requests')} over {meta.get('n_items')} items",
    ]
    if sysm.get("low_power_mode"):
        lines.append("- **WARNING: Low Power Mode was ON. Discard this run.**")
    if sysm.get("system") == "Darwin" and sysm.get("on_ac_power") is False:
        lines.append("- **WARNING: running on battery. Re-run on AC power.**")

    lines += ["", "## The tax: multiplier vs English on the same content",
              "", "Median of per-item ratios, 95% bootstrap interval over items. 1.00 = parity.", ""]
    for label in dict.fromkeys(t["label"] for t in tax_rows):
        lines += [f"### {label}", "", "| variant | " + " | ".join(t for _, t in TAX_METRICS) + " |",
                  "|---|" + "---:|" * len(TAX_METRICS)]
        mine = [t for t in tax_rows if t["label"] == label]
        for variant in dict.fromkeys(t["variant"] for t in mine):
            cells = []
            for metric, _ in TAX_METRICS:
                hit = [t for t in mine if t["variant"] == variant and t["metric"] == metric]
                cells.append(
                    f"{hit[0]['tax_median']:.2f}x ({hit[0]['ci_lo']:.2f}-{hit[0]['ci_hi']:.2f})" if hit else "-"
                )
            lines.append(f"| {variant} | " + " | ".join(cells) + " |")
        lines.append("")

    lines += ["## Absolute numbers per variant", "",
              "| model | variant | n | failed | in tok | out tok | TTFT p50 s | total p50 s | total p95 s | decode tok/s "
              "| graphemes/s | cut off | in expected script | J/answer | accuracy | J/correct |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for a in abs_rows:
        lines.append(
            f"| {a['label']} | {a['variant']} | {a['n']} | {a['failed']} | {_f(a['prompt_tokens_mean'], '.0f')} "
            f"| {_f(a['completion_tokens_mean'], '.0f')} | {_f(a['ttft_s_p50'])} | {_f(a['total_s_p50'])} "
            f"| {_f(a['total_s_p95'])} | {_f(a['decode_tok_s_p50'], '.1f')} | {_f(a['graphemes_per_s_p50'], '.1f')} "
            f"| {_f(a['truncated_rate'], '.0%')} | {_f(a['expected_script_share'], '.0%')} "
            f"| {_f(a['energy_j_p50'], '.1f')} | {_f(a['accuracy'], '.0%')} | {_f(a['energy_j_per_correct'], '.1f')} |"
        )
    noisy = [a for a in abs_rows if (a.get("unparsed_rate") or 0) > 0.05]
    if noisy:
        lines += ["", "**WARNING: more than 5% of answers could not be parsed as A-D for: "
                  + ", ".join(f"{a['variant']} ({a['unparsed_rate']:.0%})" for a in noisy)
                  + ". Accuracy for these rows is a lower bound; read the raw outputs.**"]
    if any(a["failed"] for a in abs_rows):
        lines += ["", "**Failed requests are excluded from every statistic above, including accuracy. "
                  "If failures differ by variant, the comparison is compromised: fix the cause and re-run.**"]
    lines += ["", "Reading guide: decode tok/s should be roughly equal across variants (a token is a token). "
              "The tax shows up as *more tokens for the same content*, which inflates time and energy per answer, "
              "and as answers cut off at the token cap. 'In expected script' below ~90% means the model is "
              "drifting into English or the wrong script.", ""]
    return "\n".join(lines)


def write(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    rows = load_rows(run_dir)
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    abs_rows, tax_rows = absolute(rows), tax(rows)
    md = to_markdown(meta, abs_rows, tax_rows)
    (run_dir / "summary.md").write_text(md, encoding="utf-8")
    for name, table in (("absolute.csv", abs_rows), ("tax.csv", tax_rows)):
        if table:
            with open(run_dir / name, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(table[0].keys()))
                w.writeheader()
                w.writerows(table)
    return run_dir / "summary.md"
