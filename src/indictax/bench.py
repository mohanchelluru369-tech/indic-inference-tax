"""Experiment 02: what the tokenizer tax turns into on real hardware.

Sends every (item, variant) of a parallel corpus to one serving endpoint,
single-stream (the on-device case), and records per request: prefill and
decode timing, token counts, output size in language-neutral units, energy,
temperature, whether the answer came back in the expected script, and whether
it hit the token cap.

Controls that matter (see docs/03_methodology.md):
  * request order is shuffled with a fixed seed so languages interleave and
    thermal drift is spread evenly instead of landing on whichever ran last
  * warm-up requests are sent and discarded
  * prompt/prefix caching must be OFF or repeated prompts get a free prefill
  * temperature 0 and a fixed seed
"""

from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from . import sysinfo
from .client import ChatClient, Completion
from .corpus import Item, expected_script, load_corpus, variants_in
from .energy import PowerSampler, make_sampler
from .textstats import grapheme_count, script_share, utf8_bytes, word_count

_STRIP = " \t\r\n()[]{}*_`\"'.:"
_LEADING = re.compile(r"^[\s(\[*]*([A-Da-d])\s*[).:\]]")
_CUE = re.compile(
    r"(?:answer|ans|उत्तर|जवाब|సమాధానం|జవాబు)\s*(?:is|=|:|-|है)?\s*[\s(\[*\"']*([A-Da-d])(?![A-Za-z])",
    re.IGNORECASE,
)
# a lone letter closing the output, but only after a sentence end or on its own line
_TRAILING = re.compile(r"(?:^|[.!?।\n])\s*[(\[*]*([A-D])[\s).\]*]*$")


@dataclass
class BenchConfig:
    name: str
    corpus: str
    base_url: str
    model: str
    label: str
    variants: list[str] | None = None
    system_prompt: str | None = None
    extra_body: dict | None = None
    api_key: str | None = None
    max_tokens: int = 384
    temperature: float = 0.0
    seed: int | None = 0
    repeats: int = 3
    warmup: int = 2
    cooldown_s: float = 1.0
    shuffle_seed: int = 7
    limit_items: int | None = None
    power_sampler: str = "auto"
    power_interval_ms: int = 250
    idle_baseline_s: float = 10.0

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BenchConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        ep, dec, pw = raw.get("endpoint", {}), raw.get("decode", {}), raw.get("power", {})
        return cls(
            name=raw["name"],
            corpus=raw["corpus"],
            base_url=ep["base_url"],
            model=ep["model"],
            label=ep.get("label", ep["model"]),
            extra_body=ep.get("extra_body"),
            api_key=ep.get("api_key"),
            variants=raw.get("variants"),
            system_prompt=raw.get("system_prompt"),
            max_tokens=dec.get("max_tokens", 384),
            temperature=dec.get("temperature", 0.0),
            seed=dec.get("seed", 0),
            repeats=raw.get("repeats", 3),
            warmup=raw.get("warmup", 2),
            cooldown_s=raw.get("cooldown_s", 1.0),
            shuffle_seed=raw.get("shuffle_seed", 7),
            limit_items=raw.get("limit_items"),
            power_sampler=pw.get("sampler", "auto"),
            power_interval_ms=pw.get("interval_ms", 250),
            idle_baseline_s=pw.get("idle_baseline_s", 10.0),
        )


def parse_mcq(text: str) -> str | None:
    """The letter the model chose, or None if that cannot be read with confidence.

    Deliberately conservative: grabbing the first capital A-D scores the English
    article in "A farmer must ..." as answer A, which would bias English accuracy
    and nothing else. Unparsed answers count as wrong and are reported separately."""
    t = text.strip()
    if not t:
        return None
    bare = t.strip(_STRIP)
    if len(bare) == 1 and bare.upper() in "ABCD":
        return bare.upper()
    for rx in (_LEADING, _CUE, _TRAILING):
        m = rx.search(t)
        if m:
            return m.group(1).upper()
    return None


def score_mcq(text: str, gold: str) -> bool:
    return parse_mcq(text) == gold.strip().upper()


def _pct(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def _messages(cfg: BenchConfig, prompt: str) -> list[dict]:
    msgs = [{"role": "system", "content": cfg.system_prompt}] if cfg.system_prompt else []
    return msgs + [{"role": "user", "content": prompt}]


def _row(cfg: BenchConfig, run_id: str, item: Item, variant: str, repeat: int,
         c: Completion, sampler: PowerSampler, idle_w: float | None) -> dict:
    gaps_ms = [g * 1000 for g in c.inter_chunk_gaps()]
    script = expected_script(variant)
    energy = sampler.energy_j(c.t_start, c.t_end) if c.ok else None
    net = None
    if energy is not None and idle_w is not None:
        net = max(0.0, energy - idle_w * c.total_s)
    timings = c.server_timings or {}
    return {
        "run_id": run_id,
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "label": cfg.label,
        "item_id": item.id,
        "domain": item.domain,
        "kind": item.kind,
        "variant": variant,
        "repeat": repeat,
        "ok": c.ok,
        "error": c.error,
        "prompt_tokens": c.prompt_tokens,
        "completion_tokens": c.completion_tokens,
        "ttft_s": c.ttft_s,
        "total_s": c.total_s,
        "decode_tok_s": c.decode_tok_s,
        # Gaps between *visible chunks*, not tokens: when a script is split into byte
        # fragments a chunk needs several decode steps, so text arrives in slower bursts.
        "chunk_gap_p50_ms": _pct(gaps_ms, 50),
        "chunk_gap_p95_ms": _pct(gaps_ms, 95),
        "n_chunks": len(c.chunk_times),
        "out_words": word_count(c.text),
        "out_graphemes": grapheme_count(c.text),
        "out_bytes": utf8_bytes(c.text),
        "reasoning_chars": c.reasoning_chars,
        "expected_script": script,
        "expected_script_share": script_share(c.text, script) if script and c.text else None,
        "finish_reason": c.finish_reason,
        "truncated": c.finish_reason == "length",
        "energy_j": energy,
        "net_energy_j": net,
        "mean_w": sampler.mean(c.t_start, c.t_end, sampler.primary) if c.ok else None,
        "gpu_temp_c": sampler.mean(c.t_start, c.t_end, "gpu_temp_c") if c.ok else None,
        "server_prefill_ms": timings.get("prompt_ms"),
        "server_prefill_tok_s": timings.get("prompt_per_second"),
        "server_decode_tok_s": timings.get("predicted_per_second"),
        "parsed_answer": parse_mcq(c.text) if (item.gold and c.ok) else None,
        "correct": score_mcq(c.text, item.gold) if (item.gold and c.ok) else None,
        "text": c.text,
    }


def run(cfg: BenchConfig, out_root: str | Path = "results") -> Path:
    items = load_corpus(cfg.corpus)
    if cfg.limit_items:
        items = items[: cfg.limit_items]
    variants = cfg.variants or variants_in(items)

    jobs = [
        (it, v, r)
        for it in items
        for v in variants
        if v in it.variants
        for r in range(cfg.repeats)
    ]
    random.Random(cfg.shuffle_seed).shuffle(jobs)

    run_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{cfg.name}"
    out_dir = Path(out_root) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    client = ChatClient(cfg.base_url, api_key=cfg.api_key)
    ok, detail = client.ping()
    if not ok:
        raise SystemExit(
            f"Cannot reach {cfg.base_url} ({detail}).\n"
            "Start a server first, e.g.  ./scripts/serve_llamacpp.sh"
        )
    print(f"endpoint ok: {cfg.base_url}  models: {detail}")

    sampler = make_sampler(cfg.power_sampler, cfg.power_interval_ms)
    sampler.start()
    idle_w = None
    if sampler.name != "none":
        print(f"power sampler: {sampler.name}; measuring idle baseline for {cfg.idle_baseline_s:.0f}s ...")
        idle_w = sampler.idle_baseline_w(cfg.idle_baseline_s)
        if idle_w is None:
            print(f"  WARNING: no power readings ({sampler.error or 'unknown reason'}); timing only.")
        else:
            print(f"  idle baseline: {idle_w:.2f} W")
    else:
        print("power sampler: none (timing only)")

    meta = {
        "run_id": run_id,
        "config": cfg.__dict__,
        "system": sysinfo.collect(),
        "power_sampler": sampler.name,
        "power_channel": sampler.primary,
        "idle_baseline_w": idle_w,
        "n_items": len(items),
        "variants": variants,
        "n_requests": len(jobs),
        "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    common = dict(model=cfg.model, max_tokens=cfg.max_tokens, temperature=cfg.temperature,
                  seed=cfg.seed, extra_body=cfg.extra_body)
    try:
        for i in range(cfg.warmup):
            # warm up on prompts from the far end of the schedule, not the ones measured next
            it, v, _ = jobs[-1 - (i % len(jobs))]
            client.complete(messages=_messages(cfg, it.variants[v]), **common)
            print(f"warm-up {i + 1}/{cfg.warmup} done")

        failures = 0
        with open(out_dir / "raw.jsonl", "w", encoding="utf-8") as fh:
            for n, (it, v, r) in enumerate(jobs, 1):
                c = client.complete(messages=_messages(cfg, it.variants[v]), **common)
                row = _row(cfg, run_id, it, v, r, c, sampler, idle_w)
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                if c.ok:
                    ttft = f"{c.ttft_s:.2f}s" if c.ttft_s is not None else "n/a"
                    rate = f"{c.decode_tok_s:.1f} tok/s" if c.decode_tok_s else "n/a"
                    print(f"[{n}/{len(jobs)}] {it.id:<10} {v:<7} in={c.prompt_tokens} "
                          f"out={c.completion_tokens} ttft={ttft} total={c.total_s:.2f}s {rate}")
                else:
                    failures += 1
                    print(f"[{n}/{len(jobs)}] {it.id:<10} {v:<7} FAILED: {c.error}")
                    if failures >= 5 and failures == n:
                        raise SystemExit("First 5 requests all failed; stopping. Check the server log.")
                time.sleep(cfg.cooldown_s)
    finally:
        sampler.stop()
        client.close()
        meta["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nraw results: {out_dir / 'raw.jsonl'}")
    return out_dir
