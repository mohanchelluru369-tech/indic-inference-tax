"""indictax command line.

    indictax doctor                       is this machine ready to produce trustworthy numbers?
    indictax fertility                    Exp 01: tokenizer tax (no model download, runs anywhere)
    indictax bench  configs/exp02_*.yaml  Exp 02: latency / throughput / energy per language
    indictax report results/<run>         summarise a run
    indictax make-belebele                build the scored parallel corpus (needs Hugging Face)
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import yaml

DEFAULT_CORPUS = "data/prompts/seed_v0.jsonl"
DEFAULT_TOKENIZERS = "configs/tokenizers.yaml"


def cmd_doctor(args: argparse.Namespace) -> int:
    import httpx

    from . import energy, sysinfo
    from .client import ChatClient
    from .corpus import load_corpus, variants_in

    problems = 0

    def line(ok: bool | None, what: str, detail: str = "") -> None:
        nonlocal problems
        mark = {True: "ok  ", False: "FAIL", None: "warn"}[ok]
        problems += ok is False
        print(f"[{mark}] {what}" + (f": {detail}" if detail else ""))

    info = sysinfo.collect()
    line(True, "machine", f"{info.get('chip') or info['machine']}, {info.get('ram_gb')} GB, {info['platform']}")
    line(True, "python", info["python"])
    if info["system"] == "Darwin":
        line(None if info.get("low_power_mode") else True, "low power mode",
             "ON - turn it off before benchmarking" if info.get("low_power_mode") else "off")
        line(True if info.get("on_ac_power") else None, "power source",
             "AC" if info.get("on_ac_power") else "battery - plug in before benchmarking")
    if info.get("gpu"):
        line(True, "gpu", info["gpu"])

    for tool in ("llama-server", "ollama", "macmon"):
        v = info.get(f"tool:{tool}")
        line(True if v else None, tool, v or "not installed")

    chk = energy.self_check(args.power)
    line(True if chk.ok else None, f"power sampler ({chk.name})",
         f"{chk.detail}; last reading {chk.example}" if chk.ok else chk.detail)

    for url in args.endpoint:
        c = ChatClient(url)
        ok, detail = c.ping()
        c.close()
        line(True if ok else None, f"endpoint {url}", detail if ok else f"not reachable ({detail})")

    try:
        r = httpx.get("https://huggingface.co/api/models/Qwen/Qwen3-0.6B", timeout=10)
        line(r.status_code == 200, "hugging face reachable", f"HTTP {r.status_code}")
    except Exception as e:  # noqa: BLE001
        line(False, "hugging face reachable", f"{type(e).__name__}: {e}")

    try:
        items = load_corpus(args.corpus)
        pending = sum(1 for it in items if it.review == "pending")
        line(True, "corpus", f"{len(items)} items, variants {variants_in(items)}")
        if pending:
            line(None, "corpus review", f"{pending} items still need native-speaker review")
    except Exception as e:  # noqa: BLE001
        line(False, "corpus", str(e))

    print("\nready." if not problems else f"\n{problems} blocking problem(s).")
    return 1 if problems else 0


def cmd_fertility(args: argparse.Namespace) -> int:
    from . import fertility
    from .corpus import load_corpus
    from .tok import load_tokenizer

    items = load_corpus(args.corpus)
    specs = args.tokenizer or [
        t["spec"] for t in yaml.safe_load(Path(args.tokenizers).read_text(encoding="utf-8"))["tokenizers"]
    ]
    rows, skipped = [], []
    for spec in specs:
        try:
            tk = load_tokenizer(spec)
        except Exception as e:  # noqa: BLE001 - one gated repo must not sink the run
            skipped.append((spec, f"{type(e).__name__}: {str(e)[:160]}"))
            print(f"skip {spec}: {skipped[-1][1]}")
            continue
        rows += fertility.measure(tk, items)
        print(f"done {spec}")
    if not rows:
        print("no tokenizer could be loaded; nothing to report.")
        return 1

    out_dir = Path(args.out) / datetime.now().strftime("%Y%m%d-%H%M%S-fertility")
    fertility.write_csv(rows, out_dir / "fertility.csv")
    md = f"# Tokenizer tax\n\ncorpus: `{args.corpus}` ({len(items)} items)\n\n" + fertility.to_markdown(rows)
    if skipped:
        md += "\n## Skipped\n\n" + "\n".join(f"- `{s}`: {why}" for s, why in skipped) + "\n"
    (out_dir / "fertility.md").write_text(md, encoding="utf-8")
    print("\n" + md)
    print(f"written: {out_dir}")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    from . import bench, report

    cfg = bench.BenchConfig.from_yaml(args.config)
    if args.limit_items:
        cfg.limit_items = args.limit_items
    if args.repeats:
        cfg.repeats = args.repeats
    run_dir = bench.run(cfg, out_root=args.out)
    summary = report.write(run_dir)
    print(summary.read_text(encoding="utf-8"))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from . import report

    summary = report.write(args.run_dir)
    print(summary.read_text(encoding="utf-8"))
    return 0


def cmd_make_belebele(args: argparse.Namespace) -> int:
    from . import belebele

    belebele.make(args.out, n=args.n, seed=args.seed)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="indictax", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("doctor", help="check this machine")
    d.add_argument("--corpus", default=DEFAULT_CORPUS)
    d.add_argument("--power", default="auto")
    d.add_argument("--endpoint", action="append",
                   default=None, help="OpenAI-compatible base URL (repeatable)")
    d.set_defaults(fn=cmd_doctor)

    f = sub.add_parser("fertility", help="Exp 01: tokenizer tax")
    f.add_argument("--corpus", default=DEFAULT_CORPUS)
    f.add_argument("--tokenizers", default=DEFAULT_TOKENIZERS, help="YAML list of tokenizer specs")
    f.add_argument("--tokenizer", action="append", help="one spec; repeatable; overrides --tokenizers")
    f.add_argument("--out", default="results")
    f.set_defaults(fn=cmd_fertility)

    b = sub.add_parser("bench", help="Exp 02: serving latency / energy per language")
    b.add_argument("config")
    b.add_argument("--out", default="results")
    b.add_argument("--limit-items", type=int, help="smoke test on the first N items")
    b.add_argument("--repeats", type=int)
    b.set_defaults(fn=cmd_bench)

    r = sub.add_parser("report", help="summarise a finished run")
    r.add_argument("run_dir")
    r.set_defaults(fn=cmd_report)

    m = sub.add_parser("make-belebele", help="build the scored parallel corpus from Belebele")
    m.add_argument("--out", default="data/prompts/belebele_100.jsonl")
    m.add_argument("--n", type=int, default=100)
    m.add_argument("--seed", type=int, default=0)
    m.set_defaults(fn=cmd_make_belebele)

    args = p.parse_args(argv)
    if args.cmd == "doctor" and not args.endpoint:
        args.endpoint = ["http://127.0.0.1:8080/v1", "http://127.0.0.1:11434/v1"]
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
