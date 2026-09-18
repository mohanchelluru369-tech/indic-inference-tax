# indic-inference-tax

**Indian languages pay more than English for the same AI answer. This project measures exactly how much, on the hardware Indians actually own, and then tries to close the gap.**

Recent papers have counted the extra *tokens* Indian languages need (about 8x English on average, 13x for Malayalam), measured *datacenter* energy per language, and shown that 4-bit quantization can collapse Hindi accuracy on small models. Nobody has put these together for Indic languages on commodity hardware, in the forms people really type (native script, romanized, code-mixed), and nobody has shown how much of the gap can be closed cheaply on an existing open model. That is the gap this repo is for. See [docs/01_problem_statement.md](docs/01_problem_statement.md) and [docs/02_prior_work.md](docs/02_prior_work.md).

## Status

v0.1. The harness is written and covered by a pytest suite, including end-to-end runs against a fake streaming server. **Experiment 01 has run** — results below and in [`results/20260918-095750-fertility/`](results/20260918-095750-fertility/). Experiment 02 runs end to end on an Apple M5 Pro (real power sampling, flat decode rate across languages) but has no run worth quoting yet: the first pass was a three-item smoke test on battery, which this project's own rules say to discard.

## First result: the tax is a property of the tokenizer, not of the language

Eighteen parallel prompts, identical content in every column, tokens counted against the English baseline. 1.00 is parity; 7.59 means the same question costs 7.59x the tokens, and therefore roughly 7.59x the prefill compute, KV-cache memory and API spend.

| tokenizer | vocab | hi | te | hi_rom | te_rom | hi_cm | te_cm |
|---|---:|---:|---:|---:|---:|---:|---:|
| Llama-3.2-3B-Instruct | 128,256 | 2.44 | **7.59** | 1.87 | 1.68 | 1.21 | 1.09 |
| SmolLM3-3B | 128,256 | 2.44 | **7.59** | 1.87 | 1.68 | 1.21 | 1.09 |
| Qwen3-4B | 151,669 | 4.30 | **6.48** | 1.87 | 1.68 | 1.22 | 1.10 |
| Krutrim-2-instruct | 131,072 | 1.75 | 2.05 | 1.82 | 1.62 | 1.19 | 1.09 |
| Phi-4-mini-instruct | 200,029 | 1.45 | 1.75 | 1.71 | 1.57 | 1.16 | 1.05 |
| gpt-oss-20b (o200k) | 200,019 | 1.45 | 1.75 | 1.71 | 1.57 | 1.16 | 1.05 |
| gemma-3-4b-it | 262,145 | 1.18 | 1.58 | 1.59 | 1.57 | 1.04 | 1.06 |
| sarvam-1 | 68,096 | **1.14** | **1.19** | 2.18 | 1.91 | 1.41 | 1.22 |

Three things fall out of it.

**The spread between models is larger than the spread between languages.** Telugu costs 7.59x on Llama 3.2 and 1.19x on Sarvam-1 — the same sentences, a 6x difference in bill, decided entirely by which model you picked. Reporting "the Indic tokenizer tax" as one number per language, as the literature currently does, averages over the thing that actually matters.

**Vocabulary size is not the lever; script coverage is.** Sarvam-1 has the *smallest* vocabulary here (68k) and the lowest tax. Llama 3.2 has twice that and the highest. The mechanism is visible in the fragment rate: 98.2% of Llama's Telugu tokens are lone bytes or partial characters — it has no merges for the script and is spelling it out byte by byte — against 0% for Gemma 3 and 6% for Sarvam-1. Qwen3 fragments 52% of Hindi and 81% of Telugu. (Llama 3.2 and SmolLM3 agree to three digits because they ship the same vocabulary; Phi-4-mini and gpt-oss agree because both are built on o200k.)

**Romanizing is not a free saving, and on a good tokenizer it is a penalty.** On Sarvam-1, typing Telugu in Latin script costs 1.91x English against 1.19x for native script — writing it the way most people actually type it makes it *more* expensive, not less. On Gemma 3 it is a wash (1.57 vs 1.58). Only on the byte-fallback tokenizers is romanizing the large win it is assumed to be (1.68 vs 7.59 on Llama). Hundreds of millions of people make this trade every day and nobody has priced it.

Caveat, and it is not a small one: this is 18 prompts, written by an LLM and **not yet reviewed by native speakers** (see [data/README.md](data/README.md)). Treat the direction as real and the third digit as noise. The paper's numbers will come from FLORES+/IN22 and Belebele.

## Quick start (macOS, Apple Silicon)

```bash
./scripts/setup_mac.sh            # brew: uv, llama.cpp, macmon; creates .venv; runs tests + doctor
QUICK=1 ./scripts/first_measurement.sh   # 5-minute smoke test (after a one-time ~2.5 GB model download)
./scripts/first_measurement.sh    # Exp 01 + Exp 02 for one model
./scripts/exp02_pair.sh           # Exp 02 for the Llama/Gemma pair, back to back (~50 min, AC power)
```

`exp02_pair.sh` is the natural experiment: two models that differ by 4.8x in Telugu token cost, on one machine, one engine, one quantization. It refuses to run on battery or in Low Power Mode, because the methodology says those numbers get discarded.

Then read `results/<newest>/summary.md`.

## The experiments

| # | Question | Command | Needs |
|---|---|---|---|
| 01 | How many more tokens does the same content cost in Hindi/Telugu, native vs romanized vs code-mixed, per tokenizer family? | `indictax fertility` | tokenizer files only |
| 02 | What does that turn into on real hardware: time to first token, time per answer, joules per answer, answers cut off at the cap, answers in the wrong script? | `indictax bench configs/exp02_*.yaml` | a local model server |
| 02b | Same, but scored: joules per *correct* answer | `indictax make-belebele` then `bench` | Hugging Face access |
| 03 | Does quantization hurt Indic more than English at the same bit width? | planned, see roadmap | |
| 04 | Does speculative decoding speed up Indic less? | planned | |
| 05 | The fix: how much of the tax can be closed on an existing 1-4B model with tokenizer expansion, Indic-calibrated quantization and language-aware drafting? | planned | |

`indictax bench` talks to any OpenAI-compatible endpoint, so one harness covers llama.cpp, Ollama, MLX (`mlx_lm.server`) and vLLM, on the Mac, the GPU workstation, a cloud T4 and an Android phone running `llama-server` under Termux.

## Layout

```
src/indictax/      the harness (cli, fertility, client, bench, energy, report, belebele)
configs/           tokenizers.yaml, one YAML per Exp 02 model
data/prompts/      seed_v0.jsonl: 18 hand-written parallel prompts (needs native review, see data/README.md)
scripts/           setup_mac.sh, serve_llamacpp.sh, first_measurement.sh, exp02_pair.sh
tools/             review.html: offline corpus review page for native speakers
docs/              problem statement, prior work, methodology, roadmap, hardware matrix
paper/             outline of the write-up
results/           one folder per run: raw.jsonl, meta.json, summary.md, *.csv
tests/             pytest suite; tests/conftest.py holds the fake streaming server
```

## Ground rules for results

1. Every number ships with its machine, OS, engine version and config (`meta.json` does this automatically).
2. Comparisons are paired on the same content and reported as ratios with bootstrap intervals over items.
3. Prompt caching off, fixed seed, temperature 0, shuffled order, warm-up discarded, AC power, Low Power Mode off.
4. Energy numbers are compared within a platform, never across platforms.
5. If a run was taken in bad conditions, delete it. Do not average it in.

Details and the reasoning: [docs/03_methodology.md](docs/03_methodology.md).

## License

Code: MIT. The seed corpus in `data/prompts/`: CC BY 4.0.
