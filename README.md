# indic-inference-tax

**Indian languages pay more than English for the same AI answer. This project measures exactly how much, on the hardware Indians actually own, and then tries to close the gap.**

Recent papers have counted the extra *tokens* Indian languages need (about 8x English on average, 13x for Malayalam), measured *datacenter* energy per language, and shown that 4-bit quantization can collapse Hindi accuracy on small models. Nobody has put these together for Indic languages on commodity hardware, in the forms people really type (native script, romanized, code-mixed), and nobody has shown how much of the gap can be closed cheaply on an existing open model. That is the gap this repo is for. See [docs/01_problem_statement.md](docs/01_problem_statement.md) and [docs/02_prior_work.md](docs/02_prior_work.md).

## Status

Scaffold, v0.1. The harness is written and covered by a pytest suite, including end-to-end runs against a fake streaming server. **No real measurement has been taken yet.** The numbers start with `./scripts/first_measurement.sh` on your machine.

## Quick start (macOS, Apple Silicon)

```bash
./scripts/setup_mac.sh            # brew: uv, llama.cpp, macmon; creates .venv; runs tests + doctor
QUICK=1 ./scripts/first_measurement.sh   # 5-minute smoke test (after a one-time ~2.5 GB model download)
./scripts/first_measurement.sh    # the real first run
```

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
scripts/           setup_mac.sh, serve_llamacpp.sh, first_measurement.sh
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
