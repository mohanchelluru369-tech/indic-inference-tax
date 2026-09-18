# 03. Methodology

The point of this project is that performance-engineering discipline is what the existing literature lacks. A reviewer should not be able to explain any headline number by a measurement artefact. This file is the list of artefacts and what the harness does about each.

## Design: paired, ratio-based

Every item in a corpus carries the same content in several variants (`en`, `hi`, `te`, `hi_rom`, `te_rom`, `hi_cm`, `te_cm`). Every comparison is a ratio to English **on the same item, same model, same machine, same run**. The reported tax is the median of per-item ratios with a 95% bootstrap interval that resamples items, not requests, because repeats of a prompt are not independent observations.

Ratios travel; absolute numbers do not. "Telugu costs 2.3x English on this model" can be compared across machines. "Telugu takes 9.1 s" cannot.

## Metrics

| Metric | Definition | Why |
|---|---|---|
| input / output tokens | from the server's `usage` block | only the server knows the true tokenization |
| TTFT | client clock, request sent to first *visible text* | what the user feels; prefill plus the hold-back described below |
| server prefill ms | llama.cpp's own `timings` | engine-level prefill, free of the hold-back effect |
| time per answer | request sent to last chunk | the user-level cost of a complete answer |
| decode tok/s | the server's own figure when it reports one, else (output tokens - 1) / (end - first chunk) | should be language-independent; it is the control |
| chunk gap p50/p95 | time between visible chunks | how smoothly text appears; chunks are not tokens |
| graphemes/s | output grapheme clusters / total time | content throughput in a unit that is fair across scripts (bytes are not: Indic code points are 3 bytes) |
| cut-off rate | share of answers with `finish_reason = length` | a fixed token cap silently truncates Indic answers first |
| expected-script share | share of output letters in the script the variant calls for | catches drift into English or the wrong script |
| energy per answer | integral of sampled power over the request window, minus idle baseline x duration | net cost of the work |
| accuracy, joules per correct answer | scored corpora only (Belebele MCQ) | cost of a *useful* answer |
| tax | variant / English, paired by item | the headline |
| fragment rate (Exp 01) | share of tokens that are lone bytes or partial UTF-8 | the mechanism behind high fertility |

## Artefacts and controls

**Prompt and prefix caching.** llama.cpp reuses the KV cache for a repeated prompt, and vLLM caches shared prefixes. Repeat 2 of a prompt would get its prefill for free and TTFT would look wonderful. Configs send `cache_prompt: false` to llama.cpp; vLLM must be started with `--no-enable-prefix-caching`. Ollama gives no such control, so do not use Ollama for TTFT claims. A test asserts the flag is sent.

**Thermal drift.** Laptops and phones slow down as they heat up. If English runs first and Telugu last, Telugu inherits the throttling. Request order is shuffled with a fixed seed so variants interleave; temperature is logged per request when the sampler provides it; `cooldown_s` sits between requests. On a phone, raise the cooldown and report a sustained-load run separately rather than hiding it.

**Warm-up.** First requests pay for shader compilation, memory mapping and cache fill. `warmup` requests are sent and discarded.

**Power state.** `meta.json` records whether the Mac was on AC power and whether Low Power Mode was on; the report prints a warning that the run should be discarded. `indictax doctor` checks before you start.

**Visible-character hold-back.** Engines do not stream half a character. When a script is tokenized into byte fragments, the first visible character needs several decode steps, so client-side TTFT for Telugu includes a few decode steps that English does not pay, and the gap between chunks is several tokens long. This is a real, language-dependent cost to the user and is reported as such, but it is not prefill. For engine-level claims use `server_prefill_ms`; for the decode-rate control the report prefers the server's own rate, because the client estimate is biased upward when the first chunk carries several tokens. A probe against a simulated server showed about 7% bias at 3 tokens per character.

**Failures.** An HTTP error, an error object inside a 200 stream (how llama.cpp reports context overflow), or a stream that closes without a `finish_reason` is a failed request: recorded with `ok: false`, excluded from statistics, counted in the `failed` column. If failures differ by variant the comparison is compromised; the report says so.

**Scoring.** Multiple-choice answers are parsed conservatively (`parse_mcq`): a bare letter, a leading `B)`, an explicit cue such as "answer is C" / "उत्तर: B" / "సమాధానం: C", or a lone letter after the final sentence. Taking the first capital A-D would score the English article in "A farmer must ..." as answer A and bias English only. Unparsed answers count as wrong, and the report warns when more than 5% of a variant's answers are unparsed.

**Energy coverage.** Energy is integrated only when power readings bracket the request and have no hole longer than max(2 s, 8 sampling intervals). Otherwise it is null. Interpolation would otherwise keep producing joules after a sampler died.

**Determinism.** Temperature 0, fixed seed, one slot (`-np 1`), single stream. Concurrency is a later experiment, not a hidden variable.

**Thinking models.** Models that emit hidden reasoning (Qwen3 default mode, and others) inflate output tokens in a way that differs by language. Either disable it through `extra_body` for that engine or report reasoning and answer tokens separately. The client records `reasoning_chars`. The first-measurement models (Gemma 3, Llama 3.2) do not reason.

**The instruction-language confound.** The system prompt and the Belebele answer instruction are in English for every variant: a constant token overhead that slightly *understates* the ratio. This is disclosed rather than "fixed" by translating the instruction, which would add translation quality as a new variable.

**Energy is per-platform.** `macmon`/`powermetrics` report SoC power (CPU + GPU + ANE, plus RAM for macmon). `nvidia-smi` reports GPU board power and excludes the host CPU. The numbers are not comparable across platforms and the paper must not put them on one axis. Within a platform, the language ratios are what is claimed. Sampling is at 250 ms; requests shorter than about 1 s have too few samples and their energy should not be reported individually.

**Same artefact everywhere.** One engine (llama.cpp) and the identical GGUF file on every device, so that the only thing that changes between hardware rows is the hardware. MLX and vLLM runs are additional, labelled rows, not substitutes.

**No invented numbers.** If a sampler produces no readings near a request window, energy is recorded as null. If a tokenizer cannot be loaded, it is listed as skipped with the reason. Failed requests are rows with `ok = false`, not silently dropped.

## Statistics

The interval is a percentile bootstrap of the median ratio over items, with an independent random stream per table cell so that an interval does not change when other variants are added to a run. With 16-18 items its real coverage is about 93-94% for a nominal 95% (checked by simulation), so treat intervals that barely exclude 1.00 as suggestive. The paper-scale corpora (100+ Belebele items, FLORES+/IN22) are what the claims should rest on.

## Corpora

- `seed_v0.jsonl`: 18 hand-written items, 7 variants for short prompts and 3 for long-context ones. Written by an LLM, **not yet reviewed by native speakers**; good enough to shake down the pipeline and get a first signal, not good enough to publish on. See `data/README.md`.
- Belebele (`indictax make-belebele`): parallel, scored, has `hin_Latn` (romanized Hindi) but no romanized Telugu.
- For the paper: FLORES+ or IN22 for token-level parallel text across all 22 scheduled languages (Exp 01 can cover all of them cheaply; Exp 02 onward stays on hi/te plus perhaps two more).
- Romanized and code-mixed Telugu at scale is an open data question. Options: collect real user-style romanization from volunteers (best), or transliterate with AI4Bharat IndicXlit and have natives fix it (cheaper). Machine romanization alone is not what people type and should not be presented as such.

## Threats to validity to state in the paper

Small seed corpus; machine-written prompts; one engine as the common denominator; English instructions; single-stream only; energy channels differ by platform; open models only; Hindi and Telugu are two of twenty-two scheduled languages and both are comparatively well resourced, so the result is a lower bound for the rest.
