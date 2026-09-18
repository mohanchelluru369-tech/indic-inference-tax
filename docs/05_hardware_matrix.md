# 05. Hardware matrix

One engine (llama.cpp), one GGUF file per model, every device. Extra engines are extra rows.

| Device | Role in the paper | Engine(s) | Power channel | Status |
|---|---|---|---|---|
| MacBook Pro, M5 Pro, 24 GB | primary dev machine; "good laptop" row | llama.cpp (Metal); MLX as an extra row | `macmon` (no sudo), `powermetrics` fallback | set up first |
| Desktop, RTX 5070 Ti 16 GB, Ryzen 7 9700X | consumer GPU row; also where training for the fix can run | llama.cpp (CUDA); vLLM as an extra row | `nvidia-smi` (GPU board only) | week 3 |
| GCP n1-standard-4 + T4 | cheap cloud GPU row, what a small Indian startup rents | llama.cpp (CUDA); vLLM | `nvidia-smi` | week 3 |
| Budget Android phone, about ₹10-15k, 4-6 GB RAM | **the row that matters** | llama.cpp under Termux (CPU) | battery current x voltage, see below | to acquire in week 3 |

What fits in 24 GB of unified memory on the Mac: up to about 8B at Q4 comfortably alongside the OS; 1-4B models for anything involving training (LoRA or embedding tuning with MLX). macOS caps GPU-visible memory at roughly two thirds to three quarters of RAM by default, so do not plan on a 14B+ model.

## Notes that were not verified on real hardware

This scaffold was written in a Linux sandbox with no Apple Silicon, no GPU and no access to Hugging Face. The following are written from documentation and memory and **must be confirmed on the first run**; `indictax doctor` is there to surface most of them:

- `macmon pipe -s 0 -i 250` flags and its JSON field names (`all_power`, `sys_power`, `ram_power`, `temp.gpu_temp_avg`), and whether the installed macmon version supports the M5 generation. If not: `sudo -v` then set `power.sampler: powermetrics` in the config. powermetrics may buffer its output when piped; if readings arrive in bursts, say so and we will switch to writing it to a file with `-o`.
- `brew install llama.cpp macmon uv` formula names.
- GGUF repo ids in the configs and scripts (`ggml-org/gemma-3-4b-it-GGUF`, `bartowski/Llama-3.2-3B-Instruct-GGUF`) and every tokenizer repo id in `configs/tokenizers.yaml`.
- That llama-server honours `cache_prompt: false` and returns `usage` and `timings` in the final streamed chunk on the installed version. Check one row of `raw.jsonl`: `prompt_tokens` and `server_prefill_tok_s` should not be null.
- The Belebele loader (`indictax make-belebele`) against the live dataset.

## Linux GPU boxes

```bash
# llama.cpp with CUDA: build from source or use a release binary, then
pip install -e ".[hf]"
./scripts/serve_llamacpp.sh ggml-org/gemma-3-4b-it-GGUF
indictax bench configs/exp02_gemma3_4b.yaml        # power.sampler: auto picks nvidia-smi

# vLLM as an extra row: prefix caching OFF or TTFT is wrong
vllm serve <model> --no-enable-prefix-caching --max-num-seqs 1
# then copy a config, point base_url at :8000/v1, set endpoint.model to the served id,
# and remove cache_prompt from extra_body (vLLM rejects unknown fields on some versions).
```

## Android under Termux (plan)

```bash
pkg install git cmake clang python
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp && cmake -B build && cmake --build build -j
./build/bin/llama-server -m model.gguf --host 0.0.0.0 --port 8080 -c 2048 -np 1
```

Run `indictax bench` from the Mac against the phone's IP so the harness does not steal CPU from the model; copy a config and change `base_url`. Energy: there is no sampler for Android yet. The plan is a small Termux loop that logs `termux-battery-status` (current in µA, no root) once a second with timestamps, joined to `raw.jsonl` afterwards by wall-clock time. Unplugged, screen on at fixed brightness, airplane mode plus Wi-Fi. Report a 10-minute sustained run as well as the interleaved one; phones throttle hard and that is part of the result.
