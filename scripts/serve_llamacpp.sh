#!/usr/bin/env bash
# Start llama.cpp's OpenAI-compatible server for one GGUF model.
#
#   ./scripts/serve_llamacpp.sh                                               # Gemma 3 4B, default quant
#   ./scripts/serve_llamacpp.sh bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M
#   PORT=8081 CTX=8192 ./scripts/serve_llamacpp.sh <hf-repo>[:quant]
#
# The same command works on the Mac (Metal), the RTX workstation and the T4 (CUDA build),
# and on Android under Termux (CPU). Same engine + same GGUF file everywhere is the point:
# the only thing that changes between rows of the results table is the hardware.
set -euo pipefail

MODEL="${1:-ggml-org/gemma-3-4b-it-GGUF}"
PORT="${PORT:-8080}"
CTX="${CTX:-4096}"

command -v llama-server >/dev/null || { echo "llama-server not found. macOS: brew install llama.cpp"; exit 1; }

# -np 1   one slot: single-stream, and the whole context belongs to the one request
# -ngl 99 offload every layer to the GPU (Metal / CUDA); harmless on CPU-only builds
exec llama-server -hf "$MODEL" --host 127.0.0.1 --port "$PORT" -c "$CTX" -np 1 -ngl 99
