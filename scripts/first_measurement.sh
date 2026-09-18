#!/usr/bin/env bash
# The first real numbers.
#   1. Exp 01: tokenizer tax across model families (downloads tokenizer.json files only)
#   2. Exp 02: Gemma 3 4B on this machine: latency, throughput, energy per language
#      (downloads ~2.5 GB on first use; starts and stops llama-server for you)
#
#   ./scripts/first_measurement.sh            # full run, roughly 20-40 minutes on an M-series Mac
#   QUICK=1 ./scripts/first_measurement.sh    # smoke test: 3 items, 1 repeat
set -euo pipefail
cd "$(dirname "$0")/.."

BIN=.venv/bin/indictax
[[ -x "$BIN" ]] || { echo "Run ./scripts/setup_mac.sh first."; exit 1; }

MODEL="${MODEL:-ggml-org/gemma-3-4b-it-GGUF}"
CONFIG="${CONFIG:-configs/exp02_gemma3_4b.yaml}"
PORT=8080
mkdir -p results

echo "=== Exp 01: tokenizer tax ==="
"$BIN" fertility || echo "(fertility step failed; continuing)"

echo
echo "=== Exp 02: serving on this machine ==="
STARTED_SERVER=0
if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo "using the llama-server already running on :$PORT"
else
  echo "starting llama-server for $MODEL (log: results/llama-server.log)"
  ./scripts/serve_llamacpp.sh "$MODEL" >results/llama-server.log 2>&1 &
  SERVER_PID=$!
  STARTED_SERVER=1
  trap '[[ $STARTED_SERVER == 1 ]] && kill "$SERVER_PID" 2>/dev/null || true' EXIT
  echo -n "waiting for the model to download and load "
  for _ in $(seq 1 900); do            # up to 30 minutes for the first download
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then echo " ready"; break; fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo; echo "llama-server exited. Last lines of its log:"; tail -20 results/llama-server.log; exit 1
    fi
    echo -n "."; sleep 2
  done
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo; echo "server not ready after 30 min"; exit 1; }
fi

if [[ "${QUICK:-0}" == "1" ]]; then
  "$BIN" bench "$CONFIG" --limit-items 3 --repeats 1
else
  "$BIN" bench "$CONFIG"
fi

echo
echo "Done. Open the newest folder under results/ and read summary.md."
