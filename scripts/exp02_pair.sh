#!/usr/bin/env bash
# Exp 02, the natural experiment.
#
# Llama 3.2 3B and Gemma 3 4B differ by 4.8x in Telugu token cost (Exp 01) but
# run on the same machine, the same engine and the same quantization. Running
# them back to back turns "Telugu needs more tokens" into "Telugu costs this
# much more time and this many more joules on this laptop", with tokenizer
# coverage as the only plausible cause. It is also the direct test of H1:
# decode tok/s should be flat across languages *and* across these two models.
#
#   ./scripts/exp02_pair.sh              # both models, full corpus (~50 min)
#   QUICK=1 ./scripts/exp02_pair.sh      # 3 items, 1 repeat, for a shakedown
#
# Refuses to run in conditions whose numbers the methodology says to discard.
set -euo pipefail
cd "$(dirname "$0")/.."

BIN=.venv/bin/indictax
PORT="${PORT:-8080}"
[[ -x "$BIN" ]] || { echo "Run ./scripts/setup_mac.sh first."; exit 1; }

# --- conditions ------------------------------------------------------------
if [[ "$(uname -s)" == "Darwin" ]]; then
  if ! pmset -g batt | grep -q "AC Power"; then
    echo "REFUSING: on battery. macOS throttles, and the run would have to be discarded."
    echo "Plug in and try again."
    exit 1
  fi
  LPM="$(pmset -g | awk '$1 == "lowpowermode" { print $2 }')"
  if [[ "${LPM:-0}" == "1" ]]; then
    echo "REFUSING: Low Power Mode is on. Turn it off in System Settings > Battery."
    exit 1
  fi
fi
echo "Close heavy apps before a run you intend to keep. Starting in 5s (Ctrl-C to abort)."
sleep 5

# model repo                                    config
PAIRS=(
  "bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M  configs/exp02_llama32_3b.yaml"
  "ggml-org/gemma-3-4b-it-GGUF                  configs/exp02_gemma3_4b.yaml"
)

SERVER_PID=""
cleanup() { [[ -n "$SERVER_PID" ]] && kill "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT

mkdir -p results
for pair in "${PAIRS[@]}"; do
  read -r MODEL CONFIG <<<"$pair"
  LOG="results/llama-server-$(basename "$CONFIG" .yaml).log"
  echo
  echo "================================================================"
  echo "  $CONFIG"
  echo "  $MODEL"
  echo "================================================================"

  ./scripts/serve_llamacpp.sh "$MODEL" >"$LOG" 2>&1 &
  SERVER_PID=$!
  echo -n "loading (first run downloads the model) "
  for _ in $(seq 1 900); do
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then echo " ready"; break; fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo; echo "llama-server exited. Last 20 lines of $LOG:"; tail -20 "$LOG"; exit 1
    fi
    echo -n "."; sleep 2
  done
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo; echo "not ready after 30 min"; exit 1; }

  # The config's `label` claims a quantization. Print what actually loaded so a
  # mislabelled table is caught now rather than in review.
  echo "--- files the server actually loaded (check this matches the config label) ---"
  grep -o "[^ /']*\.gguf" "$LOG" | sort -u || echo "(could not parse the log)"
  echo "---"

  if [[ "${QUICK:-0}" == "1" ]]; then
    "$BIN" bench "$CONFIG" --limit-items 3 --repeats 1
  else
    "$BIN" bench "$CONFIG"
  fi

  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
  SERVER_PID=""
  sleep 5   # let the port free and the SoC settle before the next model
done

echo
echo "Both runs done. Compare the two newest folders under results/:"
ls -dt results/*/ | head -2
echo
echo "Read the 'tax' table in each summary.md. The thing to look at first is"
echo "decode tok/s: if it is flat across variants in both runs, the whole tax is"
echo "token count, which is H1. If it is not flat, that is a finding."
