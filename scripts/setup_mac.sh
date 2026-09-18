#!/usr/bin/env bash
# One-time setup on macOS (Apple Silicon). Safe to re-run.
#
# Installs with Homebrew:  uv (Python env), llama.cpp (llama-server), macmon (power, no sudo)
# Creates:                 .venv with this package installed, a git repo if there is none
# Downloads no models. The first model (~2.5 GB) is fetched by first_measurement.sh.
set -euo pipefail
cd "$(dirname "$0")/.."

[[ "$(uname -s)" == "Darwin" ]] || { echo "This script is for macOS. On Linux see docs/05_hardware_matrix.md."; exit 1; }
command -v brew >/dev/null || { echo "Homebrew is required: https://brew.sh"; exit 1; }

for pkg in uv llama.cpp macmon; do
  if brew list --formula "$pkg" &>/dev/null; then
    echo "== $pkg already installed"
  else
    echo "== installing $pkg"
    brew install "$pkg"
  fi
done

[[ -d .venv ]] || uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[hf,openai-tok,dev]"

if [[ ! -d .git ]]; then
  git init -q
  git add -A
  git commit -q -m "Scaffold: indic-inference-tax measurement harness" || true
  echo "== git repository initialised"
fi

echo
echo "== tests"
.venv/bin/pytest

echo
echo "== doctor"
.venv/bin/indictax doctor || true

cat <<'EOF'

Next:
  ./scripts/first_measurement.sh        # Exp 01 (tokenizers) + Exp 02 (Gemma 3 4B on this Mac)

Before any run you intend to keep: plug in, turn Low Power Mode off, close heavy apps.
EOF
