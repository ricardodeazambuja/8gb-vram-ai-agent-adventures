#!/usr/bin/env bash
#
# start_flint.sh — Recursive Flint (40B MoE, 2.5B active) + pi, no env vars needed.
#
#   bash scripts/start_flint.sh                                 # server (if needed) + pi
#   CTX=65536 bash scripts/start_flint.sh                       # smaller context
#   bash scripts/start_flint.sh -p "explain @README.md" --no-session   # one-shot
#
# A thin wrapper over scripts/start-qwen.sh with QMODEL=flint. The server settings come from
# run-qwen.sh's flint defaults: UD-Q3_K_XL, experts in RAM, 128K context, q8_0 KV, flash
# attention on, our llama.cpp in ../llama.cpp (fork, adventures branch: upstream of 2026-10-28;
# flint is upstream since PR #30928, 2026-10-21).
# Details and measurements: docs/models/flint.md. Every start-qwen.sh / run-qwen.sh knob
# (CTX, KVQUANT, PORT, ...) still works if you set it; you never have to.
#
# Refuses to start if another model is already serving on the port: start-qwen.sh would
# reuse that server, and pi would be talking to the wrong model. Stop it first with
# scripts/stop-server.sh.
#
set -euo pipefail

for _arg in "$@"; do case "$_arg" in
  -h|--help) sed -n '2,/^[^#]/{/^#/s/^# \?//p}' "${BASH_SOURCE[0]}"; exit 0 ;;
esac; done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"

MODEL_FILE="$REPO_ROOT/models/flint-40b-a2.5b/flint-UD-Q3_K_XL.gguf"
if [ ! -f "$MODEL_FILE" ]; then
  echo "ERROR: model not found: $MODEL_FILE"
  echo "       download it: bash scripts/download-models.sh flint"
  exit 1
fi
SERVER_BIN="${ADV_LLAMA:-$REPO_ROOT/../llama.cpp}/build/bin/llama-server"
if [ ! -x "$SERVER_BIN" ]; then
  echo "ERROR: our llama.cpp build not found: $SERVER_BIN"
  echo "       build it (see docs/getting-started.md), next to this repo:"
  echo "       CUDA_VER=12.2 LLAMA_REPO=https://github.com/ricardodeazambuja/llama.cpp \\"
  echo "         LLAMA_REF=f0456c64d7b891ada1a0a924716ce3244122e92b SRC_DIR=\$PWD/../llama.cpp \\"
  echo "         bash scripts/build-llama-cuda.sh"
  exit 1
fi

# Something already serving? Fine if it is Flint, refuse otherwise.
if curl -fsS "http://$HOST:$PORT/health" >/dev/null 2>&1; then
  SERVED_ID="$(curl -fsS "http://$HOST:$PORT/v1/models" 2>/dev/null \
    | python3 -c 'import sys,json; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null || true)"
  if [ "$SERVED_ID" != "flint" ]; then
    echo "ERROR: port $PORT is already serving '${SERVED_ID:-unknown}', not Flint."
    echo "       stop it first: bash scripts/stop-server.sh"
    exit 1
  fi
  echo ">> Flint is already running on port $PORT; reusing it."
fi

export QMODEL=flint
exec bash "$REPO_ROOT/scripts/start-qwen.sh" "$@"
