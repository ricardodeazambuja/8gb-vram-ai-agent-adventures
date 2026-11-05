#!/usr/bin/env bash
# per-turn prefill cost (cached 2K base + N new tokens), Qwen3.8 cache: b512/3661 vs b2048/3174; optional extra env per config via EXTRA
Q="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ADV="$(cd "$Q/../../.." && pwd)"; L="$Q/../longctx"
export LOGDIR=${LOGDIR:-$Q/logs}
export MODEL=$ADV/models/qwen38-35b-a3b-distill/Qwen3.8-35B-A3B-IQ4_XS.gguf
BATCH=512  AFTER="${PY:-python3} -I $Q/turnbench.py" $L/srv.sh q38-turns-b512  -cmoe --moe-cache-mib 3661 --moe-cache-cpu-share 0.95
BATCH=2048 AFTER="${PY:-python3} -I $Q/turnbench.py" $L/srv.sh q38-turns-b2048 -cmoe --moe-cache-mib 3174 --moe-cache-cpu-share 0.95
