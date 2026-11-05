#!/usr/bin/env bash
# experimental build (llama.cpp-warm, branch moe-cache-warm): Qwen3.8 b2048 cache 3174 s0.95
# 1 no warm-up | 2 soft warm-up | 3 no warm-up + LLAMA_MOE_CACHE_MAX_BATCH=512 | 4 no warm-up again (drift)
Q="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ADV="$(cd "$Q/../../.." && pwd)"; L="$Q/../longctx"; T="$Q/../models"; PY="${PY:-python3} -I"
export LOGDIR=${LOGDIR:-$Q/logs}
export MODEL=$ADV/models/qwen38-35b-a3b-distill/Qwen3.8-35B-A3B-IQ4_XS.gguf BATCH=2048 RUNS=2
export LLAMA=${WARM_LLAMA:-$ADV/../llama.cpp-warm} BIN=build-warm/bin/llama-server
A=(-cmoe --moe-cache-mib 3174 --moe-cache-cpu-share 0.95)
LLAMA_MOE_CACHE_NO_WARM=1 AFTER="$PY $L/longctx.py; $PY $T/turnbench.py" $L/srv.sh exp-q38-nowarm "${A[@]}"
AFTER="$PY $L/longctx.py; $PY $T/turnbench.py" $L/srv.sh exp-q38-softwarm "${A[@]}"
LLAMA_MOE_CACHE_NO_WARM=1 LLAMA_MOE_CACHE_MAX_BATCH=512 AFTER="$PY $T/turnbench.py" $L/srv.sh exp-q38-maxb512 "${A[@]}"
LLAMA_MOE_CACHE_NO_WARM=1 $L/srv.sh exp-q38-nowarm-b "${A[@]}"
