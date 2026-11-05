#!/usr/bin/env bash
# Qwen3.8 b2048, cache 2900 (VRAM-safe size), experimental build: LLAMA_MOE_CACHE_MAX_BATCH 32 / 256 / 128 / 32; srvbench x2 + turnbench; vram peak
Q="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ADV="$(cd "$Q/../../.." && pwd)"; LD="$Q/../longctx"; T="$Q/../models"; PY="${PY:-python3} -I"
export LOGDIR=${LOGDIR:-$Q/logs}
export MODEL=$ADV/models/qwen38-35b-a3b-distill/Qwen3.8-35B-A3B-IQ4_XS.gguf BATCH=2048 RUNS=2
export LLAMA=${WARM_LLAMA:-$ADV/../llama.cpp-warm} BIN=build-warm/bin/llama-server LLAMA_MOE_CACHE_NO_WARM=1
export SIZES="16 48 100 200 300 400 600 800"
A=(-cmoe --moe-cache-mib 2900 --moe-cache-cpu-share 0.95)
for mb in 32 256 128 32; do
  L=q38-c2900-mb$mb; [ -e "$LOGDIR/srv-$L.log" ] && L=$L-b
  LLAMA_MOE_CACHE_MAX_BATCH=$mb AFTER="$PY $T/turnbench.py" $LD/srv.sh $L "${A[@]}"
done
