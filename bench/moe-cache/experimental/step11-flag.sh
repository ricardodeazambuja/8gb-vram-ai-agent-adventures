#!/usr/bin/env bash
# branch moe-cache-max-batch (69ca796dd, built in llama.cpp-warm/build-warm): --moe-cache-max-batch 32 vs 256 via the CLI flag, three models
Q="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ADV="$(cd "$Q/../../.." && pwd)"; LD="$Q/../longctx"; T="$Q/../models"; PY="${PY:-python3} -I"; MD=$ADV/models
export LOGDIR=${LOGDIR:-$Q/logs}
export LLAMA=${WARM_LLAMA:-$ADV/../llama.cpp-warm} BIN=build-warm/bin/llama-server RUNS=2 SIZES="16 48 100 200 300 400 600"
pair() { # label model batch cache
  for mb in 32 256; do MODEL=$2 BATCH=$3 AFTER="$PY $T/turnbench.py" $LD/srv.sh $1-mb$mb -cmoe --moe-cache-mib $4 --moe-cache-cpu-share 0.95 --moe-cache-max-batch $mb; done
}
pair flag-q38 $MD/qwen38-35b-a3b-distill/Qwen3.8-35B-A3B-IQ4_XS.gguf 2048 2900
pair flag-q36 $MD/qwen36-35b-a3b/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf 512 2900
pair flag-flint $MD/flint-40b-a2.5b/flint-UD-Q3_K_XL.gguf 2048 2168
