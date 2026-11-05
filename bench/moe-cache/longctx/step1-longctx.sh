#!/usr/bin/env bash
# long-context check, Qwen3.6 and Flint: -ncmoe vs cache (no warm-up) vs cache + warm-up; 30K prompt, 600 tokens x 2 requests, then llama-completion hit rates
# WARM=0 skips the warm-up configs (if the build-warm binary is not available)
Q="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ADV="$(cd "$Q/../../.." && pwd)"
W=${WARM_LLAMA:-$ADV/../llama.cpp-warm}
run() { # label model batch args...
  local L=$1 M=$2 B=$3; shift 3
  MODEL=$M BATCH=$B AFTER="OUT_PROMPT=$Q/logs/prompt-$L.txt ${PY:-python3} -I $Q/longctx.py" $Q/srv.sh lc-$L "$@"
}
QW=$ADV/models/qwen36-35b-a3b/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
FL=$ADV/models/flint-40b-a2.5b/flint-UD-Q3_K_XL.gguf
run qwen-ncmoe32 $QW 512 -ncmoe 32
LLAMA_MOE_CACHE_NO_WARM=1 run qwen-cache $QW 512 -cmoe --moe-cache-mib 2900 --moe-cache-cpu-share 0.95
[ "${WARM:-1}" = 1 ] && LLAMA=$W BIN=build-warm/bin/llama-server run qwen-warm $QW 512 -cmoe --moe-cache-mib 2900 --moe-cache-cpu-share 0.95
run qwen-ncmoe32-b $QW 512 -ncmoe 32
run flint-ncmoe42 $FL 2048 -ncmoe 42
LLAMA_MOE_CACHE_NO_WARM=1 run flint-cache $FL 2048 -cmoe --moe-cache-mib 2168 --moe-cache-cpu-share 0.95
[ "${WARM:-1}" = 1 ] && LLAMA=$W BIN=build-warm/bin/llama-server run flint-warm $FL 2048 -cmoe --moe-cache-mib 2168 --moe-cache-cpu-share 0.95
run flint-ncmoe42-b $FL 2048 -ncmoe 42
# hit rates on the same prompt
MODEL=$QW BATCH=512 LLAMA_MOE_CACHE_NO_WARM=1 $Q/hits.sh qwen-cache $Q/logs/prompt-qwen-cache.txt -cmoe --moe-cache-mib 2900 --moe-cache-cpu-share 0.95
[ "${WARM:-1}" = 1 ] && MODEL=$QW BATCH=512 LLAMA=$W BIN_COMPLETION=build-warm/bin/llama-completion $Q/hits.sh qwen-warm $Q/logs/prompt-qwen-cache.txt -cmoe --moe-cache-mib 2900 --moe-cache-cpu-share 0.95
MODEL=$FL BATCH=2048 LLAMA_MOE_CACHE_NO_WARM=1 $Q/hits.sh flint-cache $Q/logs/prompt-flint-cache.txt -cmoe --moe-cache-mib 2168 --moe-cache-cpu-share 0.95
[ "${WARM:-1}" = 1 ] && MODEL=$FL BATCH=2048 LLAMA=$W BIN_COMPLETION=build-warm/bin/llama-completion $Q/hits.sh flint-warm $Q/logs/prompt-flint-cache.txt -cmoe --moe-cache-mib 2168 --moe-cache-cpu-share 0.95
