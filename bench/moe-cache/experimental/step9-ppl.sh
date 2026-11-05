#!/usr/bin/env bash
# correctness of LLAMA_MOE_CACHE_MAX_BATCH: perplexity (8 x 2048-token chunks, ubatch 512) with the cache path for ubatch<=32 (default) vs <=512, and -ncmoe 31 as the no-cache reference
Q="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ADV="$(cd "$Q/../../.." && pwd)"; L="$Q/../longctx"; W=${WARM_LLAMA:-$ADV/../llama.cpp-warm}; M=$ADV/models/qwen38-35b-a3b-distill/Qwen3.8-35B-A3B-IQ4_XS.gguf
P="$L/logs/prompt-qwen.txt"
${PY:-python3} -I "$L/memcheck.py" "$M" || exit 1
pgrep -f "[m]emguard.py" >/dev/null || { echo "memguard not running"; exit 1; }
run() { local L=$1; shift
  ( cd $W && systemd-run --user --scope -q --unit=moep-$L mamba run --no-capture-output -n llamacpp-cuda build-warm/bin/llama-perplexity -m $M -ngl 99 -c 2048 -b 2048 -ub 512 -fa on -ctk q8_0 -ctv q8_0 -t 6 --chunks 8 -f "$P" -lv 4 "$@" > $Q/logs/ppl-$L.log 2>&1 )
  echo "### $L | $(cd $W && git log --oneline -1 | cut -c1-50) | $* | MAX_BATCH=${LLAMA_MOE_CACHE_MAX_BATCH:-32}"
  grep -aE 'Final estimate|prompt eval time|llama_moe_cache: ubatch' $Q/logs/ppl-$L.log | sed -E 's/^[0-9.]+ [IWED] //' | sed 's/^/    /'
}
run ncmoe31 -ncmoe 31
LLAMA_MOE_CACHE_NO_WARM=1 run cache-mb32 -cmoe --moe-cache-mib 3174 --moe-cache-cpu-share 0.95
LLAMA_MOE_CACHE_NO_WARM=1 LLAMA_MOE_CACHE_MAX_BATCH=512 run cache-mb512 -cmoe --moe-cache-mib 3174 --moe-cache-cpu-share 0.95
