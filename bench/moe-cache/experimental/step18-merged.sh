#!/usr/bin/env bash
# Step 18 (2026-11-07 night, desktop ~0.4 GB): does upstream's merged #30966 (14 commits, incl. c6717826c: CPU misses
# chosen by CRF instead of at random) keep our numbers? old/new/old on each model, launcher MOE_CACHE=auto configs:
#   old = main build (moe-cache-max-batch, functionally 69ca796dd = pre-merge PR 3a72c57be + our max-batch patch)
#   new = branch moe-cache-max-batch-merged (b11f5e64d, worktree ../llama.cpp-merged; NEW_LLAMA/NEW_BIN): our tested tip
#         84137489b + upstream's ClassLRU eviction policy (#31233, 794a8c4e9; replaced the 50/50 SLRU we measured) + the
#         PR's 5 final commits cherry-picked (fed73adc1 0290c8da4 c6717826c 1f17b9c97 c0b02cd6f). llama-moe-cache.* equal
#         the merged PR's byte for byte except --moe-cache-max-batch. Not a full upstream sync (252 commits, PrismML conflicts).
# srvbench x2 (decode) + turnbench (48-600-token turns); srv.sh samples VRAM every second and prints the peak.
Q="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ADV="$(cd "$Q/../../.." && pwd)"; LD="$Q/../longctx"; T="$Q/../models"; MD=$ADV/models
pgrep -f "[m]emguard.py" >/dev/null || { echo "memguard not running, refusing"; exit 1; }
: "${NEW_LLAMA:?set NEW_LLAMA (llama.cpp dir of the merged build)}" "${NEW_BIN:?set NEW_BIN (server path relative to NEW_LLAMA)}"
OLD_LLAMA=${OLD_LLAMA:-$ADV/../llama.cpp} OLD_BIN=${OLD_BIN:-build/bin/llama-server}
export LOGDIR=${LOGDIR:-$Q/logs} RUNS=2 SIZES="48 100 200 600" AFTER="${PY:-python3} -I $T/turnbench.py"
abba() { # label model batch cache maxbatch
  local l=$1 m=$2 b=$3 a=(-cmoe --moe-cache-mib "$4" --moe-cache-cpu-share 0.95 --moe-cache-max-batch "$5")
  LLAMA=$OLD_LLAMA BIN=$OLD_BIN MODEL=$m BATCH=$b "$LD"/srv.sh "s18-$l-old" "${a[@]}" || return 1
  LLAMA=$NEW_LLAMA BIN=$NEW_BIN MODEL=$m BATCH=$b "$LD"/srv.sh "s18-$l-new" "${a[@]}" || return 1
  LLAMA=$OLD_LLAMA BIN=$OLD_BIN MODEL=$m BATCH=$b "$LD"/srv.sh "s18-$l-old-b" "${a[@]}"
}
for m in ${MODELS:-q38 q36}; do
  case $m in
    q38) abba q38 "$MD/qwen38-35b-a3b-distill/Qwen3.8-35B-A3B-IQ4_XS.gguf" 2048 2770 256 ;;
    q36) abba q36 "$MD/qwen36-35b-a3b/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf" 512 2900 128 ;;
  esac
done
