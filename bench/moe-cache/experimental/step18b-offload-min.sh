#!/usr/bin/env bash
# Step 18b (2026-11-07 night): step 18's merged build lost the CPU part for batches of 32..max-batch tokens. Since
# 1f17b9c97 the cache gives no CPU map for a batch a device would offload, and CUDA offloads MUL_MAT_ID from host
# weights at >= GGML_OP_OFFLOAD_MIN_BATCH tokens (default 32). Rerun the merged build with that threshold at
# max-batch + 1 (batches above max-batch are still offloaded as before), then the old build once more for drift.
# Same configs and benches as step18-merged.sh.
Q="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ADV="$(cd "$Q/../../.." && pwd)"; LD="$Q/../longctx"; T="$Q/../models"; MD=$ADV/models
pgrep -f "[m]emguard.py" >/dev/null || { echo "memguard not running, refusing"; exit 1; }
: "${NEW_LLAMA:?set NEW_LLAMA (llama.cpp dir of the merged build)}" "${NEW_BIN:?set NEW_BIN (server path relative to NEW_LLAMA)}"
OLD_LLAMA=${OLD_LLAMA:-$ADV/../llama.cpp} OLD_BIN=${OLD_BIN:-build/bin/llama-server}
export LOGDIR=${LOGDIR:-$Q/logs} RUNS=2 SIZES="48 100 200 600" AFTER="${PY:-python3} -I $T/turnbench.py"
pair() { # label model batch cache maxbatch
  local l=$1 m=$2 b=$3 a=(-cmoe --moe-cache-mib "$4" --moe-cache-cpu-share 0.95 --moe-cache-max-batch "$5")
  echo "    GGML_OP_OFFLOAD_MIN_BATCH=$(($5 + 1)) for the merged build"
  GGML_OP_OFFLOAD_MIN_BATCH=$(($5 + 1)) LLAMA=$NEW_LLAMA BIN=$NEW_BIN MODEL=$m BATCH=$b "$LD"/srv.sh "s18b-$l-new-omb" "${a[@]}" || return 1
  LLAMA=$OLD_LLAMA BIN=$OLD_BIN MODEL=$m BATCH=$b "$LD"/srv.sh "s18b-$l-old" "${a[@]}"
}
for m in ${MODELS:-q38 q36}; do
  case $m in
    q38) pair q38 "$MD/qwen38-35b-a3b-distill/Qwen3.8-35B-A3B-IQ4_XS.gguf" 2048 2770 256 ;;
    q36) pair q36 "$MD/qwen36-35b-a3b/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf" 512 2900 128 ;;
  esac
done
