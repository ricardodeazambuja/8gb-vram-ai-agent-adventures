#!/usr/bin/env bash
# --moe-cache-max-batch 32 / 128 / 256 / 32 again on Qwen3.6 and Flint: is 128 the per-model pick that keeps the
# 48-100-token gain without step 11's loss at 200? Main build (moe-cache-max-batch 69ca796dd), memguard running.
# Qwen3.6's cache is 2800 MiB, not 2900: ~460 MiB of desktop VRAM in use, and 2900 leaves room for ~380.
Q="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ADV="$(cd "$Q/../../.." && pwd)"; LD="$Q/../longctx"; T="$Q/../models"; MD=$ADV/models
pgrep -f "[m]emguard.py" >/dev/null || { echo "memguard not running, refusing"; exit 1; }
export LOGDIR=${LOGDIR:-$Q/logs} RUNS=2 SIZES="16 48 100 128 150 200 300 400 600"
TB="${PY:-python3} -I $T/turnbench.py"
run() { # label model batch cache
  for mb in 32 128 256 32; do
    l=$1-mb$mb; [ -e "$LOGDIR/srv-$l.log" ] && l=$l-b
    MODEL=$2 BATCH=$3 AFTER="$TB" "$LD"/srv.sh "$l" -cmoe --moe-cache-mib "$4" --moe-cache-cpu-share 0.95 --moe-cache-max-batch $mb || return 1
  done
}
run mb-q36 "$MD/qwen36-35b-a3b/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf" 512 2800
run mb-flint "$MD/flint-40b-a2.5b/flint-UD-Q3_K_XL.gguf" 2048 2168
