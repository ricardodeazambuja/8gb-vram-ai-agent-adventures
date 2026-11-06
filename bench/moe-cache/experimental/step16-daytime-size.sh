#!/usr/bin/env bash
# What a busy daytime desktop costs in speed: each model's MOE_CACHE=auto size (night) against the size the launcher
# shrinks it to next to a ~1.7 GB desktop (cache - (1700 - desk)), run night / day / night for the machine's drift.
# Both sizes run tonight (desktop ~0.5 GB), so this is the cost of the smaller cache only, not of sharing the GPU.
# Flint's day size would be 968 MiB, under the launcher's 1024 floor (it refuses), so it runs at 1024.
# srvbench x2 (decode) + turnbench (48-600-token turns). Main build (moe-cache-max-batch 69ca796dd), memguard running.
Q="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ADV="$(cd "$Q/../../.." && pwd)"; LD="$Q/../longctx"; MD=$ADV/models
pgrep -f "[m]emguard.py" >/dev/null || { echo "memguard not running, refusing"; exit 1; }
export LOGDIR=${LOGDIR:-$Q/logs} RUNS=2 SIZES="48 100 200 600"
TB="${PY:-python3} -I $Q/../models/turnbench.py"
run() { # label model batch max-batch night day
  local n=$1 m=$2 b=$3 mb=$4
  for c in "$5" "$6" "$5"; do
    l=day-$n-c$c; [ -e "$LOGDIR/srv-$l.log" ] && l=$l-b
    MODEL=$m BATCH=$b AFTER="$TB" "$LD"/srv.sh "$l" -cmoe --moe-cache-mib "$c" --moe-cache-cpu-share 0.95 \
      --moe-cache-max-batch "$mb" || return 1
  done
}
for m in ${MODELS:-q36 q38 flint ornith}; do
  case $m in
    q36) run q36 "$MD/qwen36-35b-a3b/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf" 512 128 2900 1580 ;;
    q38) run q38 "$MD/qwen38-35b-a3b-distill/Qwen3.8-35B-A3B-IQ4_XS.gguf" 2048 256 2770 1770 ;;
    flint) run flint "$MD/flint-40b-a2.5b/flint-UD-Q3_K_XL.gguf" 2048 128 2168 1024 ;;
    ornith) run ornith "$MD/ornith-1.5-35b-a3b/Ornith-1.5-35B-Q4_K_M.gguf" 2048 32 2983 1713 ;;
  esac
done
