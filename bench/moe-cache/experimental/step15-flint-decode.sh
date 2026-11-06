#!/usr/bin/env bash
# Step 14 left Flint's decode at --moe-cache-max-batch 128 below every run at 32 (29.3 vs 30.8-32.9 tok/s), with no
# mechanism for it. Settle it: 32 / 128 / 32 / 128, four 400-token srvbench runs each, nothing else. memguard running.
Q="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ADV="$(cd "$Q/../../.." && pwd)"; LD="$Q/../longctx"
pgrep -f "[m]emguard.py" >/dev/null || { echo "memguard not running, refusing"; exit 1; }
export LOGDIR=${LOGDIR:-$Q/logs} RUNS=4 BATCH=2048 MODEL=$ADV/models/flint-40b-a2.5b/flint-UD-Q3_K_XL.gguf
for l in a b; do
  for mb in 32 128; do
    "$LD"/srv.sh "dec-flint-mb$mb-$l" -cmoe --moe-cache-mib 2168 --moe-cache-cpu-share 0.95 --moe-cache-max-batch $mb || exit 1
  done
done
