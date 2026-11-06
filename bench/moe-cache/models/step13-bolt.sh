#!/usr/bin/env bash
# Nemotron-3.5-Bolt-74B-A9B (20.5 GiB, text only, no mmproj): -cmoe (the launcher's placement) vs -cmoe + a 600 MiB
# cache, share 0.95, vs -cmoe again; batch 2048, 128K, q8_0 KV; srvbench x2 + the 30K long-prompt check for the first two.
# Needs memguard running (longctx/memguard.py). Run 2026-11-05 with MARGIN=3 (user's OK): 23.7 GiB available in the daytime.
Q="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ADV="$(cd "$Q/../../.." && pwd)"; L="$Q/../longctx"
pgrep -f "[m]emguard.py" >/dev/null || { echo "memguard not running, refusing"; exit 1; }
export LOGDIR=${LOGDIR:-$Q/logs}
export MODEL=$ADV/models/nemotron-bolt-74b-a9b/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf BATCH=2048 RUNS=2
LC="${PY:-python3} -I $L/longctx.py"
AFTER="$LC" "$L"/srv.sh bolt-cmoe -cmoe || exit 1
AFTER="$LC" "$L"/srv.sh bolt-cache600 -cmoe --moe-cache-mib 600 --moe-cache-cpu-share 0.95
"$L"/srv.sh bolt-cmoe-b -cmoe
