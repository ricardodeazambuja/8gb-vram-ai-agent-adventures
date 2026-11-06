#!/usr/bin/env bash
# Step 16 follow-up: at Qwen3.6's daytime VRAM, is the shrunk cache (1580 MiB) still better than static offload?
# --n-cpu-moe 36 leaves 1572 MiB of experts on the GPU (gpu_exp_bytes.py; NCMOE=32's 3070 MiB matched the 2900 cache),
# so the static run gets slightly less VRAM than the cache. static / cache / static, same benches as step 16.
Q="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ADV="$(cd "$Q/../../.." && pwd)"; LD="$Q/../longctx"
pgrep -f "[m]emguard.py" >/dev/null || { echo "memguard not running, refusing"; exit 1; }
export LOGDIR=${LOGDIR:-$Q/logs} RUNS=2 SIZES="48 100 200 600" BATCH=512
export MODEL=$ADV/models/qwen36-35b-a3b/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
TB="${PY:-python3} -I $Q/../models/turnbench.py"
AFTER="$TB" "$LD"/srv.sh day-q36-ncmoe36 -ncmoe 36 || exit 1
AFTER="$TB" "$LD"/srv.sh day-q36-c1580-c -cmoe --moe-cache-mib 1580 --moe-cache-cpu-share 0.95 --moe-cache-max-batch 128 || exit 1
AFTER="$TB" "$LD"/srv.sh day-q36-ncmoe36-b -ncmoe 36
