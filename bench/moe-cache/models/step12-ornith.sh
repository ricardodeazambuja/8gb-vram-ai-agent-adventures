#!/usr/bin/env bash
# Ornith-1.5-35B Q4_K_M (20.2 GiB): -ncmoe 34 vs the cache at the same VRAM (2983 MiB, share 0.95) vs -ncmoe 34 again;
# batch 2048, 128K, q8_0 KV; srvbench x2 + the 30K long-prompt check for the first two. Main build (moe-offload 3a72c57be).
# Step 3: -ncmoe 34 loaded 7219 MiB; cache 3486 loaded 7722 (CUDA OOM on the first request) -> cache base ~4236 MiB -> 2983.
Q="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ADV="$(cd "$Q/../../.." && pwd)"; L="$Q/../longctx"
export LOGDIR=${LOGDIR:-$Q/logs}
export MODEL=$ADV/models/ornith-1.5-35b-a3b/Ornith-1.5-35B-Q4_K_M.gguf BATCH=2048 RUNS=2
LC="${PY:-python3} -I $L/longctx.py"
AFTER="$LC" "$L"/srv.sh ornith2-ncmoe34 -ncmoe 34
AFTER="$LC" "$L"/srv.sh ornith2-cache2983 -cmoe --moe-cache-mib 2983 --moe-cache-cpu-share 0.95
"$L"/srv.sh ornith2-ncmoe34-b -ncmoe 34
