#!/usr/bin/env bash
# Qwen3.8-35B-A3B distill: best old split vs best cache at the SAME VRAM (old/new/new-s0.8/old), 6K-prompt srvbench x2 + 30K long-context.
# VRAM model (measured): -ncmoe N loads ~5926 + 408*(34-N) MiB; -cmoe + cache C loads ~3489 + C MiB  -> ncmoe 31 ~ cache 3661
Q="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ADV="$(cd "$Q/../../.." && pwd)"; L="$Q/../longctx"
export LOGDIR=${LOGDIR:-$Q/logs}
export MODEL=$ADV/models/qwen38-35b-a3b-distill/Qwen3.8-35B-A3B-IQ4_XS.gguf BATCH=${BATCH:-512} RUNS=2
NC=${NC:-31}; C=${C:-3661}
LC="${PY:-python3} -I $L/longctx.py"
AFTER="$LC" $L/srv.sh q38-ncmoe$NC -ncmoe $NC
AFTER="$LC" $L/srv.sh q38-cache$C-s0.95 -cmoe --moe-cache-mib $C --moe-cache-cpu-share 0.95
$L/srv.sh q38-cache$C-s0.8 -cmoe --moe-cache-mib $C --moe-cache-cpu-share 0.8
$L/srv.sh q38-ncmoe$NC-b -ncmoe $NC
