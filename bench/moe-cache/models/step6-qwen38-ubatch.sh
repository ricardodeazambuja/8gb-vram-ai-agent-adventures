#!/usr/bin/env bash
# Qwen3.8 cache: ubatch 512 vs 2048 vs 1024 vs 512 at the SAME VRAM (~7.2 GB loaded), share 0.95; srvbench x2 + 30K long-context
# measured base VRAM besides the cache (-cmoe, 128K q8_0): b512 3489, b1024 3677, b2048 4053 MiB -> cache 3661 / 3550 / 3174
Q="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ADV="$(cd "$Q/../../.." && pwd)"; L="$Q/../longctx"
export LOGDIR=${LOGDIR:-$Q/logs}
export MODEL=$ADV/models/qwen38-35b-a3b-distill/Qwen3.8-35B-A3B-IQ4_XS.gguf RUNS=2
LC="${PY:-python3} -I $L/longctx.py"
for bc in 512:3661 2048:3174 1024:3550 512:3661; do
  B=${bc%:*}; C=${bc#*:}; N=q38-b$B-cache$C; [ -e "$LOGDIR/srv-$N.log" ] && N=$N-b
  BATCH=$B AFTER="$LC" $L/srv.sh $N -cmoe --moe-cache-mib $C --moe-cache-cpu-share 0.95
done
