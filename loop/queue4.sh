#!/usr/bin/env bash
# Doom-loop vector on pibench: one repeat of each task, strength picked from the earlier sweep's safe range.
RES=bench/results/steering-doom; mkdir -p $RES
RAW="$PWD/$RES" MODELS=lfm3vl REPEATS=1 TIME_LIMIT=900 \
EXTRA_ARGS="--control-vector-scaled models/steering/retry-lfm3vl.gguf:0.03" \
bash loop/gpu.sh pibench 6000 bash bench/run-bench.sh; echo "pibench rc=$?"
echo finished $(date +%H:%M)
