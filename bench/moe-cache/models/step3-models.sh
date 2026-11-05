#!/usr/bin/env bash
# step3-models.sh NAME...: per model: moe-cache-tune (launcher flags), old/new/old srvbench at the matched cache size (share 0.95), long-prompt check for old and new
# names: groot distill bolt ornith lfm25
# PY_TUNE: a python with numpy (the llamacpp-cuda env). ADV_LLAMA: the fork checkout (moe-offload build).
# WARNING: moe-cache-tune.py reads the whole GGUF into the page cache. On 2026-11-05 that, with distill,
# Bolt and Ornith back to back, made systemd-oomd kill the terminal. Run the tuner only on models < 16 GB,
# and evict the previous model's pages first.
Q="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ADV="$(cd "$Q/../../.." && pwd)"; L="$Q/../longctx"; PYT=${PY_TUNE:-python3}; M=$ADV/models
LLAMA=${ADV_LLAMA:-$ADV/../llama.cpp}
model() { case $1 in
  groot)   echo "$M/groot-19b-a3b/Groot-19B-A3B-261023-Q4_K_M.gguf 2048 27 3321" ;;
  distill) echo "$M/qwen38-35b-a3b-distill/Qwen3.8-35B-A3B-IQ4_XS.gguf 512 34 3264" ;;
  ornith)  echo "$M/ornith-1.5-35b-a3b/Ornith-1.5-35B-Q4_K_M.gguf 2048 34 3486" ;;
  lfm25)   echo "$M/lfm2.5-24b-a2b/LFM2.5-24B-A2B-QAD-Q4_0.gguf 2048 27 4212" ;;
  bolt)    echo "$M/nemotron-bolt-74b-a9b/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf 2048 - -" ;;
esac; }
LC="${PY:-python3} -I $L/longctx.py"
for n in "$@"; do
  read -r MP B NC SZ <<< "$(model $n)"
  echo "======== $n: $(basename $MP) batch=$B ncmoe=$NC matched=$SZ MiB  $(date +%H:%M)"
  echo "    vram before tune: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
  (cd $LLAMA && systemd-run --user --scope -q --unit=moetune-$n $PYT scripts/moe-cache-tune.py --out $Q/tune/out-$n -- build/bin/llama-server -m $MP -ngl 99 -cmoe -c 131072 -ctk q8_0 -ctv q8_0 -fa on --parallel 1 -b $B -ub $B -t 6 --jinja > $Q/tune/$n.txt 2>&1)
  grep -E 'cache up to|^recommended' $Q/tune/$n.txt | sed 's/^/    /'
  CEIL=$(grep -oE 'cache up to [0-9]+' $Q/tune/$n.txt | grep -oE '[0-9]+$')
  [ "$SZ" = - ] && SZ=$CEIL
  [ -z "$SZ" ] && { echo "    no cache size, skipping A/B"; continue; }
  export MODEL=$MP BATCH=$B RUNS=2
  OLD=(-cmoe); [ "$NC" != - ] && OLD=(-ncmoe "$NC")
  AFTER="$LC" $L/srv.sh $n-old "${OLD[@]}"
  out=$(AFTER="$LC" $L/srv.sh $n-new -cmoe --moe-cache-mib $SZ --moe-cache-cpu-share 0.95); echo "$out"
  if grep -q 'LOAD FAILED\|out of memory' <<< "$out" && [ -n "$CEIL" ] && [ "$CEIL" -lt "$SZ" ]; then
    echo "    matched size $SZ did not fit, retrying at the tuner ceiling $CEIL"
    AFTER="$LC" $L/srv.sh $n-new-ceil -cmoe --moe-cache-mib $CEIL --moe-cache-cpu-share 0.95
  fi
  $L/srv.sh $n-old-b "${OLD[@]}"
  unset MODEL BATCH RUNS
done
