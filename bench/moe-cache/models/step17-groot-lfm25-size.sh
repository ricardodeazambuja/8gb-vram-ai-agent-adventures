#!/usr/bin/env bash
# Sizes for MOE_CACHE=auto on Groot-19B and LFM2.5-24B (2026-11-07, night, desktop ~0.5 GB): each model's launcher
# split (-ncmoe 27, batch 2048) against the cache at step 3's size and 500 MiB smaller, then the split again for drift.
# srvbench x2 (decode) + turnbench (48-600-token turns) on each; srv.sh samples VRAM every second and prints the peak.
# The auto size comes from the rule in run-qwen.sh: server peak + desktop + 300 <= 7687 MiB.
# Main build (moe-cache-max-batch, functionally 69ca796dd), the binary's default cache-path batch limit (32).
Q="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ADV="$(cd "$Q/../../.." && pwd)"; L="$Q/../longctx"; MD=$ADV/models
pgrep -f "[m]emguard.py" >/dev/null || { echo "memguard not running, refusing"; exit 1; }
export LOGDIR=${LOGDIR:-$Q/logs} BATCH=2048 RUNS=2 SIZES="48 100 200 600"
export AFTER="${PY:-python3} -I $Q/turnbench.py"
run() { # name model big small
  local n=$1 m=$2
  MODEL=$m "$L"/srv.sh "s17-$n-ncmoe27" -ncmoe 27 || return 1
  for c in "$3" "$4"; do
    MODEL=$m "$L"/srv.sh "s17-$n-c$c" -cmoe --moe-cache-mib "$c" --moe-cache-cpu-share 0.95 || return 1
  done
  MODEL=$m "$L"/srv.sh "s17-$n-ncmoe27-b" -ncmoe 27
}
for m in ${MODELS:-groot lfm25}; do
  case $m in
    groot) run groot "$MD/groot-19b-a3b/Groot-19B-A3B-261023-Q4_K_M.gguf" 3321 2821 ;;
    lfm25) run lfm25 "$MD/lfm2.5-24b-a2b/LFM2.5-24B-A2B-QAD-Q4_0.gguf" 4212 3712 ;;
  esac
done
