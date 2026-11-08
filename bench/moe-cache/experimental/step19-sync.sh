#!/usr/bin/env bash
# Step 19 (2026-11-08): the full upstream sync (fork branch adventures-sync-2026-11-08, 1f6e660c4: upstream master
# cf34d3113 merged into 3f2b2d60a) against the then-default build 3f2b2d60a (tag known-good-2026-11-08-merged), at the launcher's MOE_CACHE=auto
# settings with GGML_OP_OFFLOAD_MIN_BATCH = limit + 1 on both. A/B/A on each model: srvbench x2 (decode),
# turnbench (48-600-token turns), then a 6-turn pi-like agent loop (fake tool results, nothing runs) that prints
# the prompt tokens reprocessed per turn: the sync replaced the fork's server checkpoint reuse with upstream's,
# and the hybrids (Qwen3.6/3.8 here) depend on checkpoints between turns.
Q="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ADV="$(cd "$Q/../../.." && pwd)"; LD="$Q/../longctx"; T="$Q/../models"; MD=$ADV/models
pgrep -f "[m]emguard.py" >/dev/null || { echo "memguard not running, refusing"; exit 1; }
: "${NEW_LLAMA:?set NEW_LLAMA (llama.cpp dir of the sync build)}" "${NEW_BIN:?set NEW_BIN (server path relative to NEW_LLAMA)}"
: "${OLD_LLAMA:?set OLD_LLAMA: a llama.cpp checkout at tag known-good-2026-11-08-merged, built (srv.sh labels runs with its commit)}"
OLD_BIN=${OLD_BIN:-build/bin/llama-server}
PY="${PY:-python3} -I"
export LOGDIR=${LOGDIR:-$Q/logs} RUNS=2 SIZES="48 100 200 600"
export AFTER="$PY $T/turnbench.py; echo '-- agent loop'; $PY $Q/../agentloop.py http://127.0.0.1:8090 6"
aba() { # label model batch cache maxbatch
  local l=$1 m=$2 b=$3 a=(-cmoe --moe-cache-mib "$4" --moe-cache-cpu-share 0.95 --moe-cache-max-batch "$5")
  export GGML_OP_OFFLOAD_MIN_BATCH=$(($5 + 1))
  LLAMA=$OLD_LLAMA BIN=$OLD_BIN MODEL=$m BATCH=$b "$LD"/srv.sh "s19-$l-merged" "${a[@]}" || return 1
  LLAMA=$NEW_LLAMA BIN=$NEW_BIN MODEL=$m BATCH=$b "$LD"/srv.sh "s19-$l-sync" "${a[@]}" || return 1
  LLAMA=$OLD_LLAMA BIN=$OLD_BIN MODEL=$m BATCH=$b "$LD"/srv.sh "s19-$l-merged-b" "${a[@]}"
}
for m in ${MODELS:-q38 q36}; do
  case $m in
    q38) aba q38 "$MD/qwen38-35b-a3b-distill/Qwen3.8-35B-A3B-IQ4_XS.gguf" 2048 2770 256 ;;
    q36) aba q36 "$MD/qwen36-35b-a3b/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf" 512 2900 128 ;;
  esac
done
