#!/usr/bin/env bash
# Does frequency pruning of Qwen3.6's experts hurt? Rank experts on five pibench tasks, prune, and compare each
# pruned model's next-token distributions with the full model's on the sixth task (llama-perplexity KL divergence).
#   WORK=/big/scratch/dir bash experts/prune-eval.sh      env: HELD (feature-inventory), KEEPS ("230 192 128";
#   "256" = the writer control, "230r" = 230 kept at random)
# Steps: 1. render the 18 sessions to text (server /apply-template); 2. trace the ranking text with 16-token batches,
# so layers 34/38/39 (fewer cache slots, see README) are traced too; 3. make_rank.py; 4. the full model's logits on
# the held-out text; 5. per KEEP: prune_gguf.py into models/qwen36-pruned/, KL divergence, delete it.
# Needs memguard running; WORK needs ~6 GB for the logits plus the 217 MB trace.
set -o pipefail
E="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; D="$E/.."; ADV="$(cd "$D/../.." && pwd)"
LL=${ADV_LLAMA:-$ADV/../llama.cpp}; PPL=$LL/build/bin/llama-perplexity; PY="${PY:-python3} -I"
W=${WORK:?set WORK}; HELD=${HELD:-feature-inventory}; KEEPS=${KEEPS:-"230 192 128"}
M=$ADV/models/qwen36-35b-a3b/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf; PD=$ADV/models/qwen36-pruned; L=$E/logs
pgrep -f "[m]emguard.py" >/dev/null || { echo "memguard not running, refusing"; exit 1; }
mkdir -p "$W/text" "$PD"
CACHE=(-ngl 99 -cmoe --moe-cache-mib 2800 --moe-cache-cpu-share 0.95 -fa on -t 6)
ppl() { # label model args...: llama-perplexity in its own scope (memguard stops moe*.scope)
  local label=$1 model=$2; shift 2
  $PY "$D/longctx/memcheck.py" "$model" 4 || return 1
  echo "### $label: $(basename "$model") $*"
  systemd-run --user --scope -q --unit="moe-ppl-$label-$$" mamba run --no-capture-output -n llamacpp-cuda \
    "$PPL" -m "$model" "${CACHE[@]}" "$@" > "$L/ppl-$label.log" 2>&1
  local rc=$?
  grep -aE "^Final estimate|Mean +KLD|Same top p|RMS Δp|llama_moe_cache: ubatch" "$L/ppl-$label.log" | sed 's/^/    /'
  return $rc
}

# 1. text of every session, and the two inputs: ranking text (run 1 of each other task) and the held-out task
mapfile -t S < <(cd "$ADV" && ls bench/results/raw/qwen36/*/r*/pi.jsonl)
[ -s "$W/text/$HELD-r1.txt" ] || PI_CAPTURE=$ADV/bench/gemma12-probe/capture/agentic12/0001-req.json CTX=8192 LOGDIR=$L MODEL=$M \
  AFTER="cd $ADV && RENDER_DIR=$W/text $PY $D/replay_sessions.py /dev/null ${S[*]}" "$D"/longctx/srv.sh render-qwen36 -cmoe || exit 1
cat "$W"/text/$HELD-r*.txt > "$W/held.txt"
for f in "$W"/text/*-r1.txt; do case $f in *"/$HELD-r1.txt") ;; *) cat "$f";; esac; done > "$W/rank.txt"
wc -c "$W/held.txt" "$W/rank.txt"

# 2. + 3. the ranking: replay counts, with the short layers from a 16-token-batch trace of the ranking text
[ -s "$W/rank-ub16.trace" ] || LLAMA_MOE_CACHE_TRACE=$W/rank-ub16.trace ppl rank-ub16 "$M" -f "$W/rank.txt" -c 4096 -b 512 -ub 16 \
  --moe-cache-max-batch 16 --chunks 10 || exit 1
$PY "$E/make_rank.py" "$L/qwen36-agent-counts.npz" "$HELD" "$L/rank-no-$HELD.npy" "$W/rank-ub16.trace" | tee "$L/rank-no-$HELD.txt" || exit 1

# 4. + 5. the full model's logits on the held-out text, then each pruned model against them
[ -s "$W/base.kld" ] || ppl base "$M" -f "$W/held.txt" -c 4096 --chunks 5 --kl-divergence-base "$W/base.kld" || exit 1
# KEEP 256 is the writer's control (nothing dropped: KLD must be ~0); KEEP NNNr ranks at random (seed 0)
$PY -c "import numpy as np; np.save('$W/rank-random.npy', np.random.default_rng(0).random((40, 256)))"
for K in $KEEPS; do
  R=$L/rank-no-$HELD.npy; case $K in *r) R=$W/rank-random.npy;; esac
  P=$PD/Qwen3.6-35B-A3B-UD-IQ4_XS-keep$K.gguf
  systemd-run --user --scope -q --unit="moe-prune-$K-$$" env GGUF_PY="$LL/gguf-py" ${PY} "$E/prune_gguf.py" \
    "$M" "$P" "$R" "${K%r}" || exit 1
  ppl "keep$K" "$P" -f "$W/held.txt" -c 4096 --chunks 5 --kl-divergence-base "$W/base.kld" --kl-divergence
  ls -la "$P"; rm -f "$P"
done
