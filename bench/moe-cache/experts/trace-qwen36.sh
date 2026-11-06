#!/usr/bin/env bash
# Qwen3.6 expert trace on real agent work: replay its 18 pibench sessions (bench/results/raw/qwen36, which stay on
# this disk) as prompts on a cache run with LLAMA_MOE_CACHE_TRACE. --moe-cache-max-batch 512 sends every ubatch
# through the cache's lookup, so prompt tokens are traced too (the cache size doesn't change routing).
#   TRACE=/path/qwen36-agent.trace bash experts/trace-qwen36.sh   then: python3 experts/agent_experts.py ...
E="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; D="$E/.."; ADV="$(cd "$D/../.." && pwd)"
pgrep -f "[m]emguard.py" >/dev/null || { echo "memguard not running, refusing"; exit 1; }
export LOGDIR=${LOGDIR:-$E/logs} LLAMA_MOE_CACHE_TRACE=${TRACE:?set TRACE to the trace file to write}
export MODEL=$ADV/models/qwen36-35b-a3b/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf BATCH=512
export PI_CAPTURE=$ADV/bench/gemma12-probe/capture/agentic12/0001-req.json
mapfile -t S < <(cd "$ADV" && ls bench/results/raw/qwen36/*/r*/pi.jsonl)
AFTER="cd $ADV && ${PY:-python3} -I $D/replay_sessions.py $E/logs/sessions-qwen36.json ${S[*]}" \
  "$D"/longctx/srv.sh trace-qwen36 -cmoe --moe-cache-mib 2700 --moe-cache-cpu-share 0.95 --moe-cache-max-batch 512
