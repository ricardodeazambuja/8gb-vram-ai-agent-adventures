#!/usr/bin/env bash
# hits.sh LABEL PROMPT_FILE [args...]: llama-completion on the templated long prompt, 600 tokens greedy; prints decode t/s and MoE cache hit rates
Q="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ADV="$(cd "$Q/../../.." && pwd)"
L=$1; P=$2; shift 2
LOGDIR=${LOGDIR:-$Q/logs}; mkdir -p "$LOGDIR"
cd "${LLAMA:-${ADV_LLAMA:-$ADV/../llama.cpp}}" || exit 1
BIN=${BIN_COMPLETION:-build/bin/llama-completion}
M=${MODEL:-$ADV/models/qwen36-35b-a3b/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf}
KV=${KV:-q8_0}
echo "### hits $L | $BIN | $(basename "$M") | $* | env NO_WARM=${LLAMA_MOE_CACHE_NO_WARM:-}"
systemd-run --user --scope -q --unit=moeh-$$ mamba run --no-capture-output -n llamacpp-cuda "$BIN" -m "$M" -ngl 99 -c ${CTX:-131072} -ctk $KV -ctv $KV -fa on -t 6 -b ${BATCH:-512} -ub ${BATCH:-512} "$@" -f "$P" -n 600 --ignore-eos --temp 0 -no-cnv < /dev/null > "$LOGDIR/hits-$L.log" 2>&1
grep -E 'prompt eval time|       eval time|hit rate|copied from the cache|kept in free slots|out of memory' "$LOGDIR/hits-$L.log" | sed -E 's/^.*(common_perf_print|llama_moe_cache): *//' | sed 's/^/    /'
