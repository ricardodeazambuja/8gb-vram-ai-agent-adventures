#!/usr/bin/env bash
# srv.sh LABEL [server args...]: start llama-server in its own scope on port 8090, run srvbench.py (RUNS times) and/or $AFTER, stop it.
# env: MODEL CTX BATCH KV(q8_0) RUNS(0) AFTER  LLAMA(llama.cpp dir) BIN(server path relative to LLAMA)  LOGDIR
#      any other env (e.g. LLAMA_MOE_CACHE_NO_WARM=1) is passed to the server
Q="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; S="$(dirname "$Q")"
ADV="$(cd "$Q/../../.." && pwd)"
L=$1; shift
LOGDIR=${LOGDIR:-$Q/logs}; mkdir -p "$LOGDIR"
cd "${LLAMA:-${ADV_LLAMA:-$ADV/../llama.cpp}}" || exit 1
BIN=${BIN:-build/bin/llama-server}
M=${MODEL:-$ADV/models/qwen36-35b-a3b/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf}
KV=${KV:-q8_0}
${PY:-python3} -I "$Q/memcheck.py" "$M" "${MARGIN:-4}" || { echo "    LOAD SKIPPED (RAM)"; exit 1; }
pgrep -f "[m]emguard.py" >/dev/null || echo "    WARNING: memguard.py not running"
echo "### $L | $(git log --oneline -1 | cut -c1-60) | $BIN | $(basename "$M") ctx=${CTX:-131072} batch=${BATCH:-512} kv=$KV | $* | env NO_WARM=${LLAMA_MOE_CACHE_NO_WARM:-}"
echo "    vram before: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
systemd-run --user --scope -q --unit=moeq-$$ mamba run --no-capture-output -n llamacpp-cuda "$BIN" -m "$M" -ngl 99 -c ${CTX:-131072} -ctk $KV -ctv $KV -fa on -t 6 -b ${BATCH:-512} -ub ${BATCH:-512} --parallel 1 --jinja --port 8090 --host 127.0.0.1 -lv ${LV:-4} "$@" > "$LOGDIR/srv-$L.log" 2>&1 &
ok=0; for _ in $(seq 400); do curl -sf 127.0.0.1:8090/health >/dev/null && { ok=1; break; }; sleep 1; done
if [ $ok = 1 ]; then
  VRAM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader); echo "    vram loaded: $VRAM"
  [ "${RUNS:-0}" -gt 0 ] && ${PY:-python3} -I "$S/srvbench.py" "$L" "$VRAM" "$RUNS"
  [ -n "$AFTER" ] && eval "$AFTER"
  echo "    vram end: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
else
  echo "    LOAD FAILED"
fi
systemctl --user stop moeq-$$.scope 2>/dev/null; pkill -f 'llama-server.*--port 8090'; sleep 4
grep -aE 'out of memory|failed to allocate|MoE cache size|llama_moe_cache' "$LOGDIR/srv-$L.log" | sed -E 's/^[0-9.]+ [IWE] //' | cut -c1-160 | head -8 | sed 's/^/    /'
