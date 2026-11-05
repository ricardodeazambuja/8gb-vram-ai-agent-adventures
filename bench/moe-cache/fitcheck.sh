#!/usr/bin/env bash
# fitcheck.sh NAME LAUNCHER... (run from anywhere; logs to $OUT, default bench/moe-cache/logs) : start a launcher on port 8090 in its own scope, send a ~25K-token prompt + 200 tokens, report VRAM and result
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
S="${OUT:-bench/moe-cache/logs}"
N=$1; shift
systemd-run --user --scope -q --unit="fit-$N" env PORT=8090 "$@" > "$S/fit-$N.log" 2>&1 &
ok=0; for _ in $(seq 240); do curl -sf 127.0.0.1:8090/health >/dev/null && { ok=1; break; }; sleep 2; done
echo "$N healthy=$ok vram_after_load=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
if [ $ok = 1 ]; then
python3 - <<'PY'
import json, urllib.request, glob
txt = "".join(open(f).read() for f in sorted(glob.glob("docs/*.md")))[:100000]
body = {"messages": [{"role": "user", "content": txt + "\n\nSummarize the above in 3 sentences."}], "max_tokens": 200, "temperature": 0}
r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8090/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=1200))
t = r["timings"]; print(f"prompt {t['prompt_n']} tok {t['prompt_per_second']:.0f} t/s, gen {t['predicted_n']} tok {t['predicted_per_second']:.1f} t/s")
PY
echo "vram_after_prompt=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
fi
grep -iE "out of memory|error|moe.cache|hit rate" "$S/fit-$N.log" | grep -v "^$" | tail -5
systemctl --user stop "fit-$N.scope" 2>/dev/null; sleep 5
