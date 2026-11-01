#!/usr/bin/env bash
# Big models that already do both, on the first 6 scenes of each kind, thinking on. One at a time.
while pgrep -f "[e]yes_probe.py" >/dev/null; do sleep 15; done; sleep 10
serve() { n=$1; shift
  systemd-run --user --scope --quiet --unit="ov-srv-$n-$(date +%s)" -p MemoryMax=14G timeout 3000 env "$@" >"loop/logs/srv-$n.log" 2>&1 &
  for i in $(seq 150); do curl -sf 127.0.0.1:8080/health >/dev/null && { sleep 2; return 0; }; sleep 2; done; echo "no server $n"; return 1; }
for m in gemma26b groot; do
  serve $m QMODEL=$m bash scripts/run-qwen.sh --image || { systemctl --user stop 'ov-srv-*.scope'; continue; }
  nvidia-smi --query-gpu=memory.used --format=csv,noheader
  timeout 2600 python3 bench/visual-control/run_control.py glance $m --think on --n 6 >loop/logs/glance-$m.log 2>&1; echo "$m rc=$?"
  systemctl --user stop 'ov-srv-*.scope'; sleep 8
done
echo finished
