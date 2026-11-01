#!/usr/bin/env bash
serve() { n=$1; p=$2; shift 2
  systemd-run --user --scope --quiet --unit="ov-srv-$n-$(date +%s)" -p MemoryMax=10G timeout 2400 env PORT=$p "$@" >"loop/logs/srv-$n.log" 2>&1 &
  for i in $(seq 90); do curl -sf 127.0.0.1:$p/health >/dev/null && { sleep 2; return 0; }; sleep 2; done; echo "no server $n"; return 1; }
serve drv 8080 QMODEL=lfm3 CTX=16384 bash scripts/run-qwen.sh || exit 1
serve eye 8081 QMODEL=lfm3vl CTX=8192 bash scripts/run-qwen.sh --image || { systemctl --user stop 'ov-srv-*.scope'; exit 1; }
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
timeout 2000 python3 bench/beyond-vision/eyes_probe.py 12 >loop/logs/eyes.log 2>&1; echo "eyes rc=$?"
nvidia-smi --query-gpu=memory.used --format=csv,noheader
systemctl --user stop 'ov-srv-*.scope'; echo finished
