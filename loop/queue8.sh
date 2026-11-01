#!/usr/bin/env bash
serve() { n=$1; shift
  systemd-run --user --scope --quiet --unit="ov-srv-$n-$(date +%s)" -p MemoryMax=12G timeout 2400 env "$@" >"loop/logs/srv-$n.log" 2>&1 &
  for i in $(seq 90); do curl -sf 127.0.0.1:8080/health >/dev/null && { sleep 2; return 0; }; sleep 2; done; echo "no server $n"; return 1; }
serve g2 QMODEL=lfm3 CTX=32768 MMPROJ="$PWD/models/lfm3-vl-2b/mmproj-LFM3-VL-2B-F16.gguf" MMPROJ_GPU=1 bash scripts/run-qwen.sh --image || exit 1
timeout 900 python3 bench/visual-control/run_control.py glance graft --think off --n 30 >loop/logs/graft-off.log 2>&1; echo "off rc=$?"
timeout 1300 python3 bench/visual-control/run_control.py glance graft --think on --n 12 >loop/logs/graft-on.log 2>&1; echo "on rc=$?"
systemctl --user stop 'ov-srv-*.scope'; sleep 5
RAW=$PWD/bench/results/temp07-doom MODELS=lfm3vl REPEATS=1 TIME_LIMIT=900 TEMP=0.7 bash loop/gpu.sh pibench-t 6000 bash bench/run-bench.sh; echo "pibench rc=$?"
echo finished
