#!/usr/bin/env bash
# Replay test on VL and 1.7B, then the graft: 1.7B text weights with VL's projector.
serve() { # NAME, env..., ending with args for run-qwen
  n=$1; shift
  systemd-run --user --scope --quiet --unit="ov-srv-$n-$(date +%s)" -p MemoryMax=12G timeout 3000 env "$@" >"loop/logs/srv-$n.log" 2>&1 &
  for i in $(seq 90); do curl -sf 127.0.0.1:8080/health >/dev/null && { sleep 2; return 0; }; sleep 2; done
  echo "server $n did not start"; return 1
}
stop() { systemctl --user stop 'ov-srv-*.scope'; sleep 5; }
V=models/lfm3-vl-2b/mmproj-LFM3-VL-2B-F16.gguf
serve vl QMODEL=lfm3vl bash scripts/run-qwen.sh && timeout 1500 python3 bench/beyond-vision/replay.py run lfm3vl; echo "replay vl rc=$?"; stop
serve t QMODEL=lfm3 CTX=32768 bash scripts/run-qwen.sh && timeout 600 python3 bench/beyond-vision/replay.py run lfm3 --arms default; echo "replay 1.7b rc=$?"; stop
serve g QMODEL=lfm3 CTX=32768 MMPROJ="$PWD/$V" MMPROJ_GPU=1 bash scripts/run-qwen.sh --image || exit 1
timeout 300 python3 bench/visual-control/run_control.py glance graft --think on --n 30; echo "graft on rc=$?"
timeout 300 python3 bench/visual-control/run_control.py glance graft --think off --n 30; echo "graft off rc=$?"
stop
echo finished
