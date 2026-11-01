#!/usr/bin/env bash
# Replay again, this time with thinking in the history as real runs had it, then collect 1.7B's replies.
serve() { n=$1; shift
  systemd-run --user --scope --quiet --unit="ov-srv-$n-$(date +%s)" -p MemoryMax=12G timeout 2400 env "$@" >"loop/logs/srv-$n.log" 2>&1 &
  for i in $(seq 90); do curl -sf 127.0.0.1:8080/health >/dev/null && { sleep 2; return 0; }; sleep 2; done; echo "no server $n"; return 1; }
stop() { systemctl --user stop 'ov-srv-*.scope'; sleep 5; }
R=bench/beyond-vision
serve vl2 QMODEL=lfm3vl bash scripts/run-qwen.sh && timeout 1300 python3 $R/replay.py run lfm3vl-think --arms default,temp0.7; echo "vl rc=$?"; stop
serve t2 QMODEL=lfm3 CTX=32768 bash scripts/run-qwen.sh && timeout 500 python3 $R/replay.py run lfm3-think --arms default --save $R/data/stuck.json; echo "1.7b rc=$?"; stop
echo finished
