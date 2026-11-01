#!/usr/bin/env bash
# Gemma 4 26B on the same 30 glance scenes as Groot. Projector on the CPU, so slow. Waits for queue13.
while pgrep -f "[q]ueue13.sh" >/dev/null; do sleep 20; done; sleep 10
systemd-run --user --scope --quiet --unit="ov-srv-gemma-$(date +%s)" -p MemoryMax=20G timeout 3500 env BACKEND=cuda CTX=32768 NCMOE=22 KVQUANT=q8_0 bash scripts/run-server.sh --image >loop/logs/srv-gemma.log 2>&1 &
for i in $(seq 150); do curl -sf 127.0.0.1:8080/health >/dev/null && break; sleep 2; done
curl -sf 127.0.0.1:8080/health >/dev/null || { echo "no gemma"; exit 1; }
nvidia-smi --query-gpu=memory.used --format=csv,noheader
timeout 3300 python3 bench/visual-control/run_control.py glance gemma26b --think on --n 6 >loop/logs/glance-gemma26b.log 2>&1; echo "gemma rc=$?"
systemctl --user stop 'ov-srv-*.scope'; echo finished
