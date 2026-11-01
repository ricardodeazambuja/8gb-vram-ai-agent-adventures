#!/usr/bin/env bash
# After training: convert the LoRA, score replay and vision with it, then pibench.
while pgrep -f "[t]rain_agent.py" >/dev/null; do sleep 20; done
A=$PWD/models/lfm3-vl-2b-adapters; L=$PWD/../llama.cpp
ls $A/agent/agent-lora/adapter_model.safetensors || { echo "no adapter"; exit 1; }
(cd $L && PYTHONPATH=gguf-py CUDA_VISIBLE_DEVICES= mamba run -n vlft python convert_lora_to_gguf.py $A/agent/agent-lora --base $OLDPWD/models/lfm3-vl-2b-hf --outfile $A/agent-lora.gguf 2>&1 | tail -2)
[ -s $A/agent-lora.gguf ] || { echo "convert failed"; exit 1; }
serve() { n=$1; shift
  systemd-run --user --scope --quiet --unit="ov-srv-$n-$(date +%s)" -p MemoryMax=12G timeout 2400 env "$@" >"loop/logs/srv-$n.log" 2>&1 &
  for i in $(seq 90); do curl -sf 127.0.0.1:8080/health >/dev/null && { sleep 2; return 0; }; sleep 2; done; echo "no server $n"; return 1; }
serve a QMODEL=lfm3vl MMPROJ_GPU=1 EXTRA_ARGS="--lora $A/agent-lora.gguf" bash scripts/run-qwen.sh --image || exit 1
timeout 900 python3 bench/beyond-vision/replay.py run agent-lora --arms default,temp0.7; echo "replay rc=$?"
timeout 1000 python3 bench/visual-control/run_control.py glance agent-lora --think off --n 30 >loop/logs/agent-glance.log 2>&1; echo "glance rc=$?"
systemctl --user stop 'ov-srv-*.scope'; sleep 5
RAW=$PWD/bench/results/agent-doom MODELS=lfm3vl REPEATS=1 TIME_LIMIT=900 TEMP=0.7 EXTRA_ARGS="--lora $A/agent-lora.gguf" bash loop/gpu.sh pibench-a 6000 bash bench/run-bench.sh; echo "pibench rc=$?"
echo finished
