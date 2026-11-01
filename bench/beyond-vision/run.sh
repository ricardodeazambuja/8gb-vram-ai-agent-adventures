#!/usr/bin/env bash
#
# run.sh STEP — every experiment in "Can LFM3-VL do more than look?" (docs/fine-tuning.md).
# One GPU job at a time; each server runs in its own scope, with its log in models/beyond-vision-logs/.
# Needs the setup in docs/reproduce.md. Minutes are from the RTX 2070 Max-Q.
# bench/check_numbers.py reads the files these steps write, so `git diff` and that script show what moved.
#
#   weights        how far apart the two models are                      CPU, 1 min
#   replay         13 stuck points on VL (2 arms) and 1.7B               25 min
#   pibench-temp   VL at temperature 0.7, 6 runs (needs the sandbox)     1 h
#   graft          VL's projector on 1.7B, unchanged                     35 min
#   train-agent    fit test, then the agent LoRA from data/agent.json    30 min
#   score-agent    the LoRA on held-out stuck points and on pictures     25 min
#   pibench-agent  the same LoRA at temperature 0.7, 6 runs              1 h
#   eyes           1.7B driving, VL reporting, 60 scenes                 20 min
#   big-models     Groot and Gemma 4 26B on 30 scenes, thinking on       25 min
#   projector      map 1.7B's GGUF into VL's code, train, eval           1 h, 10 GB of disk
#   overfit        projector on 40 scenes, 30 passes (after projector)   15 min
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
R=bench/beyond-vision
A=$PWD/models/lfm3-vl-2b-adapters
L=$(realpath "${ADV_LLAMA:-../llama.cpp}")
LOGS=$PWD/models/beyond-vision-logs; mkdir -p "$LOGS"
PY="mamba run --no-capture-output -n vlft python"
VPY=${VLFT_ENV:-$HOME/miniforge3/envs/vlft}/bin/python   # the probes draw pictures, so they need its Pillow

serve() { # NAME PORT VAR=VALUE... COMMAND...
  local n=$1 p=$2; shift 2
  systemd-run --user --scope --quiet --unit="bv-srv-$n-$(date +%s)" -p MemoryMax=20G \
    env PORT="$p" "$@" >"$LOGS/srv-$n.log" 2>&1 &
  for _ in $(seq 150); do
    curl -sf "127.0.0.1:$p/health" >/dev/null && { sleep 2; return 0; }; sleep 2
  done
  echo "server $n did not start, see $LOGS/srv-$n.log"; stop; return 1
}
stop() { systemctl --user stop 'bv-srv-*.scope' 2>/dev/null; sleep 5; }
# A failed sandbox run must stop the step: its output is untrusted until sandbox.sh has scanned it.
sandbox() { # SCRIPT OUT ARGS...: the model repo's code runs only in here
  SCRIPT="$1" OUT="$2" bash bench/finetune/sandbox.sh "${@:3}"
}
LORA="--lora $A/agent-lora.gguf"
VL_PROJ=$PWD/models/lfm3-vl-2b/mmproj-LFM3-VL-2B-F16.gguf
TEXT=$A/proj
PROJ=/bench/beyond-vision/train_projector.py

case "${1:-}" in
weights)
  PYTHONPATH=$L/gguf-py $PY $R/weight_gap.py ;;
replay)
  serve vl 8080 QMODEL=lfm3vl bash scripts/run-qwen.sh || exit 1
  python3 $R/replay.py run lfm3vl-think --arms default,temp0.7; stop
  serve t 8080 QMODEL=lfm3 CTX=32768 bash scripts/run-qwen.sh || exit 1
  python3 $R/replay.py run lfm3-think --arms default --save "$LOGS/stuck.json"; stop ;;   # to compare with data/stuck.json
pibench-temp)
  RAW=$PWD/bench/results/temp07-doom MODELS=lfm3vl REPEATS=1 TIME_LIMIT=900 TEMP=0.7 bash bench/run-bench.sh
  python3 bench/steering/compare_doom.py bench/results/temp07-doom --limit 900 \
    --note "TEMP=0.7, no adapter" --out $R/results/pibench-temp07.json ;;
graft)
  serve g 8080 QMODEL=lfm3 CTX=32768 MMPROJ="$VL_PROJ" MMPROJ_GPU=1 bash scripts/run-qwen.sh --image || exit 1
  $VPY bench/visual-control/run_control.py glance graft --think off --n 30
  # thinking on is too slow to finish: the partial score is in the log
  timeout 1300 $VPY bench/visual-control/run_control.py glance graft --think on --n 12 | tee $R/results/logs/graft-on.txt; stop ;;
train-agent)
  sandbox /bench/beyond-vision/train_agent.py "$A/agent" --limit 16 || exit 1
  sandbox /bench/beyond-vision/train_agent.py "$A/agent" --epochs 4 --max-secs 2300 | tee $R/results/logs/tr-agent.txt || exit 1
  (cd "$L" && PYTHONPATH=gguf-py CUDA_VISIBLE_DEVICES= $PY convert_lora_to_gguf.py "$A/agent/agent-lora" \
    --base "$OLDPWD/models/lfm3-vl-2b-hf" --outfile "$A/agent-lora.gguf") ;;
score-agent)
  serve a 8080 QMODEL=lfm3vl MMPROJ_GPU=1 EXTRA_ARGS="$LORA" bash scripts/run-qwen.sh --image || exit 1
  python3 $R/replay.py run agent-lora --arms default,temp0.7
  $VPY bench/visual-control/run_control.py glance agent-lora --think off --n 30; stop ;;
pibench-agent)
  RAW=$PWD/bench/results/agent-doom MODELS=lfm3vl REPEATS=1 TIME_LIMIT=900 TEMP=0.7 EXTRA_ARGS="$LORA" bash bench/run-bench.sh
  python3 bench/steering/compare_doom.py bench/results/agent-doom --limit 900 \
    --note "TEMP=0.7 with agent LoRA" --out $R/results/pibench-agent-lora.json ;;
eyes)
  serve drv 8080 QMODEL=lfm3 CTX=16384 bash scripts/run-qwen.sh || exit 1
  serve eye 8081 QMODEL=lfm3vl CTX=8192 bash scripts/run-qwen.sh --image || exit 1
  nvidia-smi --query-gpu=memory.used --format=csv,noheader
  $VPY $R/eyes_probe.py 12; stop ;;
big-models)
  serve groot 8080 QMODEL=groot bash scripts/run-qwen.sh --image || exit 1
  nvidia-smi --query-gpu=memory.used --format=csv,noheader
  $VPY bench/visual-control/run_control.py glance groot --think on --n 6; stop
  serve gemma 8080 BACKEND=cuda CTX=32768 NCMOE=22 KVQUANT=q8_0 bash scripts/run-server.sh --image || exit 1
  nvidia-smi --query-gpu=memory.used --format=csv,noheader
  $VPY bench/visual-control/run_control.py glance gemma26b --think on --n 6; stop ;;
projector)
  mkdir -p "$TEXT"
  PYTHONPATH=$L/gguf-py $PY $R/gguf_to_state.py check
  PYTHONPATH=$L/gguf-py $PY $R/gguf_to_state.py export
  sandbox $PROJ "$TEXT" --limit 16 || exit 1
  sandbox $PROJ "$TEXT" --n-train 300 --max-secs 2700 || exit 1
  OUT=$TEXT bash bench/finetune/sandbox.sh eval --model /model --adapter /out/run --n 30 --key proj &&
    cp "$TEXT/eval-proj.json" $R/results/projector-eval.json && cp "$TEXT/run/train.json" $R/results/projector-train.json ;;
overfit)
  sandbox $PROJ "$TEXT" --n-train 8 --repeat 30 --lr 1e-3 --fresh --max-secs 1200 --out /out/over |
    tee $R/results/logs/over-proj.txt && cp "$TEXT/over/train.json" $R/results/projector-overfit.json ;;
*)
  sed -n '3,18p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 2 ;;
esac
