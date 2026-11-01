#!/usr/bin/env bash
# Projector pilot: fit test, control scoring of the unchanged graft in this harness, then a run. Waits for queue12.
while pgrep -f "[q]ueue12.sh" >/dev/null; do sleep 20; done; sleep 10
export OUT=$PWD/models/lfm3-vl-2b-adapters/proj
S=/bench/beyond-vision/train_projector.py
SCRIPT=$S bash loop/gpu.sh fit-proj 300 bash bench/finetune/sandbox.sh --limit 16; echo "fit rc=$?"; grep -E "^(step|text|train)|Error" loop/logs/fit-proj.log | tail -5
grep -q "^step 2" loop/logs/fit-proj.log || exit 1
SCRIPT=$S bash loop/gpu.sh tr-proj 3300 bash bench/finetune/sandbox.sh --n-train 300 --max-secs 2700; echo "train rc=$?"; tail -2 loop/logs/tr-proj.log
bash loop/gpu.sh ev-proj 1000 bash bench/finetune/sandbox.sh eval --model /model --adapter /out/run --n 30 --key proj; echo "eval rc=$?"; grep "^==" loop/logs/ev-proj.log
echo finished
