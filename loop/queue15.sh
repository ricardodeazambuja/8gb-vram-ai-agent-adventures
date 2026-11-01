#!/usr/bin/env bash
# Can the projector carry picture information at all? Overfit 8 scenes of each kind, loss should fall towards 0.
while pgrep -f "[q]ueue14.sh" >/dev/null; do sleep 20; done; sleep 10
export OUT=$PWD/models/lfm3-vl-2b-adapters/proj
SCRIPT=/bench/beyond-vision/train_projector.py bash loop/gpu.sh over-proj 1500 bash bench/finetune/sandbox.sh --n-train 8 --repeat 30 --lr 1e-3 --fresh --max-secs 1200 --out /out/over; echo "over rc=$?"
grep "^step" loop/logs/over-proj.log | sed -n '1p;5p;10p;15p;20p;$p' | cut -c1-60
echo finished
