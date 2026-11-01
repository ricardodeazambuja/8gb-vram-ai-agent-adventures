#!/usr/bin/env bash
P="mamba run --no-capture-output -n vlft python"
bash loop/gpu.sh inv-fit 300 $P bench/imagegen/invert.py --set subject --word zxqtok --fit; echo "fit rc=$?"
grep -E "^done|Error" loop/logs/inv-fit.log | cut -c1-150
if grep -q "^done" loop/logs/inv-fit.log; then
  bash loop/gpu.sh inv 3000 $P bench/imagegen/invert.py --set subject --word zxqtok; echo "inv rc=$?"
  bash loop/gpu.sh ev-word 900 $P bench/imagegen/eval_sdcli.py --run subject-word --trigger zxqtok; echo "ev word rc=$?"
fi
for r in subject-lora subject-norm; do bash loop/gpu.sh ev-$r 900 $P bench/imagegen/eval_sdcli.py --run $r --trigger zxq; echo "ev $r rc=$?"; done
bash loop/gpu.sh sweep2 1800 python3 bench/steering/sweep.py --model models/lfm3-1.7b/LFM3-1.7B-Q8_0.gguf --vector models/steering/concise-lfm3.gguf --name concise-lfm3-fine --scales=-0.3,-0.2,-0.1,-0.05,-0.02,0,0.02,0.05,0.1,0.2,0.3 --ngl 99; echo "sweep rc=$?"
echo finished $(date +%H:%M)
