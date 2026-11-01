#!/usr/bin/env bash
P="mamba run --no-capture-output -n vlft python"; T="$P bench/imagegen/train_style.py"
bash loop/gpu.sh reg 1800 $P bench/imagegen/make_reg.py --word man --count 20; echo "reg rc=$?"
bash loop/gpu.sh tr-reg 3000 $T train --set subject --method lora --trigger zxq --reg reg-man --tag=-reg; echo "tr reg rc=$?"
bash loop/gpu.sh ev-reg 1200 $T eval --run subject-lora-reg --trigger zxq; echo "ev reg rc=$?"
for s in vangogh:vgword gris:grword; do
  bash loop/gpu.sh inv-${s%%:*} 3000 $P bench/imagegen/invert.py --set ${s%%:*} --word ${s##*:} --like painting; echo "inv $s rc=$?"
  bash loop/gpu.sh evw-${s%%:*} 900 $P bench/imagegen/eval_sdcli.py --run ${s%%:*}-word --trigger ${s##*:}; echo "ev $s rc=$?"
done
bash loop/gpu.sh tr-long 7000 $T train --set vangogh --method lora --trigger vgstyl --steps 3000 --tag=-3k; echo "long rc=$?"
bash loop/gpu.sh ev-long 1200 $T eval --run vangogh-lora-3k --trigger vgstyl; echo "ev long rc=$?"
echo finished $(date +%H:%M)
