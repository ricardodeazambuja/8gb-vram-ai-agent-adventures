#!/usr/bin/env bash
P="mamba run --no-capture-output -n vlft python"; T="$P bench/imagegen/train_style.py"
bash loop/gpu.sh klein-ref 900 $P bench/imagegen/klein_person.py; echo "klein rc=$?"
bash loop/gpu.sh reg 1800 $P bench/imagegen/make_reg.py --word man --n 20; echo "reg rc=$?"
bash loop/gpu.sh tr-reg 3000 $T train --set subject --method lora --trigger zxq --reg reg-man --tag -reg --ckpts 250,500; echo "tr reg rc=$?"
bash loop/gpu.sh ev-reg 1200 $T eval --run subject-lora-reg --trigger zxq; echo "ev reg rc=$?"
bash loop/gpu.sh tr-regn 3000 $T train --set subject --method norm --trigger zxq --reg reg-man --tag -reg; echo "tr regn rc=$?"
bash loop/gpu.sh ev-regn 1200 $T eval --run subject-norm-reg --trigger zxq; echo "ev regn rc=$?"
bash loop/gpu.sh tr-dora2 3000 $T train --set subject --method dora --trigger zxq --tag -2; echo "dora rc=$?"
bash loop/gpu.sh ev-dora2 1200 $T eval --run subject-dora-2 --trigger zxq; echo "ev dora rc=$?"
bash loop/gpu.sh tr-long 7000 $T train --set vangogh --method lora --trigger vgstyl --steps 3000 --tag -3k; echo "long rc=$?"
bash loop/gpu.sh ev-long 1200 $T eval --run vangogh-lora-3k --trigger vgstyl; echo "ev long rc=$?"
echo finished $(date +%H:%M)
