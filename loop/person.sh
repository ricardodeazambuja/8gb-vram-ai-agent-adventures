#!/usr/bin/env bash
# DoRA and LoRA + norm on the person, one at a time. Checkpoints at 250 and 500 on the combined run.
PY="mamba run --no-capture-output -n vlft python bench/imagegen/train_style.py"
for m in dora lora+norm; do
  bash loop/gpu.sh tr-$m 1800 $PY train --set subject --method $m --trigger zxq --ckpts 250,500
  echo "train $m rc=$?"
  bash loop/gpu.sh ev-$m 1200 $PY eval --run subject-$m --trigger zxq
  echo "eval $m rc=$?"
done
for c in 250 500; do bash loop/gpu.sh ev-c$c 1200 $PY eval --run subject-lora+norm --trigger zxq --ckpt $c; echo "ck $c rc=$?"; done
echo finished $(date +%H:%M)
