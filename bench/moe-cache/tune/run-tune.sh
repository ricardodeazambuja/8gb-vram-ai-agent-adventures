#!/usr/bin/env bash
# runs scripts/moe-cache-tune.py for Qwen3.6, Gemma 4 26B and Flint with the launcher flags
T=/tmp/claude-1000/-home-ricardodeazambuja-backup-GitStuff-llama-cpp/4aefea9c-6ebe-4f38-bd47-5d9195dd716b/scratchpad/tune
M=$HOME/GitStuff/8gb-vram-ai-agent-adventures/models
cd /home/ricardodeazambuja/backup/GitStuff/llama.cpp
B=build/bin/llama-server
PY=/home/ricardodeazambuja/miniforge3/envs/llamacpp-cuda/bin/python
run() { n=$1; shift; echo "=== $n: $*" ; $PY scripts/moe-cache-tune.py --out $T/out-$n -- "$@" 2>&1 | tee $T/$n.txt; }
run qwen36 $B -m $M/qwen36-35b-a3b/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf -ngl 99 -cmoe -c 131072 -ctk q8_0 -ctv q8_0 -fa on --parallel 1 -b 512 -ub 512 -t 6 --jinja
run gemma26b $B -m $M/gemma4-26b-a4b-qat/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf -ngl 99 -cmoe --load-mode none -c 32768 -fa on -t 6 --jinja
run flint $B -m $M/flint-40b-a2.5b/flint-UD-Q3_K_XL.gguf -ngl 99 -cmoe -c 131072 -ctk q8_0 -ctv q8_0 -fa on --parallel 1 -b 2048 -ub 2048 -t 6 --jinja
