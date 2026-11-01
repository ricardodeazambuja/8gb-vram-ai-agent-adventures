#!/usr/bin/env bash
P="mamba run --no-capture-output -n vlft python"
bash loop/gpu.sh stack 1800 $P bench/imagegen/stack.py; echo "stack rc=$?"
