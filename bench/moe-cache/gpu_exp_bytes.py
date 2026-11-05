import collections
import os
import re
import sys

sys.path.insert(0, os.path.join(os.environ.get('ADV_LLAMA', os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../llama.cpp')), 'gguf-py'))  # our fork's gguf-py
from gguf import GGUFReader

path, ncmoe = sys.argv[1], int(sys.argv[2])
per = collections.defaultdict(int)
for t in GGUFReader(path).tensors:
    if re.search(r'_exps\.', t.name): per[int(t.name.split('.')[1])] += t.n_bytes
on_gpu = {L: b for L, b in per.items() if L >= ncmoe}
print(f"{path.split('/')[-1]} ncmoe={ncmoe}: GPU expert layers {sorted(on_gpu)} = {sum(on_gpu.values())/2**20:.0f} MiB ({sum(on_gpu.values())/sum(per.values()):.1%} of experts)")
