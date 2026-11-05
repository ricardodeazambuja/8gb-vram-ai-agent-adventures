import collections
import os
import re
import sys

sys.path.insert(0, os.path.join(os.environ.get('ADV_LLAMA', os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../llama.cpp')), 'gguf-py'))  # our fork's gguf-py
from gguf import GGUFReader

for path in sys.argv[1:]:
    r = GGUFReader(path)
    kv = {f.name: f for f in r.fields.values()}
    arch = bytes(kv['general.architecture'].parts[-1]).decode()
    def g(k, kv=kv, arch=arch):
        f = kv.get(f'{arch}.{k}')
        return None if f is None else f.parts[-1][0]
    nexp, topk, nl = g('expert_count'), g('expert_used_count'), g('block_count')
    per_layer = collections.defaultdict(int); types=collections.Counter(); total=0; exp_total=0; shexp=0
    for t in r.tensors:
        total += t.n_bytes
        if re.search(r'_exps\.', t.name):
            L = int(t.name.split('.')[1]); per_layer[L] += t.n_bytes; exp_total += t.n_bytes; types[(t.name.split('.')[2], t.tensor_type.name)] += 1
    moe_layers = len(per_layer)
    print(f"{path.split('/')[-1]}: arch={arch} layers={nl} moe_layers={moe_layers} n_exp={nexp} top_k={topk}")
    print(f"  file tensors {total/2**30:.2f} GiB, experts {exp_total/2**30:.2f} GiB, non-expert {(total-exp_total)/2**30:.2f} GiB")
    vals = sorted(set(per_layer.values()))
    print(f"  per-expert bytes (all 3 mats) min {min(vals)/nexp/2**20:.3f} MiB max {max(vals)/nexp/2**20:.3f} MiB mean {exp_total/moe_layers/nexp/2**20:.3f} MiB")
    print("  types:", dict(types))
