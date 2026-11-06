#!/usr/bin/env python3
"""Write a copy of a MoE GGUF without the least-picked experts of each layer (frequency pruning, the simplest
cousin of REAP). Every layer keeps the same number of experts, since llama.cpp has one expert_count.

    GGUF_PY=$ADV_LLAMA/gguf-py prune_gguf.py IN.gguf OUT.gguf RANK.npy KEEP    (the fork's gguf-py)

RANK.npy: picks per layer and expert (n_layer x n_expert), e.g. from agent_experts.py. KEEP: experts to keep per
layer. Drops the experts' slices of every blk.N.ffn_*_exps tensor and their rows of blk.N.ffn_gate_inp (the
router), keeps the survivors in their old order, and sets {arch}.expert_count. Shared experts are untouched.
"""
import os
import re
import sys

import numpy as np

if os.environ.get("GGUF_PY"):
    sys.path.insert(0, os.environ["GGUF_PY"])
from gguf import GGUFReader, GGUFValueType, GGUFWriter, Keys

EXPERT_TENSOR = re.compile(r"blk\.(\d+)\.(ffn_(?:gate|up|down|gate_up)_exps\.weight|ffn_gate_inp\.weight|exp_probs_b\.bias)$")


def main():
    src, dst, rank_path, keep = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    reader = GGUFReader(src)
    arch = reader.fields[Keys.General.ARCHITECTURE].contents()
    rank = np.load(rank_path)
    n_expert = reader.fields[f"{arch}.expert_count"].contents()
    if rank.shape[1] != n_expert or not 0 < keep <= n_expert:
        sys.exit(f"rank has {rank.shape[1]} experts, the model {n_expert}; keep {keep}")
    # the experts to keep, per layer, in their old order (stable sort: ties go to the lower id)
    kept = {il: np.sort(np.argsort(-rank[il], kind="stable")[:keep]) for il in range(rank.shape[0])}

    writer = GGUFWriter(dst, arch, endianess=reader.endianess)
    for field in reader.fields.values():
        if field.name == Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == GGUFValueType.ARRAY else None
        value = keep if field.name == f"{arch}.expert_count" else field.contents()
        writer.add_key_value(field.name, value, val_type, sub_type=sub_type)

    def data_of(t):
        m = EXPERT_TENSOR.match(t.name)
        if m is None:
            return t.data
        il = int(m[1])
        if il not in kept:
            sys.exit(f"{t.name}: no ranking for layer {il}")
        if t.data.shape[0] != n_expert:
            sys.exit(f"{t.name}: expected {n_expert} experts on the first axis, got shape {t.data.shape}")
        return t.data[kept[il]]

    for t in reader.tensors:
        d = data_of(t)
        writer.add_tensor_info(t.name, d.shape, d.dtype, d.nbytes, t.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for t in reader.tensors:
        writer.write_tensor_data(np.ascontiguousarray(data_of(t)), tensor_endianess=reader.endianess)
    writer.close()
    print(f"{dst}: {keep} of {n_expert} experts per layer, {len(kept)} layers")


if __name__ == "__main__":
    main()
