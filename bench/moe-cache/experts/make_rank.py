#!/usr/bin/env python3
"""The expert ranking prune_gguf.py uses: picks per layer and expert on every task but the held-out one.

    make_rank.py COUNTS.npz HELDOUT_TASK OUT.npy [EXTRA.trace]

COUNTS.npz is agent_experts.py's (server replay; layers whose prompt batches skipped the traced path are short of
picks there). EXTRA.trace, if given, is a trace of the ranking tasks' text taken with batches small enough for every
layer to be traced (llama-perplexity -ub 16): its counts fill in the short layers. Prints where each layer's
ranking came from and how many picks it rests on.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from expert_usage import read_trace


def main():
    z = np.load(sys.argv[1])
    counts, names, layers = z["counts"], list(z["names"]), [int(x) for x in z["layers"]]
    held = sys.argv[2]
    tasks = [n.split("/")[-3] for n in names]
    if held not in tasks:
        sys.exit(f"no task {held}: {sorted(set(tasks))}")
    rank = counts[[i for i, t in enumerate(tasks) if t != held]].sum(axis=0)
    tot = rank.sum(axis=1)
    short = [i for i in range(len(layers)) if tot[i] < tot.max()]
    if short and len(sys.argv) < 5:
        sys.exit(f"layers {[layers[i] for i in short]} are short of picks; give EXTRA.trace")
    if short:
        _, records = read_trace(sys.argv[4])
        extra = np.zeros((len(layers), rank.shape[1]), dtype=np.int64)
        for il, _, ids in records:
            extra[layers.index(il)] += np.bincount(ids, minlength=rank.shape[1])
        for i in short:
            print(f"layer {layers[i]}: {extra[i].sum()} picks from {sys.argv[4]} (the replay had {tot[i]})")
            rank[i] = extra[i]
    print(f"other layers: {tot.max()} picks each, from {len(names) - tasks.count(held)} sessions without {held}")
    np.save(sys.argv[3], rank)


if __name__ == "__main__":
    main()
