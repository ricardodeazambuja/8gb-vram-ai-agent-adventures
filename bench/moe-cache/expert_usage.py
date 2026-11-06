#!/usr/bin/env python3
"""How concentrated is expert use? Reads a MoE cache trace (the fork's MCT1 format, written by
scripts/moe-cache-tune.py) and counts, per layer, how often the router picked each expert (one count per token
and pick, prompt and generation alike).

    expert_usage.py TRACE.trace.xz [...]

Per model: the share of experts never picked, and the share needed to cover 90 / 99 / 99.9% of the picks
(median over layers, and the worst layer). Then a held-out check of what pruning would cost: rank the experts
on the first half of the tokens, keep the top K% per layer, and count the second half's picks that fall on a
dropped expert. Frequency only: REAP also weighs the gate value and the expert's output size, which the trace
doesn't have.
"""
import lzma
import struct
import sys

import numpy as np


def read_trace(path):  # mirrors read_trace() in the fork's scripts/moe-cache-tune.py
    with (lzma.open if path.endswith(".xz") else open)(path, "rb") as f:
        data = f.read()
    if data[:4] != b"MCT1":
        sys.exit(f"{path}: not a MoE cache trace")
    pos = 4

    def take(fmt):
        nonlocal pos
        v = struct.unpack_from("<" + fmt, data, pos)
        pos += struct.calcsize("<" + fmt)
        return v

    _, _, n_group = take("3I")
    n_expert = {}
    for _ in range(n_group):
        _, ne, n_layers = take("3I")
        for il in take(f"{n_layers}I"):
            n_expert[il] = ne
        take("2d")
        take("3Q")
        (n_tensors,) = take("I")
        take(f"{n_tensors}Q")
    records = []
    while pos < len(data):
        il, n_used, n_ids = struct.unpack_from("<3i", data, pos)
        pos += 12
        records.append((il, n_used, np.frombuffer(data, dtype="<i4", count=n_ids, offset=pos)))
        pos += 4 * n_ids
    return n_expert, records


def coverage_share(counts, q):  # smallest share of experts whose picks cover q of the total
    c = np.sort(counts)[::-1]
    return (np.searchsorted(np.cumsum(c), q * c.sum()) + 1) / len(c)


def main():
    for path in sys.argv[1:]:
        n_expert, records = read_trace(path)
        per_layer = {}  # layer -> list of id arrays in order
        for il, _, ids in records:
            per_layer.setdefault(il, []).append(ids)
        first = min(per_layer)
        n_tok = sum(len(ids) for ids in per_layer[first]) // records[0][1]
        print(f"{path}\n  {len(per_layer)} MoE layers, {n_expert[first]} experts each, "
              f"{records[0][1]} per token, {n_tok} tokens")
        unused, cov = [], {q: [] for q in (0.9, 0.99, 0.999)}
        keeps = (0.5, 0.6, 0.75, 0.9)
        lost = {k: [] for k in keeps}
        for il, chunks in sorted(per_layer.items()):
            ne = n_expert[il]
            ids = np.concatenate(chunks)
            counts = np.bincount(ids, minlength=ne)
            unused.append((counts == 0).mean())
            for q, v in cov.items():
                v.append(coverage_share(counts, q))
            half = len(ids) // 2
            a, b = np.bincount(ids[:half], minlength=ne), np.bincount(ids[half:], minlength=ne)
            order = np.argsort(a)[::-1]
            for k in keeps:
                kept = order[:round(k * ne)]
                lost[k].append(1 - b[kept].sum() / b.sum())
        print(f"  never picked: median {np.median(unused):.1%} of experts per layer (max {max(unused):.1%})")
        for q, v in cov.items():
            print(f"  experts covering {q:.1%} of picks: median {np.median(v):.1%} per layer (worst layer {max(v):.1%})")
        print("  keep the top K% per layer (ranked on the first half), second-half picks that hit a dropped expert:")
        for k, v in lost.items():
            print(f"    keep {k:.0%}: median {np.median(v):.2%} of picks, worst layer {max(v):.2%}")


if __name__ == "__main__":
    main()
