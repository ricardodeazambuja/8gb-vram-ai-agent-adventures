#!/usr/bin/env python3
"""Expert use on real agent sessions, and whether a ranking of experts carries over to work it wasn't made on.

    agent_experts.py TRACE SESSIONS.json COUNTS.npz [STORY.trace.xz]    (TRACE from experts/trace-qwen36.sh)
    agent_experts.py COUNTS.npz [STORY.trace.xz]                        (again, from the saved counts)

Splits TRACE into the sessions replay_sessions.py read (in order, prompt_n tokens each; anything before the first
is the server's warm-up and is skipped) and saves the per-session counts, layer x expert, to COUNTS.npz. Then:
  1. concentration over all sessions, as in expert_usage.py;
  2. leave one task out: rank each layer's experts on the other tasks, keep the top K%, and count the held-out
     task's picks that would land on a dropped expert;
  3. the same with the ranking from the story prompt the tuner used (STORY), scored on all agent sessions.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from expert_usage import coverage_share, read_trace

KEEPS = (0.5, 0.6, 0.75, 0.9)
STRETCH = 1034  # tokens, as many as the story trace has


def split_counts(trace_path, sessions_path):
    n_expert, records = read_trace(trace_path)
    with open(sessions_path) as f:
        sessions = json.load(f)
    layers = sorted(n_expert)
    first, ne = layers[0], max(n_expert.values())
    n_used = records[0][1]
    total = sum(len(ids) // n_used for il, _, ids in records if il == first)
    bounds = np.cumsum([total - sum(s["prompt_n"] for s in sessions)] + [s["prompt_n"] for s in sessions])
    print(f"trace: {total} tokens, sessions {bounds[-1] - bounds[0]}, skipped before the first: {bounds[0]}")
    counts = np.zeros((len(sessions), len(layers), ne), dtype=np.int64)
    stretch = np.zeros((len(sessions), len(layers), 2, ne), dtype=np.int64)  # first STRETCH tokens, by half
    seen, cur, start = 0, -1, 0  # tokens seen at the first layer, their session, where this ubatch starts in it
    for il, nu, ids in records:
        if il == first:
            cur = int(np.searchsorted(bounds, seen, side="right")) - 1
            start = seen - bounds[max(cur, 0)]
            seen += len(ids) // nu
        if 0 <= cur < len(sessions):
            li = layers.index(il)
            counts[cur, li] += np.bincount(ids, minlength=ne)
            for h, (lo, hi) in enumerate(((0, STRETCH // 2), (STRETCH // 2, STRETCH))):
                a, b = max(lo - start, 0), min(hi - start, len(ids) // nu)  # this ubatch's tokens in [lo, hi)
                if a < b:
                    stretch[cur, li, h] += np.bincount(ids[a * nu:b * nu], minlength=ne)
    return counts, stretch, [s["session"] for s in sessions], layers


def lost_share(rank_counts, test_counts, keep):  # per layer: test picks on experts outside the top `keep`
    ne = rank_counts.shape[1]
    order = np.argsort(rank_counts, axis=1)[:, ::-1][:, :round(keep * ne)]
    kept = np.take_along_axis(test_counts, order, axis=1).sum(axis=1)
    return 1 - kept / test_counts.sum(axis=1)


def main():
    a = sys.argv[1:]
    if a[0].endswith(".npz"):
        z = np.load(a[0])
        counts, stretch, names, layers = z["counts"], z["stretch"], list(z["names"]), list(z["layers"])
        story = a[1] if len(a) > 1 else None
    else:
        counts, stretch, names, layers = split_counts(a[0], a[1])
        np.savez_compressed(a[2], counts=counts, stretch=stretch, names=np.array(names), layers=np.array(layers))
        story = a[3] if len(a) > 3 else None
    tasks = [n.split("/")[-3] for n in names]
    # a layer with fewer picks than the others wasn't traced on every token (the last layer only runs the tokens
    # whose output is needed; a layer can take the cache's large-batch path, which isn't traced): leave it out
    tot = counts.sum(axis=(0, 2))
    ok = tot == tot.max()
    print(f"layers left out, picks short of {tot.max()}: {[(int(layers[i]), int(tot[i])) for i in np.flatnonzero(~ok)]}")
    counts, stretch = counts[:, ok], stretch[:, ok]
    alls = counts.sum(axis=0)
    print(f"{len(names)} sessions, {len(set(tasks))} tasks, {alls.shape[0]} layers x {alls.shape[1]} experts, "
          f"{tot.max()} picks per layer")
    unused = (alls == 0).mean(axis=1)
    print(f"  never picked: median {np.median(unused):.1%} of experts per layer (max {unused.max():.1%})")
    for q in (0.9, 0.99, 0.999):
        v = [coverage_share(row, q) for row in alls]
        print(f"  experts covering {q:.1%} of picks: median {np.median(v):.1%} per layer (worst layer {max(v):.1%})")
    print("  leave one task out (ranked on the other tasks), held-out picks on dropped experts, median over layers:")
    for k in KEEPS:
        per_task = {}
        for t in sorted(set(tasks)):
            test = counts[[i for i, x in enumerate(tasks) if x == t]].sum(axis=0)
            rank = counts[[i for i, x in enumerate(tasks) if x != t]].sum(axis=0)
            per_task[t] = lost_share(rank, test, k)
        med = {t: np.median(v) for t, v in per_task.items()}
        worst = max(v.max() for v in per_task.values())
        print(f"    keep {k:.0%}: {np.mean(list(med.values())):.2%} mean over tasks "
              f"(tasks {min(med.values()):.2%}-{max(med.values()):.2%}), worst layer of any task {worst:.2%}")
    v = [np.median(lost_share(s[:, 0], s[:, 1], 0.75)) for s in stretch]
    print(f"  expert_usage.py's test on each session's first {STRETCH} tokens (rank on the first half, keep the top 75%):"
          f" median {np.median(v):.2%} of second-half picks lost (sessions {min(v):.2%}-{max(v):.2%})")
    if story:
        n_expert, records = read_trace(story)
        sl = sorted(n_expert)
        sc = np.zeros((len(sl), alls.shape[1]), dtype=np.int64)
        for il, _, ids in records:
            sc[sl.index(il)] += np.bincount(ids, minlength=sc.shape[1])
        sc = sc[ok]
        print(f"  ranked on the story prompt ({sc.sum(axis=1)[0]} picks per layer), scored on all agent sessions:")
        for k in KEEPS:
            v = lost_share(sc, alls, k)
            print(f"    keep {k:.0%}: median {np.median(v):.2%} of picks, worst layer {v.max():.2%}")
        top = round(0.25 * alls.shape[1])
        ov = [len(set(np.argsort(x)[::-1][:top]) & set(np.argsort(y)[::-1][:top])) / top for x, y in zip(sc, alls)]
        print(f"  top 25% of experts, story vs agent: median overlap {np.median(ov):.0%} per layer")


if __name__ == "__main__":
    main()
