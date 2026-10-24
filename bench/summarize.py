#!/usr/bin/env python3
"""Summarise bench/results/raw/*/*/*/score.json into bench/results/summary.md and results.csv."""
import csv
import glob
import json
import os
import statistics
from collections import defaultdict

BENCH = os.path.dirname(os.path.abspath(__file__))
scores = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(BENCH, "results/raw/*/*/*/score.json")))]
if not scores:
    raise SystemExit("no results yet")

with open(os.path.join(BENCH, "results/results.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(scores[0]))
    w.writeheader()
    w.writerows(scores)

tasks = sorted({s["task"] for s in scores})
by_model = defaultdict(list)
for s in scores:
    by_model[s["model"]].append(s)


def rate(runs):
    return sum(r["solved"] for r in runs), len(runs)


rows = []
for model, runs in by_model.items():
    solved, n = rate(runs)
    rows.append((solved / n, model, runs))
rows.sort(key=lambda r: (-r[0], r[1]))

lines = ["| Model | Solved | Tests passed | " + " | ".join(tasks) +
         " | Median time | Median output tokens | Timeouts |",
         "|---|---:|---:|" + "---:|" * len(tasks) + "---:|---:|---:|"]
for frac, model, runs in rows:
    solved, n = rate(runs)
    tp = sum(r["passed"] for r in runs) / max(sum(r["tests"] for r in runs), 1)
    per_task = []
    for t in tasks:
        tr = [r for r in runs if r["task"] == t]
        a, b = rate(tr)
        per_task.append(f"{a}/{b}" if b else "—")
    lines.append(f"| {model} | **{solved}/{n}** ({frac:.0%}) | {tp:.0%} | " + " | ".join(per_task) +
                 f" | {statistics.median(r['wall_s'] for r in runs) / 60:.1f} min"
                 f" | {statistics.median(r['output_tokens'] for r in runs):,.0f}"
                 f" | {sum(r['timed_out'] for r in runs)} |")

open(os.path.join(BENCH, "results/summary.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
