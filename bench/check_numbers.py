#!/usr/bin/env python3
"""Recompute the headline numbers of the docs from the committed result files, and say if any disagree.

Needs no GPU, no models and nothing git-ignored, so it works on a clean clone. A rerun that writes new
result files can be compared the same way, remembering that sampled runs differ a little.
Usage: python3 bench/check_numbers.py
"""
import json
import math
import os
import statistics

BENCH = os.path.dirname(os.path.abspath(__file__))
BV = f"{BENCH}/beyond-vision/results"
VC = f"{BENCH}/visual-control/results"
bad = 0


def load(path):
    with open(path) as f:
        return json.load(f)


def same(label, got, want):
    global bad
    ok = got == want
    bad += not ok
    print(f"{'ok ' if ok else 'BAD'} {label}: {got}" + ("" if ok else f" (docs say {want})"))


def repeats(name, arm):
    rows = [r for r in load(f"{BV}/replay-{name}.json") if r["arm"] == arm]
    return sum(r["repeats"] for r in rows), sum(r["samples"] for r in rows), sum(r["repeats"] > 0 for r in rows)


def glance(key):
    items = load(f"{VC}/glance-{key}.json")["items"]
    return sum(i["ok"] for i in items), len(items)


def fisher(a, b, c, d):
    """Two-sided exact test on [[a, b], [c, d]]."""
    n, row, col = a + b + c + d, a + b, a + c
    p = lambda x: math.comb(col, x) * math.comb(n - col, row - x) / math.comb(n, row)
    low, high = max(0, row + col - n), min(row, col)
    return sum(p(x) for x in range(low, high + 1) if p(x) <= p(a) * (1 + 1e-9))


def bound(n, tail=0.05):
    """95% two-sided: with 0/n, the true rate is under this."""
    return 1 - (tail / 2) ** (1 / n)


same("replay VL 0.1", repeats("lfm3vl-think", "default"), (32, 52, 11))
same("replay VL 0.7", repeats("lfm3vl-think", "temp0.7"), (8, 52, 5))
same("replay 1.7B", repeats("lfm3-think", "default"), (0, 52, 0))
rows = load(f"{BV}/replay-lfm3vl-think.json")
odd = {r["run"] for i, r in enumerate(rows[::2]) if i % 2}
base = sum(r["repeats"] for r in rows if r["arm"] == "default" and r["run"] in odd)
lora = sum(r["repeats"] for r in load(f"{BV}/replay-agent-lora.json") if r["arm"] == "default" and r["run"] in odd)
same("held-out repeats, plain", base, 21)
same("held-out repeats, agent LoRA", lora, 7)

doom = load(f"{BENCH}/steering/results/doom-lfm3vl.json")
for key, r, want in (("plain", doom["plain"], (0, 18, 0.93, 769, 14)), ("steered", doom["steered"], (0, 6, 0.80, 129, 4)),
                     ("temp 0.7", load(f"{BV}/pibench-temp07.json"), (0, 6, 0.24, 3, 3)),
                     ("agent LoRA", load(f"{BV}/pibench-agent-lora.json"), (0, 6, 0.48, 34, 3))):
    same(f"pibench {key}", (r["solved"], r["runs"], round(r["median_repeat_share"], 2), r["worst_longest_run"], r["timeouts"]), want)
for key, r, want in (("plain", doom["plain"], (76, 162)), ("steered", doom["steered"], (7, 28)),
                     ("temp 0.7", load(f"{BV}/pibench-temp07.json"), (30, 54)),
                     ("agent LoRA", load(f"{BV}/pibench-agent-lora.json"), (27, 54))):
    same(f"tests {key}", (sum(x["passed"] for x in r["per_run"]), sum(x["tests"] for x in r["per_run"])), want)

same("0/6 upper bound", round(bound(6), 2), 0.46)
same("VL 0/6 against 1.7B 12/18, Fisher p", round(fisher(0, 6, 12, 6), 2), 0.01)
same("30/30 lower bound", round(1 - bound(30), 2), 0.88)
same("Gemma 30/30 against Groot 27/30, p", round(fisher(30, 0, 27, 3), 2), 0.24)
same("agent LoRA pictures 101 against 110 of 150, p", round(fisher(101, 49, 110, 40), 2), 0.31)

same("Gemma glance", glance("gemma26b-think-on"), (30, 30))
same("Groot glance", glance("groot-think-on"), (27, 30))
same("Gemma median s", round(statistics.median(i["secs"] for i in load(f"{VC}/glance-gemma26b-think-on.json")["items"])), 17)
same("Groot median s", round(statistics.median(i["secs"] for i in load(f"{VC}/glance-groot-think-on.json")["items"])), 16)
same("graft, thinking off", glance("graft-think-off"), (14, 150))
same("agent LoRA glance", glance("agent-lora-think-off"), (101, 150))

eyes = load(f"{BV}/eyes.json")
same("two models", (sum(i["ok"] for i in eyes), len(eyes)), (57, 60))
same("two models, kinds of miss", sorted({i["scene"] for i in eyes if not i["ok"]}), ["tanks"])
same("two models, s", round(statistics.median(i["eye_secs"] + i["driver_secs"] for i in eyes)), 16)

proj = load(f"{BV}/projector-eval.json")
same("new projector", sum(i["ok"] for i in proj["items"]), 35)
for scene in ("tanks", "signal", "gauge"):
    answers = {json.dumps(i.get("calls") or i.get("text")) for i in proj["items"] if i["scene"] == scene}
    same(f"new projector, different answers to {scene}", len(answers), 1)

print("\nall agree" if not bad else f"\n{bad} disagree")
raise SystemExit(bool(bad))
