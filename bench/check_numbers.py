#!/usr/bin/env python3
"""Recompute the headline numbers of the docs from the committed result files, and say if any disagree.

Needs no GPU, no models and nothing git-ignored, so it works on a clean clone. A rerun that writes new
result files can be compared the same way, remembering that sampled runs differ a little.
Usage: python3 bench/check_numbers.py
"""
import json
import math
import os
import re
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
with open(f"{BENCH}/vision/results/summary.md") as f:
    vision = f.read()
same("vision medians, s", [re.search(rf"\| {m} \|.*\| (\d+) s \|", vision)[1] for m in ("qwen3vl", "lfm3vl", "bolt")], ["5", "8", "110"])
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

# MoE expert cache and the Gemma template (docs/how-it-fits.md §12, bench/moe-cache/)
MC = f"{BENCH}/moe-cache/logs"


def decode(label):  # mean decode tok/s of the 400-token runs in one srvbench log
    with open(f"{MC}/srv-{label}.log") as f:
        runs = [float(m[1]) for m in re.finditer(r"eval time = .*/\s+400 tokens .*?([\d.]+) tokens per second", f.read())]
    return statistics.mean(runs)


def reprocessed(label):  # prompt tokens processed per request in one agent-loop log
    with open(f"{MC}/srv-{label}.log") as f:
        return [int(m[1]) for m in re.finditer(r"prompt eval time = .*/\s+(\d+) tokens", f.read())]


for name, base, cache, want in [("Qwen3.6", "ncmoe32", "cache2900-s0.95", (24.9, 39.4, 58)),
                                ("Gemma 26B", "gemma-ncmoe22", "gemma-cache-s0.95", (21.0, 30.0, 43)),
                                ("Flint", "flint-ncmoe42", "flint-cache-s0.95", (26.9, 32.1, 19))]:
    a, b = decode(base), decode(cache)
    same(f"{name} decode, layers / cache / gain %", (round(a, 1), round(b, 1), round((b / a - 1) * 100)), want)
same("Gemma reprocessed per turn, stock template", reprocessed("pi-gemma-stock"), [1416, 743, 1147, 47, 41])
same("Gemma reprocessed per turn, no-newline template", reprocessed("pi-gemma-nonl"), [1416, 14, 14, 25, 25])

print("\nall agree" if not bad else f"\n{bad} disagree")
raise SystemExit(bool(bad))
