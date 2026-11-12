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


def decode(label, d=MC):  # mean decode tok/s of the 400-token runs in one srvbench log
    with open(f"{d}/srv-{label}.log") as f:
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
# ThinkingCap: the first two requests are the divergence check, then the agent loop (two loops, off/on, when fixed)
for label in ("tcap-stock", "tcap-nonl", "tcap-think-stock", "tcap-think-nonl"):
    same(f"ThinkingCap reprocessed per turn, {label}", reprocessed(label)[2:], [1416, 443, 616, 48, 48])
same("ThinkingCap reprocessed per turn, empty thoughts kept", reprocessed("tcap-empty")[4:],
     [2, 17, 17, 28, 28, 3, 17, 17, 28, 28])
same("Gemma reprocessed per turn, empty thoughts kept (off, on)", reprocessed("gemma-empty")[4:],
     [5, 14, 14, 25, 25, 5, 14, 14, 25, 25])
# daytime cache sizes (bench/moe-cache/README.md, step 16): night size (two runs) -> day size, decode change %
EX = f"{BENCH}/moe-cache/experimental/logs"
for name, night, day, want in [("q36", 2900, 1580, (38.7, 35.3, -9)), ("q38", 2770, 1770, (49.0, 43.1, -12)),
                               ("flint", 2168, 1024, (32.6, 29.0, -11)), ("ornith", 2983, 1713, (44.5, 38.5, -13))]:
    a = statistics.mean([decode(f"day-{name}-c{night}", EX), decode(f"day-{name}-c{night}-b", EX)])
    b = decode(f"day-{name}-c{day}", EX)
    same(f"{name} decode, night / day cache / change %", (round(a, 1), round(b, 1), round((b / a - 1) * 100)), want)
a = statistics.mean([decode("day-q36-ncmoe36", EX), decode("day-q36-ncmoe36-b", EX)])
b = decode("day-q36-c1580-c", EX)
same("Qwen3.6 daytime VRAM decode, n-cpu-moe 36 / cache 1580 / gain %", (round(a, 1), round(b, 1), round((b / a - 1) * 100)),
     (21.4, 34.8, 62))

# upstream's merged #30966 vs our pre-merge build (bench/moe-cache/README.md, step 18): old (two runs) / merged decode,
# change %, and the % of misses computed on the CPU in batches over 8 tokens, old / merged


def cpu_share_over8(label):
    with open(f"{EX}/srv-{label}.log") as f:
        m = re.findall(r"ubatch  > 8: hits = \d+, misses = (\d+) \((\d+) on the CPU\)", f.read())[-1]
    return round(100 * int(m[1]) / int(m[0]))


for q, want in [("q38", (48.0, 51.2, 7, 88, 9)), ("q36", (39.0, 40.4, 4, 89, 9))]:
    a = statistics.mean([decode(f"s18-{q}-old", EX), decode(f"s18-{q}-old-b", EX)])
    b = decode(f"s18-{q}-new", EX)
    same(f"{q} merged #30966: decode old / merged / change %, CPU share over 8 old / merged",
         (round(a, 1), round(b, 1), round((b / a - 1) * 100), cpu_share_over8(f"s18-{q}-old"), cpu_share_over8(f"s18-{q}-new")), want)
# step 18b: the merged build with GGML_OP_OFFLOAD_MIN_BATCH = limit + 1; old = mean of its three runs (18 twice, 18b)
for q, want in [("q38", (48.4, 53.2, 10, 89)), ("q36", (39.0, 41.5, 6, 89))]:
    a = statistics.mean([decode(f"s18-{q}-old", EX), decode(f"s18-{q}-old-b", EX), decode(f"s18b-{q}-old", EX)])
    b = decode(f"s18b-{q}-new-omb", EX)
    same(f"{q} merged + offload threshold: decode old / merged / change %, CPU share over 8",
         (round(a, 1), round(b, 1), round((b / a - 1) * 100), cpu_share_over8(f"s18b-{q}-new-omb")), want)
# step 18/18b turns (client side, turnbench rows in step18*.log): tok/s of the 48/100/200/600-token turns per label


def turns(label):
    txt = open(f"{EX}/step18.log").read() + open(f"{EX}/step18b.log").read()
    block = txt.split(f"### {label} |")[1].split("###")[0]
    return [float(m[1]) for m in re.finditer(r"^\s+\d+\s+\d+\s+\d+\s+([\d.]+)$", block, re.M)]


for q, want in [("q38", ([98.5, 146.4, 172.1, 243.8], [57.3, 78.7, 106.0, 243.9], [108.7, 165.1, 177.6, 243.7])),
                ("q36", ([76.7, 109.8, 139.5, 218.0], [63.0, 88.5, 141.7, 200.3], [83.1, 117.7, 140.9, 219.6]))]:
    old = [round(statistics.mean(v), 1) for v in zip(turns(f"s18-{q}-old"), turns(f"s18-{q}-old-b"), turns(f"s18b-{q}-old"))]
    same(f"{q} turns 48/100/200/600: old mean / merged / merged + threshold",
         (old, turns(f"s18-{q}-new"), turns(f"s18b-{q}-new-omb")), want)
# step 19/19c (bench/moe-cache/README.md, the full upstream sync): Qwen3.8 decode per run, server side
same("q38 decode, step 19: default / sync / default", [round(decode(f"s19-q38-{x}", EX), 1) for x in ("merged", "sync", "merged-b")],
     [51.8, 47.1, 51.2])
same("q36 decode, step 19: default / sync / default", [round(decode(f"s19-q36-{x}", EX), 1) for x in ("merged", "sync", "merged-b")],
     [42.6, 41.8, 41.2])
same("q38 decode, step 19c: default, sync, no-31888, sync no graphs, default no graphs, sync, default",
     [round(decode(f"s19c-{x}", EX), 1) for x in ("merged", "sync", "sync-no31888", "sync-nographs", "merged-nographs", "sync-b", "merged-b")],
     [47.6, 48.5, 49.5, 49.6, 49.5, 51.2, 50.7])
same("q38 decode at 2000 MiB, step 19c: MTP default / MTP sync / no MTP default / no MTP sync",
     [round(decode(f"s19c-{x}", EX), 1) for x in ("mtp-merged", "mtp-sync", "c2000-merged", "c2000-sync")], [43.4, 43.5, 47.0, 48.5])
# step 19 agent loop (prompt tokens reprocessed per turn) and step 19c MTP acceptance


def reproc19(label):
    block = open(f"{EX}/step19.log").read().split(f"### {label} |")[1].split("###")[0]
    rows = block.split("-- agent loop")[1].split("vram end")[0]
    return [int(m[1]) for m in re.finditer(r"^\s+\d+\s+\d+\s+\d+\s+(\d+)\s", rows, re.M)]


for q, want in [("q38", ([443, 25, 78, 78, 78, 25], [443, 25, 78, 78, 25, 25])), ("q36", ([443, 25, 212, 25, 25, 25],) * 2)]:
    same(f"{q} agent loop reprocessed per turn, step 19: default (both runs) / sync",
         (reproc19(f"s19-{q}-merged"), reproc19(f"s19-{q}-sync")), want)
    same(f"{q} agent loop, step 19: the two default runs agree", reproc19(f"s19-{q}-merged"), reproc19(f"s19-{q}-merged-b"))
for b in ("merged", "sync"):
    with open(f"{EX}/srv-s19c-mtp-{b}.log") as f:
        m = re.findall(r"#mean acc len = ([\d.]+), #acc rate/pos = \(([\d.]+)", f.read())[-1]  # cumulative over the run: the last line
    same(f"q38 MTP, step 19c {b}: mean accepted length, first-position acceptance", (m[0], m[1]), ("2.11", "0.839"))
# bench/template-compare (2026-11-11): Bonsai 2's template against Unsloth's on 18 captured pi requests x 4 efforts
TC = load(f"{BENCH}/template-compare/results/prompts.json")
tc_err = {a: sorted({k.split("#effort=")[-1] for k, r in TC.items() if r[a].startswith("ERROR")}) for a in ("bonsai2", "unsloth")}
tc_both = [r for r in TC.values() if not r["bonsai2"].startswith("ERROR") and not r["unsloth"].startswith("ERROR")]
same("template-compare: renders, errors (bonsai2), efforts that fail (bonsai2 / unsloth), identical where both render",
     (len(TC), sum(r["bonsai2"].startswith("ERROR") for r in TC.values()), tc_err["bonsai2"], tc_err["unsloth"],
      sum(r["bonsai2"] == r["unsloth"] for r in tc_both), len(tc_both)), (72, 18, ["high"], [], 54, 54))

print("\nall agree" if not bad else f"\n{bad} disagree")
raise SystemExit(bool(bad))
