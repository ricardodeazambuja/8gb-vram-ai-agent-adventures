#!/usr/bin/env python3
"""Compare a steered pibench run with the plain one: score and how much the agent repeats itself.

repeat_share is the part of a run's tool calls that are exact copies of an earlier call. longest_run
is the longest streak of one call in a row. Writes results/doom-lfm3vl.json.
"""
import glob
import json
import os
import statistics

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def calls(path):
    out = []
    with open(path, errors="replace") as f:
        for line in f:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            message = event.get("message") or {}
            if event.get("type") == "message_end" and message.get("role") == "assistant":
                for part in message.get("content") or []:
                    if isinstance(part, dict) and part.get("type") == "toolCall":
                        out.append(part.get("name", "") + json.dumps(part.get("arguments"), sort_keys=True))
    return out


def run_stats(folder):
    score = json.load(open(os.path.join(folder, "score.json")))
    made = calls(os.path.join(folder, "pi.jsonl"))
    longest = streak = 1 if made else 0
    for a, b in zip(made, made[1:]):
        streak = streak + 1 if a == b else 1
        longest = max(longest, streak)
    return {"task": score["task"], "rep": score["rep"], "solved": score["solved"], "passed": score["passed"],
            "tests": score["tests"], "wall_s": score["wall_s"], "timed_out": score["timed_out"],
            "output_tokens": score["output_tokens"], "tool_calls": len(made),
            "repeat_share": round(1 - len(set(made)) / len(made), 3) if made else 0, "longest_run": longest}


def arm(pattern, time_limit, note):
    runs = [run_stats(os.path.dirname(p)) for p in sorted(glob.glob(pattern))]
    return {"note": note, "time_limit_s": time_limit, "runs": len(runs),
            "solved": sum(r["solved"] for r in runs), "timeouts": sum(r["timed_out"] for r in runs),
            "median_repeat_share": round(statistics.median(r["repeat_share"] for r in runs), 3),
            "median_longest_run": statistics.median(r["longest_run"] for r in runs),
            "worst_longest_run": max(r["longest_run"] for r in runs),
            "per_run": runs}


result = {"plain": arm(f"{BENCH}/results/raw/lfm3vl/*/*/score.json", 1200, "no vector"),
          "steered": arm(f"{BENCH}/results/steering-doom/lfm3vl/*/*/score.json", 900,
                         "retry-lfm3vl.gguf at +0.03")}
with open(os.path.join(BENCH, "steering/results/doom-lfm3vl.json"), "w") as f:
    json.dump(result, f, indent=1)
for name, a in result.items():
    print(name, {k: v for k, v in a.items() if k != "per_run"})
