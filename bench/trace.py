#!/usr/bin/env python3
"""Condense one run's pi transcript into a readable trace, for qualitative notes.

usage: trace.py RUN_DIR [RUN_DIR ...]      e.g. trace.py bench/results/raw/flint/*/r1
       trace.py --stats                    habits per model, over every finished run

Per run: the score, then one line per tool call (name + a short view of its arguments), the
size of each reasoning block, and the model's final message.
"""
import json
import os
import sys


def short(s, n=140):
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 1] + "…"


def events(d):
    for line in open(os.path.join(d, "pi.jsonl"), encoding="utf-8", errors="replace"):
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            pass


def habits(d):
    """What one run did: tests run, outputs checked, starter files overwritten, tool errors..."""
    task_dir = os.path.join(os.path.dirname(BENCH_RAW), "..", "tasks", json.load(open(os.path.join(d, "score.json")))["task"])
    starters = set(os.listdir(os.path.join(task_dir, "files"))) if os.path.isdir(os.path.join(task_dir, "files")) else set()
    h = dict(pytest=0, ran_python=0, curl=0, checked_output=0, overwrote_starter=0, tool_errors=0, think_chars=0,
             written=0)
    wrote_something = False
    for ev in events(d):
        if ev.get("type") == "tool_execution_end" and ev.get("isError"):
            h["tool_errors"] += 1
        if ev.get("type") != "message_end" or (ev.get("message") or {}).get("role") != "assistant":
            continue
        for c in ev["message"].get("content") or []:
            if c.get("type") == "thinking":
                h["think_chars"] += len(c.get("thinking", ""))
            if c.get("type") != "toolCall":
                continue
            a = c.get("arguments") or {}
            name, cmd, path = c.get("name"), str(a.get("command", "")), str(a.get("path", ""))
            if name == "write":
                h["written"] += 1
                wrote_something = True
                if os.path.basename(path) in starters and not path.endswith(".py"):
                    h["overwrote_starter"] = 1
            if name == "bash":
                h["pytest"] += "pytest" in cmd
                h["ran_python"] += "python" in cmd
                h["curl"] += "curl" in cmd
                if wrote_something and any(k in cmd for k in ("cat ", "head", "json.load", "csv.reader", "wc -l")):
                    h["checked_output"] = 1
            if name == "read" and wrote_something:
                h["checked_output"] = 1
    return h


BENCH_RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "raw")
if sys.argv[1:] == ["--stats"]:
    import glob
    from collections import defaultdict
    agg = defaultdict(lambda: defaultdict(float))
    n = defaultdict(int)
    for sp in sorted(glob.glob(os.path.join(BENCH_RAW, "*", "*", "*", "score.json"))):
        d, sc = os.path.dirname(sp), json.load(open(sp))
        m = sc["model"]
        n[m] += 1
        for k, v in habits(d).items():
            agg[m][k] += v
        for k in ("tool_calls", "output_tokens", "solved"):
            agg[m][k] += sc[k]
    cols = ["solved", "tool_calls", "output_tokens", "think_chars", "pytest", "curl", "tool_errors",
            "checked_output", "overwrote_starter"]
    print("per-run averages (checked_output / overwrote_starter / solved = share of runs)")
    print(f"{'model':10} {'runs':>4} " + " ".join(f"{c[:12]:>12}" for c in cols))
    for m in agg:
        print(f"{m:10} {n[m]:4} " + " ".join(f"{agg[m][c] / n[m]:12.2f}" for c in cols))
    sys.exit(0)

for d in sys.argv[1:]:
    score = json.load(open(os.path.join(d, "score.json")))
    print(f"=== {score['model']} {score['task']} r{score['rep']}: {score['passed']}/{score['tests']}, "
          f"{score['wall_s'] / 60:.1f} min, {score['output_tokens']} tok, {score['tool_calls']} calls"
          f"{' TIMEOUT' if score['timed_out'] else ''}")
    final = ""
    for line in open(os.path.join(d, "pi.jsonl"), encoding="utf-8", errors="replace"):
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "message_end" and (ev.get("message") or {}).get("role") == "assistant":
            for c in ev["message"].get("content") or []:
                t = c.get("type")
                if t == "thinking":
                    print(f"   . think {len(c.get('thinking', ''))} chars")
                elif t == "toolCall":
                    a = c.get("arguments") or {}
                    view = a.get("command") or a.get("path") or a
                    print(f"   > {c.get('name')}: {short(view)}")
                elif t == "text" and c.get("text", "").strip():
                    final = c["text"]
        elif ev.get("type") == "tool_execution_end" and ev.get("isError"):
            print(f"     ! tool error: {short(ev.get('result'), 100)}")
    print(f"   final: {short(final, 300)}")
    print()
