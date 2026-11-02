#!/usr/bin/env python3
"""Per-model numbers quoted in docs/pibench.md and the model pages for a pibench batch.

    python3 bench/batch_stats.py dory qwen3vl agentic12 coder12 vibe3b > bench/results/stats-2026-11-02.txt

Reads the raw transcripts in bench/results/raw/<model>/ (RAW=... for another folder): solved, tests, median
time, timeouts, peak context, tool calls by name, tool errors, `edit` calls that changed nothing, calls
rejected for a missing `path`, thinking characters, and which hidden tests the failed books attempts missed.
"""
import collections
import glob
import json
import os
import statistics
import sys
import xml.etree.ElementTree as ET

BENCH = os.path.dirname(os.path.abspath(__file__))
RAW = os.environ.get("RAW", os.path.join(BENCH, "results/raw"))


def events(path):
    with open(path) as f:
        for line in f:
            try:
                yield json.loads(line)
            except ValueError:
                continue


def run_stats(run):
    tools = collections.Counter()
    errors = noop_edits = no_path = think = longest_args = 0
    for e in events(os.path.join(run, "pi.jsonl")):
        t = e.get("type")
        if t == "tool_execution_start":
            tools[e.get("toolName")] += 1
        elif t == "tool_execution_end":
            res = e.get("result") or {}
            text = json.dumps(res.get("content", ""))
            if e.get("isError") or res.get("isError"):
                errors += 1
            if e.get("toolName") == "edit" and "No changes made" in text:
                noop_edits += 1
            if "must have required properties path" in text:
                no_path += 1
        elif t == "message_end":
            m = e.get("message") or {}
            if isinstance(m, dict) and m.get("role") == "assistant":
                for c in m.get("content") or []:
                    if c.get("type") == "thinking":
                        think += len(c.get("thinking") or "")
        elif t == "message_update":
            d = (e.get("assistantMessageEvent") or {})
            if d.get("type") == "toolcall_delta":
                longest_args += len(d.get("delta") or "")
    return tools, errors, noop_edits, no_path, think, longest_args


def failed_tests(run):
    p = os.path.join(run, "junit.xml")
    if not os.path.exists(p):
        return []
    out = []
    for tc in ET.parse(p).iter("testcase"):
        f = tc.find("failure")
        if f is None:
            f = tc.find("error")
        if f is not None:
            out.append(f"{tc.get('name')}: {(f.get('message') or '').splitlines()[0][:110]}")
    return out


def main(models):
    for model in models:
        runs = sorted(glob.glob(os.path.join(RAW, model, "*", "r[0-9]")))
        if not runs:
            print(f"== {model}: no runs in {RAW}")
            continue
        scores = []
        tools = collections.Counter()
        errors = noop = nopath = think = 0
        for run in runs:
            with open(os.path.join(run, "score.json")) as f:
                scores.append(json.load(f))
            t, er, no, npth, th, _ = run_stats(run)
            tools += t
            errors += er
            noop += no
            nopath += npth
            think += th
        n = len(scores)
        tests = sum(s["passed"] for s in scores), sum(s["tests"] for s in scores)
        print(f"== {model}: {n} attempts")
        print(f"solved {sum(s['solved'] for s in scores)}/{n}; hidden tests {tests[0]}/{tests[1]} "
              f"({100 * tests[0] / tests[1]:.0f}%)")
        print(f"median time {statistics.median(s['wall_s'] for s in scores) / 60:.1f} min; "
              f"timeouts {sum(s['timed_out'] for s in scores)}; "
              f"peak context {max(s['peak_context'] for s in scores)}; "
              f"api errors {sum(s['api_errors'] for s in scores)}")
        print(f"tool calls {sum(tools.values())} ({sum(tools.values()) / n:.1f} a job): {dict(sorted(tools.items()))}")
        print(f"tool errors {errors} ({errors / n:.1f} a job); edits that changed nothing {noop} of "
              f"{tools.get('edit', 0)}; calls without a path {nopath}; attempts that wrote or edited a file "
              f"{sum(1 for r in runs if run_stats(r)[0].get('write', 0) + run_stats(r)[0].get('edit', 0))}")
        print(f"thinking {think / n / 1000:.1f}K characters a job")
        for run, s in zip(runs, scores):
            if s["solved"]:
                continue
            rel = os.path.relpath(run, RAW)
            longest = run_stats(run)[5]
            extra = f", {longest} chars of tool-call arguments in all" if longest > 20000 else ""
            print(f"  {rel}: {s['passed']}/{s['tests']}, {s['wall_s'] / 60:.1f} min"
                  f"{' TIMEOUT' if s['timed_out'] else ''}{extra}")
            if s["task"] == "scrape-books":
                for line in failed_tests(run):
                    print(f"      {line}")
        print()


if __name__ == "__main__":
    main(sys.argv[1:] or sorted(os.listdir(RAW)))
