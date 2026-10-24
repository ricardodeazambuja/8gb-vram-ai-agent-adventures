#!/usr/bin/env python3
"""Turn one run's junit.xml + pi.jsonl into score.json (called by run-bench.sh).

usage: score.py RUN_DIR MODEL_KEY MODEL_ID TASK REP EXIT_CODE T0 T1 TIME_LIMIT
"""
import json
import os
import sys
import xml.etree.ElementTree as ET

d, key, mid, task, rep, rc, t0, t1, limit = sys.argv[1:10]
rc, wall, limit = int(rc), float(t1) - float(t0), int(limit)

# Hidden tests: pytest's junit report (absent or empty if pytest never ran, e.g. no file to import).
tests = failed = 0
try:
    root = ET.parse(os.path.join(d, "junit.xml")).getroot()
    for suite in ([root] if root.tag == "testsuite" else root.iter("testsuite")):
        tests += int(suite.get("tests", 0))
        failed += int(suite.get("failures", 0)) + int(suite.get("errors", 0))
except (ET.ParseError, FileNotFoundError):
    pass
passed = max(tests - failed, 0)

# pi's event stream: every finished assistant message carries its token usage and tool calls.
out_tokens = in_tokens = turns = tool_calls = errors = 0
last_stop = None
with open(os.path.join(d, "pi.jsonl"), encoding="utf-8", errors="replace") as f:
    for line in f:
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") != "message_end":
            continue
        m = ev.get("message") or {}
        if m.get("role") != "assistant":
            continue
        turns += 1
        u = m.get("usage") or {}
        out_tokens += u.get("output", 0)
        in_tokens = max(in_tokens, u.get("input", 0) + u.get("cacheRead", 0))
        tool_calls += sum(1 for c in m.get("content") or [] if c.get("type") == "toolCall")
        last_stop = m.get("stopReason")
        errors += last_stop == "error"

score = {
    "model": key, "model_id": mid, "task": task, "rep": int(rep),
    "tests": tests, "passed": passed, "solved": tests > 0 and passed == tests,
    "wall_s": round(wall, 1), "timed_out": rc in (124, 137) or wall >= limit,
    "exit_code": rc, "output_tokens": out_tokens, "peak_context": in_tokens,
    "turns": turns, "tool_calls": tool_calls, "api_errors": errors, "last_stop": last_stop,
}
json.dump(score, open(os.path.join(d, "score.json"), "w"), indent=1)
print(f"     -> {passed}/{tests} tests, {wall:.0f}s, {out_tokens} out tokens, {tool_calls} tool calls"
      f"{' TIMEOUT' if score['timed_out'] else ''}")
