#!/usr/bin/env python3
"""qwen-codecheck.py — the head-to-head code check from docs/models/qwen3.6-35b-a3b.md.

Asks whatever llama-server is listening for a `merge_intervals` function with 5
doctests, then EXECUTES the answer instead of trusting it:
  1. the model's own doctests (python -m doctest)
  2. 500 random inputs against a reference implementation (seeded, so every model
     gets the same inputs)
and reports tokens used, generation speed and both results.

The model's code runs in a subprocess for the doctests, but the random check
exec()s it in this process — only point this at a server you trust, and read
the saved file if in doubt. With --sandbox, both checks run instead in a
throwaway container (the pibench image) with no network, no capabilities, a
read-only filesystem and only the saved answer mounted: the way to run it on a
model that hasn't earned trust yet (docs/security.md).

Usage:  python3 scripts/qwen-codecheck.py [--port 8080] [--out /tmp/merge_intervals.py] [--sandbox]
"""
import argparse
import json
import os
import random
import re
import subprocess
import sys
import urllib.request

PROMPT = ("Write a Python function merge_intervals(intervals) that merges overlapping intervals "
          "(lists of [start, end]), returning a sorted list. Include a docstring with 5 doctests "
          "covering edge cases (empty, touching intervals, nested, unsorted input). "
          "Reply with one python code block only.")


def reference(iv):
    res = []
    for s, e in sorted(iv):
        if res and s <= res[-1][1]:
            res[-1][1] = max(res[-1][1], e)
        else:
            res.append([s, e])
    return res


def check(path):
    """Run the model's own doctests, then 500 random inputs against the reference."""
    with open(path) as f:
        code = f.read()
    r = subprocess.run([sys.executable, "-I", "-m", "doctest", "-v", path],
                       capture_output=True, text=True, timeout=60)
    print("own doctests:", (r.stdout.strip().splitlines() or ["?"])[-1], r.stderr[-300:])

    ns = {}
    exec(compile(code, path, "exec"), ns)
    random.seed(0)
    bad = 0
    for _ in range(500):
        iv = [[s, s + random.randint(0, 5)]
              for s in (random.randint(0, 30) for _ in range(random.randint(0, 8)))]
        got = ns["merge_intervals"]([x[:] for x in iv])
        if [list(x) for x in got] != reference(iv):
            bad += 1
    print("random tests vs reference: %d/500 wrong" % bad)


def check_in_sandbox(path, image):
    here = os.path.abspath(__file__)
    subprocess.run(["docker", "run", "--rm", "--network", "none", "--cap-drop", "ALL",
                    "--security-opt", "no-new-privileges", "--read-only", "--tmpfs", "/tmp",
                    "--memory", "1g", "--pids-limit", "128",
                    "-v", f"{here}:/c/check.py:ro", "-v", f"{os.path.abspath(path)}:/c/answer.py:ro",
                    image, "timeout", "120", "python3", "-I", "/c/check.py", "--check", "/c/answer.py"],
                   check=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--out", default="/tmp/merge_intervals.py")
    ap.add_argument("--sandbox", action="store_true", help="run the model's code in a no-network container")
    ap.add_argument("--image", default="pibench:1", help="container image for --sandbox")
    ap.add_argument("--check", metavar="FILE", help=argparse.SUPPRESS)  # used inside the sandbox
    a = ap.parse_args()
    if a.check:
        check(a.check)
        return

    body = {"messages": [{"role": "user", "content": PROMPT}], "max_tokens": 6000}
    req = urllib.request.Request(f"http://127.0.0.1:{a.port}/v1/chat/completions",
                                 json.dumps(body).encode(), {"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=1800))
    c, t = d["choices"][0], d["timings"]
    print("finish=%s tokens=%d gen=%.1f t/s time=%.0fs" % (
        c["finish_reason"], t["predicted_n"], t["predicted_per_second"], t["predicted_ms"] / 1000))

    code = re.findall(r"```(?:python)?\n(.*?)```", c["message"].get("content") or "", re.S)
    if not code:
        print("NO CODE BLOCK in the answer")
        sys.exit(1)
    with open(a.out, "w") as f:
        f.write(code[0])
    print("saved:", a.out)
    if a.sandbox:
        check_in_sandbox(a.out, a.image)
    else:
        check(a.out)


if __name__ == "__main__":
    main()
