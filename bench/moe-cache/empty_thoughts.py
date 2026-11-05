#!/usr/bin/env python3
"""How often a model makes a tool call with no thinking, in real pibench transcripts.

    python3 bench/moe-cache/empty_thoughts.py bench/results/raw/gemma26b bench/results/raw/tcgemma ...

Those turns are where ThinkingCap's stock template dropped the empty thought block and made llama-server
reprocess the whole previous turn (README, "ThinkingCap: the empty thought block").
"""
import collections
import glob
import json
import os
import sys


def count(folder):
    c = collections.Counter()
    for f in glob.glob(os.path.join(folder, "*", "r*", "pi.jsonl")):
        with open(f) as fh:
            for line in fh:
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                m = e.get("message") or {}
                content = m.get("content")
                if e.get("type") != "message_end" or m.get("role") != "assistant" or not isinstance(content, list):
                    continue
                if any(x.get("type") == "toolCall" for x in content):
                    thought = any(x.get("thinking") for x in content if x.get("type") == "thinking")
                    c["with thinking" if thought else "without"] += 1
    return c


def main():
    for folder in sys.argv[1:]:
        c = count(folder)
        n = c["with thinking"] + c["without"]
        print(f"{folder}: {n} tool-call turns, {c['without']} without thinking ({100 * c['without'] / max(n, 1):.0f}%)")


if __name__ == "__main__":
    main()
