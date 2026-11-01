#!/usr/bin/env python3
"""Turn LFM3-1.7B's solved pibench runs into training examples for an agent LoRA on LFM3-VL.

Each example is one assistant turn: the conversation so far as text, and the turn itself, thinking
included. Stuck points with LFM3-1.7B's reply are added from replay.py --save; odd-numbered points are
kept out of training, so the replay can score them. Needs bench/results/raw/lfm3/ (git-ignored), so the
result, data/agent.json, is committed.
"""
import collections
import glob
import json
import os
import random
import sys

from replay import BENCH, HERE, TOOLS, load, to_openai
from transformers import AutoTokenizer

MAX_TOKENS = 2300
MAX_PER_TASK = 70
tok = AutoTokenizer.from_pretrained(os.path.join(BENCH, "..", "models/lfm3-vl-2b-hf"), trust_remote_code=False)


def fix(messages):
    """The template wants arguments as mappings."""
    out = []
    for m in messages:
        m = dict(m)
        if m.get("tool_calls"):
            m["tool_calls"] = [{"type": "function", "function": {
                "name": c["function"]["name"], "arguments": json.loads(c["function"]["arguments"] or "{}")}}
                for c in m["tool_calls"]]
        out.append(m)
    return out


def trimmed(prefix, keep):
    """System prompt, task, then the last `keep` messages, starting on an assistant turn."""
    rest = prefix[2:]
    if keep >= len(rest):
        return prefix
    rest = rest[-keep:]
    while rest and rest[0]["role"] != "assistant":
        rest = rest[1:]
    return prefix[:2] + rest


def example(prefix, target):
    """Old turns are dropped until it fits: long histories run the card out of memory."""
    for keep in (len(prefix), 16, 12, 8, 6, 4, 2, 0):
        short = trimmed(prefix, keep)
        head = tok.apply_chat_template(short, tools=TOOLS, tokenize=False, add_generation_prompt=True)
        full = tok.apply_chat_template([*short, target], tools=TOOLS, tokenize=False)
        if not full.startswith(head):
            raise ValueError("template prefix mismatch")
        if not full.rstrip().endswith("<|im_end|>"):
            full += "<|im_end|>\n"
        n = len(tok(full).input_ids)
        if n <= MAX_TOKENS:
            return {"head": head, "tail": full[len(head):], "tokens": n}
    return None


rows, skipped = [], 0
for score in sorted(glob.glob(f"{BENCH}/results/raw/lfm3/*/*/score.json")):
    with open(score) as f:
        if not json.load(f)["solved"]:
            continue
    system, messages = load(score.replace("score.json", "pi.jsonl"))
    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        prefix = fix(to_openai(system, messages[:i]))
        target = fix(to_openai(system, [m]))[1]
        ex = example(prefix, target)
        if ex:
            rows.append(ex | {"source": score.split("/")[-3]})
        else:
            skipped += 1
stuck = 0
if len(sys.argv) > 1:
    with open(sys.argv[1]) as f:
        for i, t in enumerate(json.load(f)):
            if i % 2:
                continue
            r = t["reply"]
            target = fix([{"role": "assistant", "content": r.get("content") or "", "tool_calls": r.get("tool_calls")}])[0]
            target["reasoning_content"] = r.get("reasoning_content") or ""
            ex = example(fix(t["messages"]), target)
            if ex:
                rows += [ex | {"source": "stuck"}] * 3
                stuck += 1
random.Random(0).shuffle(rows)
seen, kept = collections.Counter(), []
for r in rows:  # one task with long runs shouldn't fill the set
    seen[r["source"]] += 1
    if seen[r["source"]] <= MAX_PER_TASK or r["source"] == "stuck":
        kept.append(r)
rows = kept
os.makedirs(f"{HERE}/data", exist_ok=True)
with open(f"{HERE}/data/agent.json", "w") as f:
    json.dump(rows, f)
print(len(rows), "examples,", skipped, "too long,", stuck, "stuck points; mean tokens",
      sum(r["tokens"] for r in rows) // len(rows))
