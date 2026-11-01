#!/usr/bin/env python3
"""Replay stuck pibench conversations: does this model and sampling repeat the call, or get out?

A stuck point is the first place in a run where the last two turns (or the last two pairs of turns, for
loops that alternate) made identical tool calls and the run really did repeat them next. The prefix up to
there is sent to the server on 127.0.0.1:8080, several times with different seeds, per sampling arm.
Points are the unit of analysis: at low temperature, samples from one point are nearly copies.

Usage: replay.py find                       list the stuck points
       replay.py run NAME [--arms a,b] [--samples 4]   writes results/replay-NAME.json
"""
import argparse
import glob
import json
import os
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
URL = "http://127.0.0.1:8080/v1/chat/completions"
MAX_CHARS = 60000  # keep prefixes well inside the context
TOOLS = [{"type": "function", "function": {"name": n, "description": d, "parameters": {
    "type": "object", "properties": {k: {"type": t} for k, t in p.items()}, "required": r}}} for n, d, p, r in [
    ("read", "Read file contents", {"path": "string", "offset": "number", "limit": "number"}, ["path"]),
    ("bash", "Execute bash commands", {"command": "string", "timeout": "number"}, ["command"]),
    ("edit", "Make precise file edits with exact text replacement", {"path": "string", "edits": "array"},
     ["path", "edits"]),
    ("write", "Create or overwrite files", {"path": "string", "content": "string"}, ["path", "content"])]]
ARMS = {"default": {},
        "temp0.7": {"temperature": 0.7},
        "window": {"repeat_last_n": 4096},
        "temp0.7+window": {"temperature": 0.7, "repeat_last_n": 4096}}


def sig(message):
    return tuple(sorted(p["name"] + json.dumps(p.get("arguments"), sort_keys=True)
                        for p in message.get("content") or [] if isinstance(p, dict) and p.get("type") == "toolCall"))


def load(path):
    """System prompt and the messages before the first compaction."""
    system, messages = "", []
    with open(path, errors="replace") as f:
        for line in f:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "compaction_start":
                break
            m = event.get("message") or {}
            if event.get("type") == "message_end":
                if m.get("role") == "system":
                    system = "\n\n".join(m["sections"].values())
                elif m.get("role") in ("user", "assistant", "toolResult"):
                    messages.append(m)
    return system, messages


def to_openai(system, messages):
    out = [{"role": "system", "content": system}]
    for m in messages:
        text = "".join(p.get("text", "") for p in m.get("content") or [] if isinstance(p, dict) and p.get("type") == "text")
        if m["role"] == "user":
            out.append({"role": "user", "content": text})
        elif m["role"] == "toolResult":
            out.append({"role": "tool", "tool_call_id": m["toolCallId"], "content": text})
        else:
            calls = [{"id": p["id"], "type": "function", "function": {
                "name": p["name"], "arguments": json.dumps(p.get("arguments"))}}
                for p in m.get("content") or [] if isinstance(p, dict) and p.get("type") == "toolCall"]
            out.append({"role": "assistant", "content": text, **({"tool_calls": calls} if calls else {})})
    return out


def find_point(path):
    system, messages = load(path)
    turns = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
    sigs = [sig(messages[i]) for i in turns]
    for t in range(4, len(turns)):
        for period in (1, 2):
            if t - 3 * period < 0 or not sigs[t]:
                continue
            if sigs[t] == sigs[t - period] == sigs[t - 2 * period] and \
                    (period == 1 or sigs[t - 1] == sigs[t - 3] and sigs[t] != sigs[t - 1]):
                prefix = to_openai(system, messages[:turns[t]])
                if sum(len(str(m["content"])) for m in prefix) > MAX_CHARS:
                    return None
                return {"run": "/".join(path.split("/")[-3:-1]), "turn": t, "period": period,
                        "expected": list(sigs[t]), "messages": prefix}
    return None


def points():
    found = (find_point(p) for p in sorted(glob.glob(f"{BENCH}/results/raw/lfm3vl/*/*/pi.jsonl")))
    return [p for p in found if p]


def ask(messages, sampling, seed):
    body = {"model": "x", "messages": messages, "tools": TOOLS, "max_tokens": 1200, "seed": seed, **sampling}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r)["choices"][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["find", "run"])
    ap.add_argument("name", nargs="?")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--samples", type=int, default=4)
    a = ap.parse_args()
    pts = points()
    if a.mode == "find":
        for p in pts:
            print(p["run"], "turn", p["turn"], "period", p["period"], len(p["messages"]), "messages",
                  sum(len(str(m["content"])) for m in p["messages"]), "chars", p["expected"][0][:90])
        return
    rows = []
    for p in pts:
        for arm in a.arms.split(","):
            hits, nocall, other = 0, 0, []
            for seed in range(a.samples):
                c = ask(p["messages"], ARMS[arm], seed + 1)
                calls = c["message"].get("tool_calls") or []
                got = sorted(x["function"]["name"] + json.dumps(json.loads(x["function"]["arguments"]), sort_keys=True)
                             for x in calls)
                if not got:
                    nocall += 1
                elif set(got) & set(p["expected"]):
                    hits += 1
                else:
                    other.append(got[0][:120])
            rows.append({"run": p["run"], "turn": p["turn"], "arm": arm, "repeats": hits, "no_call": nocall,
                         "new_call": len(other), "samples": a.samples, "new_examples": other[:2]})
            print(rows[-1], flush=True)
            with open(f"{HERE}/results/replay-{a.name}.json", "w") as f:
                json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
