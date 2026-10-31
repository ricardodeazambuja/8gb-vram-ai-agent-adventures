#!/usr/bin/env python3
"""Relaxed draft acceptance on Gemma 4 26B + its MTP head: does it speed things up, and what breaks?

One arm = one server. LLAMA_SPEC_RELAX=tau (llama.cpp-relaxed build) keeps a drafted token when the
target gives it at least tau times the probability of its top choice. Each arm answers the same
seeded tasks (arithmetic, sorting, JSON, tool calls) and a timed writing prompt. Model output is
only parsed, never executed.

Usage: run_relaxed.py --temp 1.0 [--rounds 2] [--arms none,exact,prob,0.5,0.2,0.05,none,exact]
"""
import argparse
import json
import os
import random
import re
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BIN = ROOT.parent / "llama.cpp-relaxed/build/bin/llama-server"
GEMMA = ROOT / "models/gemma4-26b-a4b-qat"
MODEL = GEMMA / "gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf"
HEAD = GEMMA / "mtp-gemma-4-26B-A4B-it.gguf"
PORT = 8090
URL = f"http://127.0.0.1:{PORT}"
WORDS = ["anchor", "basil", "cedar", "delta", "ember", "fable", "glider", "harbor", "island", "jasper", "kettle", "lantern", "meadow", "nectar", "orchid", "pepper", "quartz", "river", "saddle", "timber", "umbra", "velvet", "willow", "yarrow", "zephyr"]
TOOLS = [
    {"type": "function", "function": {"name": "read", "description": "Read a file",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "search", "description": "Search the code for a string",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "write", "description": "Write text to a file",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "text": {"type": "string"}},
                    "required": ["path", "text"]}}},
]
ESSAY = "Explain in about 400 words how a hash map handles collisions, with a short Python example."


def tasks(seed):
    r = random.Random(seed)
    out = []
    for _ in range(15):
        a, b, c = r.randint(12, 99), r.randint(12, 99), r.randint(100, 999)
        out.append(("arith", f"What is {a} * {b} + {c}? End with a line 'Answer: N'.", a * b + c))
    for _ in range(10):
        w = r.sample(WORDS, 7)
        prompt = f"Sort these words alphabetically and give them on one line, comma-separated, nothing else: {', '.join(w)}"
        out.append(("sort", prompt, sorted(w)))
    for _ in range(10):
        d = {"name": r.choice(WORDS), "port": r.randint(1000, 9999), "tags": r.sample(WORDS, 3), "on": r.random() < .5}
        prompt = (f"Output only this as one line of minified JSON with keys name, port, tags, on: name is {d['name']}, "
                  f"port is {d['port']}, tags are {' '.join(d['tags'])}, on is {str(d['on']).lower()}.")
        out.append(("json", prompt, d))
    for _ in range(5):
        p = f"src/{r.choice(WORDS)}.py"
        out.append(("tool", f"Show me what is in {p}.", ("read", {"path": p})))
        q = r.choice(WORDS) + "_handler"
        out.append(("tool", f"Find where {q} is used in the code.", ("search", {"query": q})))
        p, t = f"notes/{r.choice(WORDS)}.txt", r.choice(WORDS)
        out.append(("tool", f"Write the single word {t} to the file {p}.", ("write", {"path": p, "text": t})))
    return out


def ask(messages, seed, max_tokens=1500, tools=None):
    body = {"messages": messages, "seed": seed, "max_tokens": max_tokens, "timings_per_token": False}
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as f:
        return json.load(f)


def score(kind, want, msg):
    text = (msg.get("content") or "").strip()
    if kind == "arith":
        m = re.findall(r"Answer:\D*([\d,]+)", text)
        return bool(m) and int(m[-1].replace(",", "")) == want
    if kind == "sort":
        return [w.strip(" .`") for w in text.split(",")] == want
    if kind == "json":
        try:
            return json.loads(re.sub(r"^```\w*|```$", "", text).strip()) == want
        except ValueError:
            return False
    calls = msg.get("tool_calls") or []
    if len(calls) != 1:
        return False
    fn = calls[0]["function"]
    try:
        return fn["name"] == want[0] and json.loads(fn["arguments"]) == want[1]
    except ValueError:
        return False


def start(arm, temp):
    unit = "llama-relax"
    args = ["-m", MODEL, "--alias", "gemma", "-ngl", "99", "--n-cpu-moe", "27", "--load-mode", "none",
            "-c", "32768", "-ctk", "q8_0", "-ctv", "q8_0", "-fa", "on", "--jinja", "--temp", temp,
            "--top-p", "0.95", "--top-k", "64", "--host", "127.0.0.1", "--port", PORT]
    if arm != "none":
        args += ["--spec-draft-model", HEAD, "--spec-type", "draft-mtp", "--spec-draft-n-max", "2"]
    if arm == "prob":
        args += ["--spec-draft-sampling", "probabilistic"]
    env = ["-E", f"LLAMA_SPEC_RELAX={arm}"] if arm not in ("none", "exact", "prob") else []
    subprocess.run(["systemd-run", "--user", "-q", f"--unit={unit}", "-E", f"PATH={os.environ['PATH']}", *env,
                    "mamba", "run", "--no-capture-output", "-n", "llamacpp-cuda", str(BIN), *map(str, args)], check=True)
    for _ in range(200):
        try:
            urllib.request.urlopen(URL + "/health", timeout=2)
            return unit
        except (urllib.error.URLError, OSError):
            r = subprocess.run(["systemctl", "--user", "is-active", "-q", unit], check=False)
            if r.returncode:
                raise RuntimeError(f"{arm}: server died") from None
            time.sleep(2)
    raise RuntimeError(f"{arm}: server timed out")


def stop(unit):
    subprocess.run(["systemctl", "--user", "stop", unit], check=False)
    subprocess.run(["systemctl", "--user", "reset-failed", unit], check=False, capture_output=True)
    time.sleep(4)


def run_arm(arm, temp, rounds, first=0, kinds=None, max_tokens=1500):
    unit = start(arm, temp)
    res = {"arm": arm, "temp": temp, "ok": {}, "n": {}, "items": [], "speed": []}
    try:
        for rnd in range(first, first + rounds):
            for i, (kind, prompt, want) in enumerate(tasks(rnd)):
                if kinds and kind not in kinds:
                    continue
                r = ask([{"role": "user", "content": prompt}], 1000 * rnd + i, max_tokens,
                        TOOLS if kind == "tool" else None)
                msg = r["choices"][0]["message"]
                ok = score(kind, want, msg)
                res["ok"][kind] = res["ok"].get(kind, 0) + ok
                res["n"][kind] = res["n"].get(kind, 0) + 1
                res["items"].append({"kind": kind, "round": rnd, "i": i, "ok": ok, "content": msg.get("content"),
                                     "tool_calls": msg.get("tool_calls"), "finish": r["choices"][0]["finish_reason"],
                                     "tokens": r["usage"]["completion_tokens"]})
        for s in range(0 if kinds else 3):
            r = ask([{"role": "user", "content": ESSAY}], 7 + s, max_tokens=600)
            t = r["timings"]
            res["speed"].append({"tps": t["predicted_per_second"], "n": t["predicted_n"],
                                 "draft": t.get("draft_n", 0), "accepted": t.get("draft_n_accepted", 0)})
            res.setdefault("essay", []).append(r["choices"][0]["message"].get("content"))
    finally:
        stop(unit)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--temp", default="1.0")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--arms", default="none,exact,prob,0.5,0.2,0.05,none,exact")
    ap.add_argument("--first", type=int, default=0, help="first task seed")
    ap.add_argument("--kinds", help="only these task kinds, comma-separated; skips the timed prompt")
    ap.add_argument("--max-tokens", type=int, default=1500)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    outdir = Path(__file__).parent / "results"
    outdir.mkdir(exist_ok=True)
    for n, arm in enumerate(a.arms.split(",")):
        try:
            res = run_arm(arm, a.temp, a.rounds, a.first, a.kinds and a.kinds.split(","), a.max_tokens)
        except (RuntimeError, urllib.error.URLError, OSError) as e:
            print(f"== {arm}: FAILED {e}", flush=True)
            continue
        (outdir / f"t{a.temp}{a.tag}-{n}-{arm}.json").write_text(json.dumps(res, indent=1))
        sp = res["speed"]
        drafted, kept = sum(s["draft"] for s in sp), sum(s["accepted"] for s in sp)
        print(f"== {arm} t={a.temp}: {res['ok']} of {res['n']}; tok/s {[round(s['tps'], 1) for s in sp]}; "
              f"draft {kept}/{drafted}", flush=True)


if __name__ == "__main__":
    main()
