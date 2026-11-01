#!/usr/bin/env python3
"""Strength sweep for a llama.cpp control vector: same questions, several strengths, answer length and text.

Usage: sweep.py --model M.gguf --vector V.gguf --name concise [--scales -2,-1,0,1,2] [--ngl 0]
Results go to bench/steering/results/<name>.json. Runs on the CPU by default so it can share the box.
"""
import argparse
import json
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SERVER = ROOT.parent / "llama.cpp/build/bin/llama-server"
Q = ["Why do cats purr?", "How do I undo my last git commit?", "What is a hash table?", "Why does ice float?",
     "What's the difference between TCP and UDP?", "How do vaccines work?", "Name the capital of Australia.",
     "What is 17 times 23?"]
ANSWERS = {"Name the capital of Australia.": "canberra", "What is 17 times 23?": "391"}
PORT = 8097


def ask(q):
    body = {"messages": [{"role": "user", "content": q}], "max_tokens": 300, "temperature": 0, "seed": 1,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=300))
    m = d["choices"][0]["message"]
    return (m.get("content") or "") + (m.get("reasoning_content") or ""), d["timings"]["predicted_n"], \
        d["timings"]["predicted_per_second"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--vector", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--scales", default="-2,-1,-0.5,0,0.5,1,2")
    ap.add_argument("--ngl", default="0")
    a = ap.parse_args()
    rows = []
    for s in map(float, a.scales.split(",")):
        cv = [] if s == 0 else ["--control-vector-scaled", f"{a.vector}:{s}"]
        p = subprocess.Popen([SERVER, "-m", a.model, "-ngl", a.ngl, "-c", "2048", "-t", "4", "--port", str(PORT),
                              "--host", "127.0.0.1", *cv], stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
        try:
            for _ in range(120):
                try:
                    urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
                    break
                except OSError:
                    time.sleep(1)
            out = [(q, *ask(q)) for q in Q]
        finally:
            p.terminate()
            p.wait()
        right = sum(ANSWERS[q].lower() in t.lower() for q, t, *_ in out if q in ANSWERS)
        n = sum(o[2] for o in out) / len(out)
        print(f"scale {s:+}: mean {n:.0f} tokens, {right}/{len(ANSWERS)} facts right, {out[0][3]:.0f} t/s", flush=True)
        print("   ", out[1][1][:150].replace("\n", " "), flush=True)
        rows.append({"scale": s, "mean_tokens": round(n, 1), "facts_right": right, "tps": round(out[0][3], 1),
                     "answers": [{"q": q, "text": t, "tokens": k} for q, t, k, _ in out]})
    r = Path(__file__).parent / "results"
    r.mkdir(exist_ok=True)
    (r / f"{a.name}.json").write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
