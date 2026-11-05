#!/usr/bin/env python3
"""Agent-turn prefill cost: a cached base prompt (~2K tokens) + N new tokens, for N in SIZES. Reports the
server's prompt_n / prompt_ms for the new tokens only (median of REPS). The base is re-sent between sizes, so
the cache always holds exactly the base.

    turnbench.py [URL]    env: SIZES="16 48 100 200 400 800 1500 2048" REPS=3 ADV_LLAMA=<fork checkout>

The text is the fork's tools/server/server-context.cpp (at moe-offload 3a72c57be), split into words.
"""
import json
import os
import statistics
import sys
import urllib.request

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8090"
SIZES = [int(x) for x in os.environ.get("SIZES", "16 48 100 200 400 800 1500 2048").split()]
REPS = int(os.environ.get("REPS", "3"))
LLAMA = os.environ.get("ADV_LLAMA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../llama.cpp"))


def post(path, body):
    r = urllib.request.Request(URL + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=3600) as f:
        return json.load(f)


def tok(text):
    return post("/tokenize", {"content": text})["tokens"]


def main():
    with open(os.path.join(LLAMA, "tools/server/server-context.cpp")) as f:
        words = f.read().split()
    base = tok(" ".join(words[:1500]))[:2000]
    pool = tok(" ".join(words[1500:12000]))
    print(f"base {len(base)} tokens; new-token sizes {SIZES}, {REPS} reps")
    print(f"{'new':>5} {'prompt_n':>8} {'ms':>8} {'t/s':>7}")
    for n in SIZES:
        ms = []
        for k in range(REPS):
            post("/completion", {"prompt": base, "n_predict": 1, "cache_prompt": True})
            off = (k * 977) % max(1, len(pool) - n)
            t = post("/completion", {"prompt": base + pool[off:off + n], "n_predict": 1, "cache_prompt": True})["timings"]
            ms.append((t["prompt_n"], t["prompt_ms"]))
        pn = statistics.median(x[0] for x in ms)
        pm = statistics.median(x[1] for x in ms)
        print(f"{n:>5} {pn:>8.0f} {pm:>8.0f} {1000 * pn / pm:>7.1f}", flush=True)


if __name__ == "__main__":
    main()
