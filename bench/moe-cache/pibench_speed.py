#!/usr/bin/env python3
"""Model speed inside real pibench sessions, from pi's own per-call records.

    python3 bench/moe-cache/pibench_speed.py bench/results/raw bench/results/raw-moecache qwen36 flint gemma26b

Every assistant message in pi.jsonl carries usage (input = new prompt tokens, output) and durationMs. For each
batch folder and model this prints:
  - calls, total output tokens, total model time;
  - output tokens per second of model time (all calls);
  - a least-squares fit durationMs = a*input + b*output + c, i.e. prefill and decode speed as pi saw them;
  - decode speed of the "short-prompt" calls (input <= 256 new tokens), the bulk of an agent session.
"""
import glob
import json
import os
import sys

import numpy as np


def calls(raw, model):
    out = []
    for f in sorted(glob.glob(os.path.join(raw, model, "*", "r*", "pi.jsonl"))):
        with open(f) as fh:
            for line in fh:
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                m = e.get("message") or {}
                if e.get("type") != "message_end" or m.get("role") != "assistant":
                    continue
                u, d = m.get("usage") or {}, m.get("durationMs")
                if not d or not u.get("output"):
                    continue
                out.append((u.get("input", 0), u["output"], d / 1000.0))
    return np.array(out, dtype=float)


def report(raw, model):
    a = calls(raw, model)
    if not len(a):
        print(f"{model:10s} {raw}: no calls")
        return
    inp, outp, sec = a[:, 0], a[:, 1], a[:, 2]
    coef, *_ = np.linalg.lstsq(np.column_stack([inp, outp, np.ones_like(inp)]), sec, rcond=None)
    short = inp <= 256
    print(f"{model:10s} {os.path.basename(raw.rstrip('/')):14s} calls {len(a):4d}  out {int(outp.sum()):7d} tok"
          f"  time {sec.sum() / 60:6.1f} min  overall {outp.sum() / sec.sum():5.1f} tok/s"
          f"  | fit: prefill {1 / coef[0]:6.0f} tok/s, decode {1 / coef[1]:5.1f} tok/s, {coef[2]:4.2f} s/call"
          f"  | short calls {short.sum():4d}: {outp[short].sum() / sec[short].sum():5.1f} tok/s")


def main():
    args = sys.argv[1:]
    raws = [x for x in args if os.path.isdir(x)]
    models = [x for x in args if not os.path.isdir(x)]
    for model in models:
        for raw in raws:
            report(raw, model)
        print()


if __name__ == "__main__":
    main()
