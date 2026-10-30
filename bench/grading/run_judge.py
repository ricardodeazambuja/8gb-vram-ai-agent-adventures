#!/usr/bin/env python3
"""Judge probe: can a decision model tell, from an agent's transcript, whether it solved the chore?

    python3 bench/grading/run_judge.py judge KEY [--max-chars 6000]   # llama-server with /v1/systemone on :8080
    python3 bench/grading/run_judge.py summary

Every finished pibench attempt (bench/results/raw/MODEL/TASK/rN) becomes one state: the task's
prompt, then a condensed transcript (each tool call with a short view of its arguments and of its
output, then the agent's last message), cut from the front to --max-chars. The model answers one
question, "did the agent fully solve the task?", with a probability. The truth is the hidden
tests (score.json "solved"), which the judge never sees.

Baseline to beat: "solved unless it timed out" is right for 204 of the 252 attempts.

Only text goes to the server and probabilities come back: nothing from the transcripts is run.
"""
import argparse
import glob
import json
import os
import statistics
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
RES = os.path.join(HERE, "results")
URL = "http://127.0.0.1:8080/v1/systemone"

QUESTION = {"solved": {
    "type": "choice",
    "instructions": "A coding agent was given the task at the top and worked on it alone; its transcript "
                    "follows. Hidden tests will check its work strictly. Did it fully and correctly solve "
                    "the task, so that every test passes?",
    "criteria": {"yes": "the work is complete and correct; the agent checked real outputs and they match "
                        "what the task asks",
                 "no": "something is missing, wrong, unchecked, or the agent ran out of time or gave up"}}}


def short(s, n):
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 1] + "…"


def condensed(run):
    calls, final = [], ""
    with open(os.path.join(run, "pi.jsonl"), encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = ev.get("type")
            if t == "message_end" and (ev.get("message") or {}).get("role") == "assistant":
                for c in ev["message"].get("content") or []:
                    if c.get("type") == "toolCall":
                        a = c.get("arguments") or {}
                        view = a.get("command") or a.get("path") or json.dumps(a)
                        calls.append(f"> {c.get('name')}: {short(view, 200)}")
                    elif c.get("type") == "text" and c.get("text", "").strip():
                        final = c["text"].strip()
            elif t == "tool_execution_end":
                out = "".join(x.get("text", "") for x in (ev.get("result") or {}).get("content") or [])
                calls.append(f"  {'ERROR ' if ev.get('isError') else ''}{short(out, 300)}")
    return "\n".join(calls), final


def runs():
    for score in sorted(glob.glob(os.path.join(BENCH, "results", "raw", "*", "*", "r*", "score.json"))):
        run = os.path.dirname(score)
        if os.path.exists(os.path.join(run, "pi.jsonl")):
            with open(score) as f:
                yield run, json.load(f)


def state_for(run, sc, max_chars):
    with open(os.path.join(BENCH, "tasks", sc["task"], "prompt.md"), encoding="utf-8") as f:
        task = f.read().strip()
    calls, final = condensed(run)
    end = f"\n\nAgent's last message:\n{short(final, 1500) or '(none)'}"
    room = max(0, max_chars - len(task) - len(end) - 40)
    body = calls if len(calls) <= room else "…\n" + calls[-room:]
    return f"Task:\n{task}\n\nTranscript:\n{body}{end}"


def post(body):
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r), time.time() - t


def auc(pos, neg):
    """Probability that a random solved attempt gets a higher score than a random failed one."""
    if not pos or not neg:
        return float("nan")
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def judge(key, max_chars):
    items = []
    for run, sc in runs():
        st = state_for(run, sc, max_chars)
        try:
            out, dt = post({"state": st, "questions": QUESTION})
            p = out["answers"]["solved"]["probabilities"]["yes"]
            err = None
        except Exception as e:  # noqa: BLE001
            p, dt, err, out = None, 0.0, f"{type(e).__name__}: {e}", {}
        rel = os.path.relpath(run, os.path.join(BENCH, "results", "raw"))
        items.append({"run": rel, "solved": bool(sc.get("solved")), "timed_out": bool(sc.get("timed_out")),
                      "p_yes": p, "secs": round(dt, 3), "chars": len(st), "error": err,
                      "tokens": (out.get("usage") or {}).get("input_tokens")})
        print(f"{rel:40s} solved={sc.get('solved')!s:5s} p={p if p is None else round(p, 2)} {dt:.2f}s"
              + (f" {err}" if err else ""), flush=True)
    return {"test": "judge", "key": key, "max_chars": max_chars, "items": items}


def stats(d):
    it = [x for x in d["items"] if x["p_yes"] is not None]
    pos = [x["p_yes"] for x in it if x["solved"]]
    neg = [x["p_yes"] for x in it if not x["solved"]]
    acc = statistics.mean((x["p_yes"] > 0.5) == x["solved"] for x in it)
    caught = sum(1 for x in it if not x["solved"] and not x["timed_out"] and x["p_yes"] <= 0.5)
    quiet_fails = sum(1 for x in it if not x["solved"] and not x["timed_out"])
    false_alarm = sum(1 for x in it if x["solved"] and x["p_yes"] <= 0.5)
    return {"n": len(it), "errors": len(d["items"]) - len(it), "acc": acc, "auc": auc(pos, neg),
            "caught": caught, "quiet_fails": quiet_fails, "false_alarm": false_alarm, "solved_n": len(pos),
            "median_s": statistics.median(x["secs"] for x in it) if it else float("nan")}


def summary():
    lines = ["# Judge probe: results", "",
             ("Each finished pibench attempt; truth = the hidden tests. Baseline \"solved unless it timed out\": "
              "204/252 = 81%, and it catches none of the 47 failures that didn't time out."), "",
             "| Judge | Chars | Attempts | Accuracy (p>0.5) | AUC | Quiet failures caught | Solved runs flagged | Median time |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for p in sorted(glob.glob(os.path.join(RES, "*.json"))):
        with open(p) as f:
            d = json.load(f)
        s = stats(d)
        lines.append(f"| {d['key']} | {d['max_chars']} | {s['n']} | {s['acc']:.0%} | {s['auc']:.2f} | "
                     f"{s['caught']}/{s['quiet_fails']} | {s['false_alarm']}/{s['solved_n']} | {s['median_s']:.2f} s |")
    with open(os.path.join(RES, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("test", choices=["judge", "summary", "show"])
    ap.add_argument("key", nargs="?", default="model")
    ap.add_argument("--max-chars", type=int, default=6000)
    a = ap.parse_args()
    if a.test == "summary":
        return summary()
    if a.test == "show":  # print one state, to check what the judge sees
        run, sc = next(runs())
        return print(state_for(run, sc, a.max_chars))
    os.makedirs(RES, exist_ok=True)
    out = judge(a.key, a.max_chars)
    with open(os.path.join(RES, f"{a.key}-{a.max_chars}.json"), "w") as f:
        json.dump(out, f, indent=1)
    summary()


if __name__ == "__main__":
    main()
