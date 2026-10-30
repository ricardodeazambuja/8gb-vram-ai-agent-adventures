#!/usr/bin/env python3
"""Repo search probe: can an embedding model find the doc that answers a question about this repo?

    python3 bench/retrieval/run_retrieval.py embed KEY     # needs an embedding llama-server on :8080
    python3 bench/retrieval/run_retrieval.py bm25          # keyword baseline, no server
    python3 bench/retrieval/run_retrieval.py summary

The corpus is docs/*.md and docs/models/*.md, split at "## " headings (one chunk per section,
heading included, long sections cut to CHUNK characters). Each question below is phrased the way
a user would ask it, mostly without the doc's own keywords, and lists the file(s) that answer it.
A question is a hit@k if one of its files is among the k best chunks' files.

Only text goes to the server and only vectors come back: nothing is run.
"""
import argparse
import collections
import glob
import json
import math
import os
import re
import statistics
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
RES = os.path.join(HERE, "results")
URL = "http://127.0.0.1:8080/v1/embeddings"
CHUNK = 2000

# (question, files that answer it); paths relative to docs/
QUESTIONS = [
    ("Why did my whole terminal window close while a big model was loading?", ["troubleshooting.md"]),
    ("How do I keep the MoE expert weights in system RAM instead of on the GPU?", ["how-it-fits.md"]),
    ("The sandbox could reach my home network after I rebooted. What happened?", ["security.md", "pibench.md"]),
    ("Which model squeezes a 27B network into about 6 GB with three-valued weights?",
     ["models/bonsai2-27b.md", "how-it-fits.md"]),
    ("How are the image tests graded, and what pictures do they use?", ["vision-probe.md"]),
    ("The model only prints <unused32> tokens over and over.", ["troubleshooting.md"]),
    ("My Hugging Face download is stuck and shows no progress.", ["troubleshooting.md"]),
    ("Killing the server with pkill also killed my own shell.", ["troubleshooting.md"]),
    ("Which model should I use for long agent sessions with 128K context?", ["scoreboard.md", "models/flint.md"]),
    ("Why is the dense 27B Qwen not in the lineup?", ["graveyard.md"]),
    ("Does speculative decoding with a draft model help Gemma 4?", ["graveyard.md"]),
    ("Why not use Vulkan instead of CUDA?", ["graveyard.md", "how-it-fits.md"]),
    ("What does the agent benchmark ask the models to do?", ["pibench.md"]),
    ("What separated the agent runs that passed from the ones that failed?", ["pibench.md"]),
    ("How can a downloaded GGUF file attack me?", ["security.md"]),
    ("What hidden characters should I scan my commits for?", ["security.md"]),
    ("Can a small vision model watch a dial and open a valve by calling a tool?",
     ["visual-control.md", "models/lfm3-vl-2b.md"]),
    ("Does forcing a tool call with tool_choice required guarantee the model calls a tool?", ["visual-control.md"]),
    ("Bounding boxes from Gemma land in the wrong place on the image.", ["troubleshooting.md", "vision-probe.md"]),
    ("Which model can change how many times it loops over its layers?", ["models/dory1.0-4b.md"]),
    ("Spark returns HTTP 400 about the reasoning effort.", ["troubleshooting.md", "models/spark-x2.6-4b.md"]),
    ("Why does the server die with CUDA out of memory on the first request even though it loaded?",
     ["troubleshooting.md", "models/gemma4-12b-agentic.md"]),
    ("How much memory does the KV cache take, and should I quantize it?", ["how-it-fits.md"]),
    ("Which models are waiting to be tested next?", ["up-next.md"]),
    ("Could we fine-tune the small vision model with LoRA on this laptop?", ["up-next.md"]),
    ("Is there a model that answers a fixed set of questions with probabilities instead of generating text?",
     ["up-next.md"]),
    ("Can I talk to an agent with my voice and hear it answer?", ["up-next.md"]),
    ("What do I need to install before running the first model?", ["getting-started.md"]),
    ("Which 2B model scored zero on the agent benchmark because it kept repeating a command?",
     ["models/lfm3-vl-2b.md", "pibench.md"]),
    ("How fast is the 74B Nemotron and how many bits per weight does it use?",
     ["models/nemotron-bolt-74b.md", "scoreboard.md"]),
]


def chunks():
    out = []
    for path in sorted(glob.glob(os.path.join(REPO, "docs", "*.md")) + glob.glob(os.path.join(REPO, "docs", "models", "*.md"))):
        rel = os.path.relpath(path, os.path.join(REPO, "docs"))
        with open(path, encoding="utf-8") as f:
            text = f.read()
        title = text.splitlines()[0].lstrip("# ").strip()
        for part in re.split(r"(?m)^(?=## )", text):
            part = part.strip()
            if len(part) > 40:
                out.append({"file": rel, "title": title, "text": part[:CHUNK]})
    return out


def post(texts):
    req = urllib.request.Request(URL, json.dumps({"input": texts}).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    return [x["embedding"] for x in sorted(d["data"], key=lambda x: x["index"])]


def norm(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def score(ranked_files):
    """ranked_files: per question, the chunk files best first."""
    hits = {1: 0, 3: 0, 5: 0}
    rr = []
    per_q = []
    for (q, want), files in zip(QUESTIONS, ranked_files):
        first = next((i for i, f in enumerate(files) if f in want), None)
        rr.append(0 if first is None else 1 / (first + 1))
        for k in hits:
            hits[k] += first is not None and first < k
        per_q.append({"q": q, "want": want, "top": files[:5], "rank": None if first is None else first + 1})
    n = len(QUESTIONS)
    return {"hit@1": hits[1] / n, "hit@3": hits[3] / n, "hit@5": hits[5] / n, "mrr": statistics.mean(rr),
            "questions": per_q}


def run_embed(key, dims):
    cs = chunks()
    t = time.time()
    dv = []
    for i in range(0, len(cs), 16):
        dv += post([f"title: {c['title']} | text: {c['text']}" for c in cs[i:i + 16]])
    t_docs = time.time() - t
    t = time.time()
    qv = post([f"task: search result | query: {q}" for q, _ in QUESTIONS])
    t_q = time.time() - t
    out = {"test": "retrieval", "key": key, "chunks": len(cs), "dim": len(dv[0]),
           "secs_docs": round(t_docs, 2), "secs_queries": round(t_q, 2), "by_dim": {}}
    bm = bm25_orders(cs)
    for d in dims:  # Matryoshka: the first d components, renormalized
        D = [norm(v[:d]) for v in dv]
        emb_orders = []
        for v in qv:
            v = norm(v[:d])
            sims = [sum(a * b for a, b in zip(v, x)) for x in D]
            emb_orders.append(sorted(range(len(cs)), key=lambda i: -sims[i]))
        for name, orders in ((str(d), emb_orders), (f"{d}+bm25", [rrf(e, b) for e, b in zip(emb_orders, bm)])):
            s = out["by_dim"][name] = score([[cs[i]["file"] for i in o] for o in orders])
            print(f"{name:>9s}: hit@1 {s['hit@1']:.0%}  hit@3 {s['hit@3']:.0%}  hit@5 {s['hit@5']:.0%}  MRR {s['mrr']:.2f}")
    print(f"{len(cs)} chunks embedded in {t_docs:.1f} s ({len(cs) / t_docs:.1f}/s); {len(QUESTIONS)} queries in {t_q:.2f} s")
    return out


def tokens(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def bm25_orders(cs, k1=1.5, b=0.75):
    """Per question, the chunk indices best first by BM25."""
    docs = [tokens(c["title"] + " " + c["text"]) for c in cs]
    avg = statistics.mean(len(d) for d in docs)
    df = collections.Counter(t for d in docs for t in set(d))
    N = len(docs)
    tfs = [collections.Counter(d) for d in docs]
    orders = []
    for q, _ in QUESTIONS:
        sc = []
        for d, tf in zip(docs, tfs):
            s = 0.0
            for t in set(tokens(q)):
                if t in tf:
                    idf = math.log(1 + (N - df[t] + 0.5) / (df[t] + 0.5))
                    s += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(d) / avg))
            sc.append(s)
        orders.append(sorted(range(N), key=lambda i: -sc[i]))
    return orders


def run_bm25():
    cs = chunks()
    s = score([[cs[i]["file"] for i in o] for o in bm25_orders(cs)])
    print(f"BM25: hit@1 {s['hit@1']:.0%}  hit@3 {s['hit@3']:.0%}  hit@5 {s['hit@5']:.0%}  MRR {s['mrr']:.2f}")
    return {"test": "retrieval", "key": "bm25", "chunks": len(cs), "by_dim": {"bm25": s}}


def rrf(*orders, k=60):
    """Reciprocal rank fusion of several rankings of the same items."""
    sc = collections.Counter()
    for o in orders:
        for r, i in enumerate(o):
            sc[i] += 1 / (k + r + 1)
    return sorted(sc, key=lambda i: -sc[i])


def summary():
    lines = ["# Repo search probe: results", "",
             f"{len(QUESTIONS)} questions, corpus docs/*.md + docs/models/*.md split at '## '.", "",
             "| Run | Dims (+bm25 = fused with BM25) | hit@1 | hit@3 | hit@5 | MRR | Chunks/s |", "|---|---:|---:|---:|---:|---:|---:|"]
    for p in sorted(glob.glob(os.path.join(RES, "*.json"))):
        with open(p) as f:
            d = json.load(f)
        speed = f"{d['chunks'] / d['secs_docs']:.0f}" if d.get("secs_docs") else "—"
        for dim, s in d["by_dim"].items():
            lines.append(f"| {d['key']} | {dim} | {s['hit@1']:.0%} | {s['hit@3']:.0%} | {s['hit@5']:.0%} | "
                         f"{s['mrr']:.2f} | {speed} |")
    with open(os.path.join(RES, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("test", choices=["embed", "bm25", "summary"])
    ap.add_argument("key", nargs="?", default="model")
    ap.add_argument("--dims", default="768,512,256,128")
    a = ap.parse_args()
    if a.test == "summary":
        return summary()
    os.makedirs(RES, exist_ok=True)
    out = run_bm25() if a.test == "bm25" else run_embed(a.key, [int(x) for x in a.dims.split(",")])
    key = "bm25" if a.test == "bm25" else a.key
    with open(os.path.join(RES, f"{key}.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
