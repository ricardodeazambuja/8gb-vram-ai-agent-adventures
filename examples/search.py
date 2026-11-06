#!/usr/bin/env python3
"""Search a folder of text or code by meaning, with EmbeddingGemma 2.

    python3 examples/search.py "how do I make a big model fit in 8 GB"            searches docs/
    python3 examples/search.py "where is the KV cache size chosen" --dir scripts --ext .sh
    python3 examples/search.py "parse tool calls from the reply" --dir bench --ext .py --code

Splits each file into chunks, embeds them (768 numbers each), embeds the question, and prints the
closest chunks by cosine similarity. The prompts are EmbeddingGemma's own: documents as
"title: ... | text: ...", questions as "task: search result | query: ..." (or "code retrieval" with
--code). --dims 256 keeps the first 256 numbers, which lost nothing in our test
(docs/models/embeddinggemma-2.md); mixing in keyword search (BM25) did better still.
"""
import argparse
import math
from pathlib import Path

import _server


def chunks(text, size):
    """Paragraph-ish chunks of about `size` characters, with their first line number."""
    out, cur, start, line = [], [], 1, 1
    paras = []  # (text, lines it advances); a long paragraph (a big table) is cut into pieces of `size`
    for para in text.split("\n\n"):
        pieces = [para[i:i + size] for i in range(0, len(para), size)] or [""]
        paras += [(p, p.count("\n")) for p in pieces[:-1]] + [(pieces[-1], pieces[-1].count("\n") + 2)]
    for para, advance in paras:
        if cur and sum(len(p) for p in cur) + len(para) > size:
            out.append((start, "\n\n".join(cur)))
            cur, start = [], line
        cur.append(para)
        line += advance
    if cur:
        out.append((start, "\n\n".join(cur)))
    return out


def embed(texts, dims):
    vecs = []
    for i in range(0, len(texts), 32):
        r = _server.post("/v1/embeddings", {"input": texts[i:i + 32]})
        for d in sorted(r["data"], key=lambda d: d["index"]):
            v = d["embedding"][:dims]
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            vecs.append([x / n for x in v])
    return vecs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("query")
    ap.add_argument("--dir", default="docs", help="folder to search (default docs/)")
    ap.add_argument("--ext", default=".md", help="file extension(s), comma-separated (default .md)")
    ap.add_argument("--code", action="store_true", help="use the code-retrieval query prompt")
    ap.add_argument("--dims", type=int, default=768, choices=[128, 256, 512, 768])
    ap.add_argument("-k", type=int, default=5, help="how many results")
    ap.add_argument("--chunk", type=int, default=1200, help="chunk size in characters")
    _server.add_server_args(ap)
    a = ap.parse_args()
    exts = tuple(a.ext.split(","))
    files = [p for p in sorted(Path(a.dir).rglob("*")) if p.is_file() and p.suffix in exts]
    if not files:
        raise SystemExit(f"no {a.ext} files under {a.dir}")
    _server.ensure("eg2", a)

    items = [(p, ln, txt) for p in files for ln, txt in chunks(p.read_text(errors="replace"), a.chunk)]
    docs = [f"title: {p.name} | text: {txt}" for p, _, txt in items]
    print(f"[embedding {len(items)} chunks from {len(files)} files ...]")
    dv = embed(docs, a.dims)
    task = "code retrieval" if a.code else "search result"
    (qv,) = embed([f"task: {task} | query: {a.query}"], a.dims)
    scored = sorted(((sum(x * y for x, y in zip(qv, v)), i) for i, v in enumerate(dv)), reverse=True)[:a.k]
    for s, i in scored:
        p, ln, txt = items[i]
        snippet = " ".join(txt.split())[:160]
        print(f"\n{s:.3f}  {p}:{ln}\n       {snippet}")


if __name__ == "__main__":
    main()
