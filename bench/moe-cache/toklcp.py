import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agentloop import (  # the agent loop's tools and opening messages (PI_CAPTURE applies)
    msgs,
    tools,
)

U = "http://127.0.0.1:8090"
def post(path, body):
    r = urllib.request.Request(U + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=3600))
for seed in range(3):
    p0 = post("/apply-template", {"messages": msgs, "tools": tools})["prompt"]
    t0 = post("/tokenize", {"content": p0, "add_special": True, "parse_special": True})["tokens"]
    g = post("/completion", {"prompt": t0, "n_predict": 400, "temperature": 0.6, "seed": seed, "special": True, "return_tokens": True})
    gen_tok, raw = g["tokens"], g["content"]
    cached = t0 + gen_tok
    retok = post("/tokenize", {"content": p0 + raw, "add_special": True, "parse_special": True})["tokens"]
    i = next((k for k in range(min(len(cached), len(retok))) if cached[k] != retok[k]), min(len(cached), len(retok)))
    pieces = lambda ts: [post("/detokenize", {"tokens": [t]})["content"] for t in ts]
    print(f"seed {seed}: prompt {len(t0)} tok, generated {len(gen_tok)}, re-tokenized {len(retok)-len(t0)}; first token mismatch at {i} ({i-len(t0)} into gen)")
    if i < len(cached):
        print("   generated :", pieces(cached[i:i+6])); print("   retokenized:", pieces(retok[i:i+6]))
