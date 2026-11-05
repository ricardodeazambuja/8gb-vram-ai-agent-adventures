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
p0 = post("/apply-template", {"messages": msgs, "tools": tools})["prompt"]
# what the model generates, raw (temp 0)
raw = post("/completion", {"prompt": p0, "n_predict": 400, "temperature": 0, "special": True})["content"]
# the same turn through the chat API (parsed), temp 0
m = post("/v1/chat/completions", {"messages": msgs, "tools": tools, "temperature": 0, "max_tokens": 400})["choices"][0]["message"]
a = {"role": "assistant", "content": m.get("content")}
if m.get("reasoning_content"): a["reasoning_content"] = m["reasoning_content"]
if m.get("tool_calls"): a["tool_calls"] = m["tool_calls"]
tid = (m.get("tool_calls") or [{"id": "x"}])[0]["id"]
p1 = post("/apply-template", {"messages": msgs + [a, {"role": "tool", "tool_call_id": tid, "content": "ok"}], "tools": tools})["prompt"]
cached = p0 + raw
i = next((k for k in range(min(len(cached), len(p1))) if cached[k] != p1[k]), min(len(cached), len(p1)))
print("prompt0 len", len(p0), "raw gen len", len(raw), "first difference at char", i, f"({i - len(p0)} chars into the generation)")
print("GENERATED :", repr(cached[max(len(p0), i - 60): i + 120]))
print("RERENDERED:", repr(p1[max(len(p0), i - 60): i + 120]))
