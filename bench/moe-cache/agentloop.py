#!/usr/bin/env python3
"""Pi-like agent loop against llama-server: echo each assistant turn back (reasoning_content + tool_calls
with JSON arguments, as pi does), add a fake tool result, and record how many prompt tokens the server
reprocesses each turn (timings.prompt_n) vs reuses (timings.cache_n)."""
import json
import os
import sys
import urllib.request

tools = [
    {"type": "function", "function": {"name": "read", "description": "Read a file",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "bash", "description": "Run a shell command",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}, "timeout": {"type": "integer"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "write", "description": "Write a file",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
]
msgs = [
    {"role": "system", "content": "You are a coding agent. Use the tools to inspect and fix the project. Call one tool at a time."},
    {"role": "user", "content": os.environ.get("TASK", "The tests in /work fail. Find out why with the tools (start by listing files, then read them), then fix it.")},
]
if os.environ.get("PI_CAPTURE"):
    with open(os.environ["PI_CAPTURE"]) as f:
        cap = json.load(f)
    cap = cap.get("body", cap)
    tools = cap["tools"]; msgs[0] = cap["messages"][0]
    print(f"pi capture: system prompt {len(msgs[0]['content'])} chars, {len(tools)} tools", flush=True)
# divergence.py, toklcp.py and toolsdiff.py import `tools` and `msgs` from here; the loop runs only as a script.
fake = {
    "bash": "inventory.py\ntest_inventory.py\nREADME.md\n",
    "read": "def total(items):\n    return sum(i['qty'] * i['price'] for i in items)\n\ndef restock(items, name, n):\n    for i in items:\n        if i['name'] == name:\n            i['qty'] =+ n\n    return items\n",
    "write": "Successfully wrote file",
}

def post(body):
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=1800))

URL = "http://127.0.0.1:8090"


def main():
    global URL
    URL = sys.argv[1] if len(sys.argv) > 1 else URL
    turns = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    print(f"{'turn':>4} {'prompt_tok':>10} {'cached':>7} {'reproc':>7} {'gen':>5} {'think_chars':>11}  tool")
    for t in range(turns):
        r = post({"messages": msgs, "tools": tools, "temperature": 0.6, "seed": 1 + t, "max_tokens": int(os.environ.get("MAXTOK", "3000"))})
        m = r["choices"][0]["message"]; tm = r.get("timings", {})
        tc = m.get("tool_calls") or []
        print(f"{t:>4} {tm.get('cache_n', 0) + tm.get('prompt_n', 0):>10} {tm.get('cache_n'):>7} {tm.get('prompt_n'):>7} {tm.get('predicted_n'):>5} {len(m.get('reasoning_content') or ''):>11}  {tc[0]['function']['name'] if tc else '-'}", flush=True)
        a = {"role": "assistant", "content": m.get("content")}
        if m.get("reasoning_content"):
            a["reasoning_content"] = m["reasoning_content"]
        if tc:
            a["tool_calls"] = [{"id": c["id"], "type": "function", "function": {"name": c["function"]["name"], "arguments": c["function"]["arguments"]}} for c in tc]
        msgs.append(a)
        if t == int(os.environ.get("NEWUSER_AT", "-1")):
            for c in tc:
                msgs.append({"role": "tool", "tool_call_id": c["id"], "content": fake.get(c["function"]["name"], "Successfully wrote file" if "content" in c["function"]["arguments"] else "inventory.py\ntest_inventory.py\n")})
            msgs.append({"role": "user", "content": "Good. Now also add a docstring to every function and write the file again."})
            print("     (new user message)")
            continue
        if not tc:
            msgs.append({"role": "user", "content": "Continue."})
            continue
        for c in tc:
            msgs.append({"role": "tool", "tool_call_id": c["id"], "content": fake.get(c["function"]["name"], "Successfully wrote file" if "content" in c["function"]["arguments"] else "inventory.py\ntest_inventory.py\n")})


if __name__ == "__main__":
    main()
