#!/usr/bin/env python3
"""Thinking off vs on: the rendered prompt's tail, the raw generation's head, and where the template's
re-render of that turn first differs from what the model generated (temp 0, so deterministic).

    PI_CAPTURE=../gemma12-probe/capture/agentic12/0001-req.json TASK="..." python3 divergence2.py

Found ThinkingCap's empty-thought bug (logs/tcap-step2c.log, fixed in logs/tcap-step2d.log).
"""
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agentloop import (  # pi's system prompt and tools with PI_CAPTURE, the TASK as user turn
    msgs,
    tools,
)

U = "http://127.0.0.1:8090"


def post(path, body):
    r = urllib.request.Request(U + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=3600))


def main():
    for think in (False, True):
        kw = {"chat_template_kwargs": {"enable_thinking": think}}
        p0 = post("/apply-template", {"messages": msgs, "tools": tools, **kw})["prompt"]
        raw = post("/completion", {"prompt": p0, "n_predict": 1500, "temperature": 0, "special": True})["content"]
        m = post("/v1/chat/completions", {"messages": msgs, "tools": tools, "temperature": 0, "max_tokens": 1500, **kw})
        m = m["choices"][0]["message"]
        a = {"role": "assistant", "content": m.get("content")}
        if m.get("reasoning_content"):
            a["reasoning_content"] = m["reasoning_content"]
        if m.get("tool_calls"):
            a["tool_calls"] = m["tool_calls"]
        tid = (m.get("tool_calls") or [{"id": "x"}])[0]["id"]
        p1 = post("/apply-template", {"messages": [*msgs, a, {"role": "tool", "tool_call_id": tid, "content": "ok"}],
                                      "tools": tools, **kw})["prompt"]
        cached = p0 + raw
        n = min(len(cached), len(p1))
        i = next((k for k in range(n) if cached[k] != p1[k]), n)
        print(f"== enable_thinking={think}")
        print("PROMPT TAIL:", repr(p0[-80:]))
        print("RAW HEAD   :", repr(raw[:160]))
        print("parsed: reasoning chars", len(m.get("reasoning_content") or ""), "content",
              repr((m.get("content") or "")[:80]), "tool", (m.get("tool_calls") or [{}])[0].get("function", {}).get("name"))
        print("raw gen len", len(raw), "first difference", i - len(p0), "chars into the generation")
        print("GENERATED :", repr(cached[max(len(p0), i - 60): i + 100]))
        print("RERENDERED:", repr(p1[max(len(p0), i - 60): i + 100]))


if __name__ == "__main__":
    main()
