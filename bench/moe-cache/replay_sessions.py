#!/usr/bin/env python3
"""Replay pibench sessions as prompts, one after another, so a server started with LLAMA_MOE_CACHE_TRACE records
the experts the router picks on real agent work. The router's pick for a token depends only on the token and the
context before it, so reading a session back routes its tokens as they were routed when it ran, except where the
chat template renders the history differently (Qwen3.6's drops the thinking of earlier turns).

    replay_sessions.py OUT.json PI_JSONL...    env: URL (http://127.0.0.1:8090), PI_CAPTURE (system prompt + tools)

Writes, in order, each session's name and the number of prompt tokens the server read, which splits the trace.
With RENDER_DIR set, it only writes each rendered session to RENDER_DIR/TASK-RUN.txt (for llama-perplexity).
"""
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agentloop import msgs as capture_msgs  # pi's system prompt from PI_CAPTURE
from agentloop import tools  # and pi's tool definitions

URL = os.environ.get("URL", "http://127.0.0.1:8090")


def post(path, body):
    r = urllib.request.Request(URL + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=3600) as f:
        return json.load(f)


def text_of(content):
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content if b.get("type") == "text")


def messages(path):
    out = [capture_msgs[0]]
    with open(path) as f:
        rows = [json.loads(line) for line in f]
    for m in (r["message"] for r in rows if r["type"] == "message_end"):
        role, content = m.get("role"), m.get("content") or []
        if role == "user":
            out.append({"role": "user", "content": text_of(content)})
        elif role == "assistant":
            a = {"role": "assistant", "content": text_of(content)}
            thinking = "".join(b.get("thinking", "") for b in content if b.get("type") == "thinking")
            if thinking:
                a["reasoning_content"] = thinking
            calls = [b for b in content if b.get("type") == "toolCall"]
            if calls:
                a["tool_calls"] = [{"id": c["id"], "type": "function",
                                    "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}} for c in calls]
            out.append(a)
        elif role == "toolResult":
            out.append({"role": "tool", "tool_call_id": m["toolCallId"], "content": text_of(content)})
    return out


def main():
    done = []
    for path in sys.argv[2:]:
        prompt = post("/apply-template", {"messages": messages(path), "tools": tools})["prompt"]
        if os.environ.get("RENDER_DIR"):
            name = "-".join(path.split("/")[-3:-1])
            with open(os.path.join(os.environ["RENDER_DIR"], name + ".txt"), "w") as f:
                f.write(prompt)
            print(f"{len(prompt):>8} chars  {name}", flush=True)
            continue
        t = post("/completion", {"prompt": prompt, "n_predict": 1, "cache_prompt": False})["timings"]
        done.append({"session": path, "prompt_n": t["prompt_n"], "prompt_ms": t["prompt_ms"]})
        print(f"{t['prompt_n']:>7} tokens {t['prompt_n'] / t['prompt_ms'] * 1000:7.1f} t/s  {path}", flush=True)
    with open(sys.argv[1], "w") as f:
        json.dump(done, f, indent=1)


if __name__ == "__main__":
    main()
