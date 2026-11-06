#!/usr/bin/env python3
"""Ask any local chat model one question and stream the answer (OpenAI chat API).

    python3 examples/chat.py "Why is the sky blue? Two sentences."
    python3 examples/chat.py -m groot "Write a Python function that reverses a linked list"
    python3 examples/chat.py -m lfm3 --no-think "Give me three names for a cat"

The model's thinking (if it thinks) streams in grey before the answer. Names: scripts/llm list.
"""
import argparse
import json
import sys
import urllib.request

import _server


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question")
    ap.add_argument("-m", "--model", default="lfm3", help="a chat model from scripts/llm list (default lfm3)")
    ap.add_argument("--system", default="", help="a system prompt")
    ap.add_argument("--no-think", action="store_true", help="ask the model not to think first (if it supports it)")
    ap.add_argument("--max-tokens", type=int, default=4096)
    _server.add_server_args(ap)
    a = ap.parse_args()
    _server.ensure(a.model, a)

    messages = ([{"role": "system", "content": a.system}] if a.system else []) + [{"role": "user", "content": a.question}]
    body = {"messages": messages, "stream": True, "max_tokens": a.max_tokens}
    if a.no_think:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    req = urllib.request.Request(f"http://127.0.0.1:{_server.PORT}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    grey = sys.stdout.isatty()
    thinking = False
    with urllib.request.urlopen(req, timeout=900) as resp:
        for raw in resp:  # server-sent events: "data: {...}"
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            delta = json.loads(line[6:])["choices"][0]["delta"]
            if delta.get("reasoning_content"):
                if not thinking and grey:
                    print("\033[90m", end="")
                thinking = True
                print(delta["reasoning_content"], end="", flush=True)
            if delta.get("content"):
                if thinking:
                    print("\033[0m\n" if grey else "\n", end="")
                    thinking = False
                print(delta["content"], end="", flush=True)
    print("\033[0m" if thinking and grey else "")


if __name__ == "__main__":
    main()
