#!/usr/bin/env python3
"""Replay a captured pi request against a llama-server, with and without llama.cpp's tool-call grammar, and print
what the model wrote.

    python3 bench/gemma12-probe/replay.py CAPTURE_REQ.json [--server http://127.0.0.1:8081] [--seeds 1 2 3] [--n 700]

For each seed it makes two calls with the same sampling (taken from the server's defaults):
  raw:     /apply-template renders the prompt exactly as /v1/chat/completions would, then /completion generates
           from it with no grammar, so the text is what the model writes on its own (special tokens included);
  grammar: the captured request itself through /v1/chat/completions (not streamed), so llama.cpp's tool-call
           grammar and parser apply; prints the parsed tool calls.
"""
import argparse
import json
import urllib.request


def post(server, path, body):
    req = urllib.request.Request(server + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("--server", default="http://127.0.0.1:8081")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--n", type=int, default=700, help="tokens to generate per call")
    ap.add_argument("--repeat-penalty", type=float, default=None, help="override the server's repeat penalty")
    a = ap.parse_args()

    with open(a.capture) as f:
        req = json.load(f)
    tmpl = {k: req[k] for k in ("messages", "tools", "reasoning_effort") if k in req}
    prompt = post(a.server, "/apply-template", tmpl)["prompt"]
    print(f"== {a.capture}: prompt {len(prompt)} chars, ends with {prompt[-60:]!r}")

    extra = {} if a.repeat_penalty is None else {"repeat_penalty": a.repeat_penalty}
    for seed in a.seeds:
        raw = post(a.server, "/completion", {"prompt": prompt, "n_predict": a.n, "seed": seed, "cache_prompt": True,
                                              "special": True, **extra})
        text = raw["content"]
        print(f"\n--- seed {seed}, raw (no grammar), {raw.get('tokens_predicted')} tokens, "
              f"stop: {raw.get('stop_type')}")
        i = text.find("<|tool_call>")
        print("first tool call:", repr(text[i:i + 400]) if i >= 0 else "none")
        print("text after the thought:", repr(text[text.find('<channel|>') + 10:][:300]) if '<channel|>' in text
              else repr(text[:300]))

        body = dict(req, stream=False, seed=seed, max_completion_tokens=a.n, **extra)
        body.pop("stream_options", None)
        out = post(a.server, "/v1/chat/completions", body)
        msg = out["choices"][0]["message"]
        calls = [(c["function"]["name"], c["function"]["arguments"][:200]) for c in msg.get("tool_calls") or []]
        print(f"--- seed {seed}, grammar (chat endpoint), finish: {out['choices'][0].get('finish_reason')}, "
              f"{out.get('usage', {}).get('completion_tokens')} tokens")
        print("tool calls:", calls[:4], f"(+{len(calls) - 4} more)" if len(calls) > 4 else "")
        print("content:", repr((msg.get("content") or "")[:200]))


if __name__ == "__main__":
    main()
