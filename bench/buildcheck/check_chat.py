#!/usr/bin/env python3
"""Quick checks against a running llama-server (OpenAI API), greedy, nothing executed.

    check_chat.py URL text      one arithmetic question: is 391 in the answer (or the thinking)?
    check_chat.py URL tool      one tool-call request: does it call get_weather with Paris?
    check_chat.py URL image PNG  count the dots in bench/vision's understand-count.png (truth 7 red, 11 blue)

Prints one line per check: PASS/FAIL, decode tok/s from the server's timings, and the start of the answer.
Standard library only; a model's tool call is printed, never run.
"""
import base64
import json
import sys
import time
import urllib.request


def ask(url, body, timeout=900):
    body = {"temperature": 0, "max_tokens": 4096, **body}
    req = urllib.request.Request(f"{url}/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    return d, time.time() - t0


def line(ok, d, wall, what):
    t = d.get("timings", {})
    tps = t.get("predicted_per_second")
    n = t.get("predicted_n")
    tag = "PASS" if ok else "FAIL"
    print(f"{tag} | decode {tps:.1f} tok/s over {n} tokens | {wall:.1f} s | {what}" if tps else f"{tag} | {wall:.1f} s | {what}")
    return ok


def short(s, n=160):
    return " ".join((s or "").split())[:n]


def main():
    url, kind = sys.argv[1].rstrip("/"), sys.argv[2]
    if kind == "text":
        d, w = ask(url, {"messages": [{"role": "user", "content": "What is 17 * 23? Reply with just the number."}]})
        m = d["choices"][0]["message"]
        text = m.get("content") or ""
        ok = "391" in text or ("391" in (m.get("reasoning_content") or "") and not text.strip())
        return line(ok, d, w, f"text: {short(text) or '(empty content; thinking ' + str(len(m.get('reasoning_content') or '')) + ' chars)'}")
    if kind == "tool":
        tools = [{"type": "function", "function": {
            "name": "get_weather", "description": "Current weather for a city",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
        d, w = ask(url, {"messages": [{"role": "user", "content": "What's the weather in Paris right now? Use the tool."}],
                         "tools": tools})
        calls = d["choices"][0]["message"].get("tool_calls") or []
        names = [(c["function"]["name"], c["function"].get("arguments", "")) for c in calls]
        ok = any(n == "get_weather" and "paris" in a.lower() for n, a in names)
        return line(ok, d, w, f"tool: {names or short(d['choices'][0]['message'].get('content'))}")
    if kind == "image":
        img = base64.b64encode(open(sys.argv[3], "rb").read()).decode()
        prompt = ("Count the dots. Return a JSON object with red (number of red dots), blue (number of blue dots) "
                  "and total. Answer with JSON only, no other text.")
        d, w = ask(url, {"messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}},
            {"type": "text", "text": prompt}]}]})
        text = d["choices"][0]["message"].get("content") or ""
        # a working projector gives a JSON count; exactness depends on the model (truth 7/11/18)
        ok = '"red"' in text and '"blue"' in text
        return line(ok, d, w, f"image (truth red 7, blue 11): {short(text)}")
    sys.exit(f"unknown check {kind}")


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
