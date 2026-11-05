"""Decode speed of one chat request with and without tools (the agent loop's first turn). Usage: toolsdiff.py"""
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agentloop import (  # the agent loop's tools and opening messages (PI_CAPTURE applies)
    msgs,
    tools,
)


def post(body):
    r = urllib.request.Request("http://127.0.0.1:8090/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=3600))
for label, extra in [("chat+tools", {"tools": tools}), ("chat no tools", {}), ("chat+tools", {"tools": tools})]:
    b = {"messages": msgs, "temperature": 0.6, "seed": 7, "max_tokens": 700, **extra}
    t = post(b)["timings"]
    print(f"   {label:<14} gen {t['predicted_n']:>4} tok  {t['predicted_per_second']:6.2f} t/s", flush=True)
