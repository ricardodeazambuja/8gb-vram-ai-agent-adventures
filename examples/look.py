#!/usr/bin/env python3
"""Look at a picture and decide: LFM3-VL-2B picks one tool call (or none), the pattern of our
visual control probe (docs/visual-control.md, 144/150 with thinking).

    python3 examples/look.py                                    the built-in demo: a pressure gauge
    python3 examples/look.py photo.jpg --task "If a person is in the picture, call alert." \\
        --tools alert,ignore
    python3 examples/look.py gauge.png --tools-json my_tools.json --task "..."
    python3 examples/look.py --fast                             no thinking: 0.4 s instead of ~3 s, less accurate

It only PRINTS the call the model chose; acting on it is up to you. One picture per request: in long
conversations this model loops. --tools takes plain names (tools with no arguments);
--tools-json takes OpenAI-style tool definitions for tools with arguments.
"""
import argparse
import json
import time
from pathlib import Path

import _server

DEMO_IMAGE = _server.REPO / "bench" / "visual-control" / "examples" / "gauge.png"
DEMO_TASK = ("Read the pressure gauge. If the pressure is above 70, call open_valve. If it is below 20, call "
             "close_valve. Otherwise call report with the reading. Call exactly one tool.")


def tool(name, desc="", props=None):
    return {"type": "function", "function": {"name": name, "description": desc or name.replace("_", " "),
            "parameters": {"type": "object", "properties": props or {}, "required": list(props or {})}}}


DEMO_TOOLS = [tool("open_valve", "Open the relief valve"), tool("close_valve", "Close the relief valve"),
              tool("report", "Log the current reading", {"value": {"type": "number"}})]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", nargs="?", type=Path, default=DEMO_IMAGE)
    ap.add_argument("--task", default=DEMO_TASK, help="what to look for and which tool to call when")
    ap.add_argument("--tools", help="comma-separated tool names, no arguments")
    ap.add_argument("--tools-json", type=Path, help="a JSON list of OpenAI tool definitions")
    ap.add_argument("--fast", action="store_true", help="thinking off")
    _server.add_server_args(ap)
    a = ap.parse_args()
    if a.task != DEMO_TASK and not (a.tools or a.tools_json):
        ap.error("with your own --task, give --tools or --tools-json")
    tools = (json.loads(a.tools_json.read_text()) if a.tools_json else
             [tool(n.strip()) for n in a.tools.split(",")] if a.tools else DEMO_TOOLS)
    _server.ensure("lfm3vl", a)

    body = {"messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": _server.data_url(a.image)}},
                {"type": "text", "text": a.task + " If no tool fits, don't call any and say why."}]}],
            "tools": tools, "tool_choice": "auto", "parallel_tool_calls": True,
            "max_tokens": 600 if a.fast else 6000, "chat_template_kwargs": {"enable_thinking": not a.fast}}
    t = time.perf_counter()
    r = _server.post("/v1/chat/completions", body)
    m = r["choices"][0]["message"]
    calls = m.get("tool_calls") or []
    for c in calls:
        print(f"call: {c['function']['name']}({c['function'].get('arguments') or ''})")
    if not calls:
        print(f"no call: {(m.get('content') or '').strip()[:300]}")
    print(f"[{time.perf_counter() - t:.1f} s{', thinking' if not a.fast else ''}]")


if __name__ == "__main__":
    main()
