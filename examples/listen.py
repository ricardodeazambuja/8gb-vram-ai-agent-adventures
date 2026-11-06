#!/usr/bin/env python3
"""Speech to text, and spoken tool calls, with LFM3-Audio-2B.

    python3 examples/listen.py clip.wav                       transcribe (English)
    python3 examples/listen.py clip.wav --tools lights_on,lights_off,play_music
                                                              let the spoken request pick a tool (printed only)
    python3 examples/listen.py                                demo: one of our recorded test questions

WAV in (16 kHz mono is safest). The model also speaks: for text to speech, see examples/speak.py; for
live two-way audio, its /v1/realtime websocket (bench/voice/realtime_probe.py, docs/models/lfm3-audio-2b.md).
"""
import argparse
import time
from pathlib import Path

import _server

DEMO = _server.REPO / "bench" / "voice" / "audio" / "q0.wav"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wav", nargs="?", type=Path, default=DEMO)
    ap.add_argument("--language", default="English")
    ap.add_argument("--tools", help="comma-separated tool names: the request picks one (it is only printed)")
    _server.add_server_args(ap)
    a = ap.parse_args()
    _server.ensure("lfm3audio", a)

    audio = {"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": _server.b64(a.wav),
                                                                                  "format": "wav"}}]}
    if a.tools:
        tools = [{"type": "function", "function": {"name": n.strip(), "description": n.strip().replace("_", " "),
                                                   "parameters": {"type": "object", "properties": {}}}}
                 for n in a.tools.split(",")]
        body = {"messages": [{"role": "system", "content": "You are a voice assistant. Use a tool when one fits; "
                                                           "otherwise answer briefly."}, audio],
                "tools": tools, "max_tokens": 1500, "temperature": 0}
    else:
        body = {"messages": [{"role": "system", "content": f"Perform ASR in {a.language}."}, audio],
                "max_tokens": 200, "temperature": 0}
    t = time.perf_counter()
    m = _server.post("/v1/chat/completions", body)["choices"][0]["message"]
    for c in m.get("tool_calls") or []:
        print(f"call: {c['function']['name']}({c['function'].get('arguments') or ''})")
    if m.get("content"):
        print(m["content"].strip())
    print(f"[{time.perf_counter() - t:.1f} s]")


if __name__ == "__main__":
    main()
