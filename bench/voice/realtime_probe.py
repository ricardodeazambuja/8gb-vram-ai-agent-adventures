#!/usr/bin/env python3
"""Spoken questions through llama-server's /v1/realtime: stream the audio in real time (as a
microphone would), then measure end of speech -> first audio back. Saves each spoken reply.

    realtime_probe.py   # needs llama-server with LFM3-Audio + projector on :8080
"""
import asyncio
import base64
import json
import os
import time
import wave

import numpy as np
import websockets

HERE = os.path.dirname(os.path.abspath(__file__))
AUD = os.path.join(HERE, "audio")
OUT = os.environ.get("VOICE_OUT", AUD)
CLIPS = ["s4.wav", "q0.wav", "q3.wav"]  # "remind me to call my sister", weather in Paris, 312 / 4


def load(path, rate):
    with wave.open(path) as w:
        sr, x = w.getframerate(), np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        if w.getnchannels() > 1:
            x = x.reshape(-1, w.getnchannels())[:, 0]
    if sr != rate:  # linear resampling is enough for a probe
        t = np.arange(0, len(x) * rate / sr) * sr / rate
        x = np.interp(t, np.arange(len(x)), x.astype(np.float32)).astype(np.int16)
    return x


async def one(clip):
    async with websockets.connect("ws://127.0.0.1:8080/v1/realtime", max_size=None) as ws:
        created = json.loads(await ws.recv())
        fmt = created["session"]["audio"]
        rin, rout = fmt["input"]["format"]["rate"], fmt["output"]["format"]["rate"]
        await ws.send(json.dumps({"type": "session.update", "session": {
            "type": "realtime", "instructions": "You are a friendly voice assistant. Answer in one short sentence.",
            "output_modalities": ["text", "audio"]}}))
        x = load(os.path.join(AUD, clip), rin)
        chunk = rin // 10  # 100 ms chunks, sent at real-time pace
        for i in range(0, len(x), chunk):
            await ws.send(json.dumps({"type": "input_audio_buffer.append",
                                      "audio": base64.b64encode(x[i:i + chunk].tobytes()).decode()}))
            await asyncio.sleep(0.1)
        t0 = time.time()
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        await ws.send(json.dumps({"type": "response.create"}))
        first_audio = first_text = None
        text, pcm = "", b""
        while True:
            ev = json.loads(await asyncio.wait_for(ws.recv(), timeout=120))
            t = ev["type"]
            if t == "response.output_audio.delta":
                first_audio = first_audio or time.time() - t0
                pcm += base64.b64decode(ev["delta"])
            elif t == "response.output_text.delta":
                first_text = first_text or time.time() - t0
                text += ev["delta"]
            elif t == "error":
                print("  error:", ev.get("error"))
                break
            elif t == "response.done":
                break
        total = time.time() - t0
        out = os.path.join(OUT, f"reply-{clip}")
        with wave.open(out, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rout)
            w.writeframes(pcm)
        speech = len(pcm) / 2 / rout
        fa = f"{first_audio:.2f}s" if first_audio else "none"
        ft = f"{first_text:.2f}s" if first_text else "none"
        print(f"  {clip}: first audio {fa}, first text {ft}, done {total:.1f}s, "
              f"{speech:.1f}s of speech back | {text.strip()[:120]!r}")


async def main():
    for c in CLIPS:
        try:
            await one(c)
        except Exception as e:  # noqa: BLE001
            print(f"  {c}: failed: {type(e).__name__}: {e}")


asyncio.run(main())
