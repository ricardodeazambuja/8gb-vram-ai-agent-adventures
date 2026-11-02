#!/usr/bin/env python3
"""LFM3-Audio-2B round trip on this laptop: text -> speech (llama-tts) -> text (llama-mtmd-cli ASR),
scored by word error rate; then spoken requests to llama-server with a tool offered.

    audio_probe.py tts-asr     # needs llama-server with the model + projector on :8080
    audio_probe.py tools       # needs llama-server with the model + projector on :8080
    audio_probe.py replies     # transcribes the replies saved by realtime_probe.py
"""
import base64
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
import wave

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
BIN = os.environ.get("LLAMA_BIN", os.path.join(REPO, "..", "llama.cpp", "build", "bin"))
M = f"{REPO}/models/lfm3-audio-2b/LFM3-Audio-2B-Q8_0.gguf"
MM = f"{REPO}/models/lfm3-audio-2b/mmproj-LFM3-Audio-2B-F16.gguf"
OUT = os.environ.get("VOICE_OUT", os.path.join(HERE, "audio"))
os.makedirs(OUT, exist_ok=True)

SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Please book a table for four people at seven thirty tomorrow evening.",
    "My laptop has eight gigabytes of video memory and thirty two gigabytes of RAM.",
    "Restart the server, then check the log file for any errors.",
    "Could you remind me to call my sister on Saturday morning?",
    "The meeting moved from Tuesday to Thursday because the room was taken.",
]
REQUESTS = [  # (spoken text, expected tool, expected argument check)
    ("What's the weather like in Paris right now?", "get_weather", lambda a: "paris" in json.dumps(a).lower()),
    ("Set a timer for ten minutes, please.", "set_timer", lambda a: "10" in json.dumps(a) or "ten" in json.dumps(a).lower()),
    ("Send a message to Maria saying I'll be late.", "send_message", lambda a: "maria" in json.dumps(a).lower()),
    ("How much is three hundred and twelve divided by four?", None, lambda a: True),
]
TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
     "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "set_timer", "description": "Start a countdown timer",
     "parameters": {"type": "object", "properties": {"minutes": {"type": "number"}}, "required": ["minutes"]}}},
    {"type": "function", "function": {"name": "send_message", "description": "Send a text message to a contact",
     "parameters": {"type": "object", "properties": {"to": {"type": "string"}, "text": {"type": "string"}},
                    "required": ["to", "text"]}}},
]


def words(s):
    return re.sub(r"[^a-z0-9' ]", " ", s.lower()).split()


def wer(ref, hyp):
    r, h = words(ref), words(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)] / max(1, len(r))


def duration(path):
    with wave.open(path) as w:
        return w.getnframes() / w.getframerate()


def b64_of(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def tts(text, path, voice="female"):
    t = time.time()
    p = subprocess.run([f"{BIN}/llama-tts", "-m", M, "-mm", MM, "-ngl", "99", "--tts-voice", voice,
                        "-p", text, "-o", path], capture_output=True, text=True, timeout=300, check=False)
    if p.returncode or not os.path.exists(path):
        print("    TTS failed:", p.stderr[-400:])
        return None
    return time.time() - t


def asr(path):  # through llama-server on :8080 (model + projector loaded)
    b64 = b64_of(path)
    body = {"messages": [{"role": "system", "content": "Perform ASR in English."},
                         {"role": "user", "content": [{"type": "input_audio",
                                                       "input_audio": {"data": b64, "format": "wav"}}]}],
            "max_tokens": 200, "temperature": 0}
    req = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t = time.time()
    try:
        d = json.load(urllib.request.urlopen(req, timeout=300))
    except Exception as e:  # noqa: BLE001
        return "", time.time() - t, 1, str(e)
    return (d["choices"][0]["message"].get("content") or "").strip(), time.time() - t, 0, ""


def tts_asr():
    tot_w = tot_n = 0
    for i, s in enumerate(SENTENCES):
        voice = "female" if i % 2 == 0 else "male"
        wav = os.path.join(OUT, f"s{i}.wav")
        dt = tts(s, wav, voice)
        if dt is None:
            continue
        dur = duration(wav)
        text, at, rc, err = asr(wav)
        e = wer(s, text)
        tot_w += e * len(words(s))
        tot_n += len(words(s))
        print(f"  [{voice:6s}] {dur:4.1f}s of speech, TTS {dt:4.1f}s, ASR {at:4.1f}s, WER {e:4.0%} | {text[:110]!r}"
              + (f" rc={rc} {err!r}" if rc else ""))
    print(f"== TTS->ASR round trip: WER {tot_w / max(1, tot_n):.1%} over {tot_n} words")


def tools():
    ok = 0
    for i, (spoken, want, check) in enumerate(REQUESTS):
        wav = os.path.join(OUT, f"q{i}.wav")
        if tts(spoken, wav, "male") is None:
            continue
        b64 = b64_of(wav)
        body = {"messages": [
            {"role": "system", "content": "You are a voice assistant. Use a tool when one fits; otherwise answer briefly."},
            {"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}}]}],
            "tools": TOOLS, "max_tokens": 1500, "temperature": 0}
        req = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        t = time.time()
        try:
            d = json.load(urllib.request.urlopen(req, timeout=300))
        except Exception as e:  # noqa: BLE001
            print(f"  {spoken!r}: request failed: {e}")
            continue
        dt = time.time() - t
        m = d["choices"][0]["message"]
        calls = m.get("tool_calls") or []
        if want is None:
            good = not calls and "78" in (m.get("content") or "")
            got = f"answer {(m.get('content') or '')[:80]!r}"
            if calls:
                got += f", but called {[(c['function']['name'], c['function']['arguments']) for c in calls]}"
        else:
            names = [c["function"]["name"] for c in calls]
            args = [json.loads(c["function"]["arguments"] or "{}") for c in calls]
            good = want in names and check(args[names.index(want)])
            got = f"calls {[(n, a) for n, a in zip(names, args)]}" if calls else f"no call, said {(m.get('content') or '')[:80]!r}"
        ok += good
        print(f"  {'ok ' if good else 'MISS'} {spoken!r} -> {got} ({dt:.1f}s)")
    print(f"== spoken requests: {ok}/{len(REQUESTS)} handled right")


def replies():  # what the live test's spoken replies actually said
    for clip in ("s4.wav", "q0.wav", "q3.wav"):
        path = os.path.join(OUT, f"reply-{clip}")
        if os.path.getsize(path) > 100:
            print(f"  ASR of reply-{clip}: {asr(path)[0]!r}")


if __name__ == "__main__":
    {"tts-asr": tts_asr, "tools": tools, "replies": replies}[sys.argv[1]]()
