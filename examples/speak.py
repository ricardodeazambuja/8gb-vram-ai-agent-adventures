#!/usr/bin/env python3
"""Text to speech with LFM3-Audio-2B, through llama.cpp's llama-tts (no server needed).

    python3 examples/speak.py "Hello! The build passed."               writes models/lfm3-audio-2b/out/speech-<time>.wav
    python3 examples/speak.py "Bonjour" --voice male -o hello.wav

llama-tts loads its own copy of the model on the GPU, so stop any server first (scripts/llm stop): next
to the LFM3-Audio server and a 2.5 GB desktop it ran out of memory. It is a separate llama.cpp tool:
if it's missing, build it with
    (cd ../llama.cpp && cmake --build build --target llama-tts)
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BIN = Path(os.environ.get("ADV_LLAMA", REPO.parent / "llama.cpp")) / "build" / "bin" / "llama-tts"
M = REPO / "models" / "lfm3-audio-2b"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text")
    ap.add_argument("--voice", choices=["female", "male"], default="female")
    ap.add_argument("-o", "--out", type=Path)
    a = ap.parse_args()
    if not BIN.exists():
        sys.exit(f"llama-tts not found at {BIN}. Build it:\n  (cd ../llama.cpp && cmake --build build --target llama-tts)")
    model, proj = M / "LFM3-Audio-2B-Q8_0.gguf", M / "mmproj-LFM3-Audio-2B-F16.gguf"
    if not model.exists():
        sys.exit("LFM3-Audio-2B is not downloaded: bash scripts/download-models.sh lfm3audio")
    out = a.out or M / "out" / f"speech-{time.strftime('%Y%m%d-%H%M%S')}.wav"
    out.parent.mkdir(parents=True, exist_ok=True)
    t = time.perf_counter()
    p = subprocess.run([str(BIN), "-m", str(model), "-mm", str(proj), "-ngl", "99", "--tts-voice", a.voice,
                        "-p", a.text, "-o", str(out)], capture_output=True, text=True, check=False)
    if p.returncode or not out.exists():
        sys.exit(f"llama-tts failed:\n{p.stderr[-800:]}\n"
                 "If a server holds the GPU (scripts/llm status), stop it first: scripts/llm stop")
    print(out)
    print(f"[{time.perf_counter() - t:.1f} s]", file=sys.stderr)


if __name__ == "__main__":
    main()
