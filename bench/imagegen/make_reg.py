#!/usr/bin/env python3
"""Plain class pictures from the untouched model, for regularised training.

Usage: make_reg.py --word man --count 20
Pictures go to models/image-gen/train/reg-<word>/ (git-ignored).
"""
import argparse
import subprocess

from run_gen import MODELS, SD
from train_style import M

SCENES = ["walking on a beach", "in a kitchen", "in a park", "at a desk", "on a city street", "portrait, plain background",
          "in the mountains", "reading in a cafe", "in a blue shirt", "wearing a suit"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--word", default="man")
    ap.add_argument("--count", type=int, default=20)
    a = ap.parse_args()
    out = M / f"train/reg-{a.word}"
    out.mkdir(parents=True, exist_ok=True)
    for i in range(a.count):
        p = f"photo of a {a.word}, {SCENES[i % len(SCENES)]}"
        cmd = [SD, *MODELS["sdxl"], "-p", p, "-s", 100 + i, "-H", 1024, "-W", 1024, "--vae-tiling", "-o", out / f"{i:02}.png"]
        subprocess.run(list(map(str, cmd)), check=True, capture_output=True, timeout=300)
        print(i, flush=True)


if __name__ == "__main__":
    main()
