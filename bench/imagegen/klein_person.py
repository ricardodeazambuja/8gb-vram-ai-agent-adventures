#!/usr/bin/env python3
"""No-training baseline for the person: Klein with the reference picture and the same scenes.

Usage: klein_person.py
Pictures go to models/image-gen/out/style/klein-ref/. Compare with the trained SDXL runs.
"""
import subprocess

from run_gen import MODELS, SD
from train_style import SUBJECT_TEST, M

REF = M / "train/subject/ref.png"
OUT = M / "out/style/klein-ref"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for k, p in SUBJECT_TEST.items():
        for seed in (1, 2):
            cmd = [SD, *MODELS["klein"], "-r", REF, "-p", p.format(t="the same"), "-s", seed, "-H", 1024, "-W", 1024,
                   "--vae-tiling", "-o", OUT / f"{k}-{seed}.png"]
            subprocess.run(list(map(str, cmd)), check=True, capture_output=True, timeout=300)
            print(k, seed, flush=True)


if __name__ == "__main__":
    main()
