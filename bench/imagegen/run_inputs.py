#!/usr/bin/env python3
"""What each image generator does with a picture as input: restyle, masked redraw and reference edit.

Usage: run_inputs.py klein iris sdxl
Takes klein-photo-1.png as the input. Reports time, peak VRAM, how much the picture changed inside
the mask and well outside it, so "it ran" isn't mistaken for "it worked". Pictures go to
models/image-gen/out/inputs/ (git-ignored); results to bench/imagegen/results/inputs.json.
"""
import argparse
import json
import subprocess
import threading
import time

import numpy as np
from PIL import Image, ImageFilter
from run_gen import MODELS, OUT, RES, SD, vram

IN = OUT / "klein-photo-1.png"
D = OUT / "inputs"
EDIT = "wearing a red cap instead of his hat"
JOBS = {
    "restyle": (["-i", IN, "--strength", "0.7"], "watercolour painting of an old fisherman mending a net"),
    "mask": (["-i", IN, "--mask", D / "mask.png", "--strength", "0.9"], "a large red fishing boat"),
    "reference": (["-r", IN], f"the same man, {EDIT}"),
    "control": ([], f"the same man, {EDIT}"),  # no picture: -r must beat this to count as honoured
}


def make_mask():
    im = Image.open(IN).convert("RGB")
    m = np.zeros((im.height, im.width), np.uint8)
    h, w = m.shape
    m[h // 4:3 * h // 4, w // 4:3 * w // 4] = 255
    Image.fromarray(m).save(D / "mask.png")
    return np.asarray(im, float), m > 0


def run(model, job):
    extra, prompt = JOBS[job]
    out = D / f"{model}-{job}.png"
    cmd = [SD, *MODELS[model], *extra, "-p", prompt, "-s", 1, "-H", 1024, "-W", 1024, "--vae-tiling", "-o", out]
    peak, stop = [0], threading.Event()

    def watch():
        while not stop.wait(0.5):
            peak[0] = max(peak[0], vram())

    th = threading.Thread(target=watch)
    th.start()
    t = time.time()
    r = subprocess.run(list(map(str, cmd)), capture_output=True, text=True, check=False, timeout=900)
    stop.set()
    th.join()
    return out, {"model": model, "job": job, "rc": r.returncode, "secs": round(time.time() - t, 1), "peak_mib": peak[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", choices=list(MODELS))
    a = ap.parse_args()
    D.mkdir(parents=True, exist_ok=True)
    src, inside = make_mask()
    far = ~np.asarray(Image.fromarray(inside.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(65))).astype(bool)
    rows = []
    for model in a.models:
        for job in JOBS:
            out, row = run(model, job)
            if out.exists() and row["rc"] == 0:
                d = np.abs(np.asarray(Image.open(out).convert("RGB"), float) - src).mean(axis=2)
                row |= {"diff_all": round(float(d.mean()), 1), "diff_inside": round(float(d[inside].mean()), 1),
                        "diff_far_outside": round(float(d[far].mean()), 1)}
            print(row, flush=True)
            rows.append(row)
    (RES / "inputs.json").write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
