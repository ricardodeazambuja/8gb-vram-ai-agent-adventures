#!/usr/bin/env python3
"""Stack two learned words in one sd-cli prompt: a person and a style, at full and reduced style weight.

Pictures go to models/image-gen/out/style/stack/.
"""
import shutil
import subprocess

from run_gen import MODELS, SD
from train_style import RUNS, M

STYLES = {"vg": "vgword", "gr": "grword"}
SCENES = {"beach": "walking on a beach", "chef": "as a chef in a kitchen"}


def main():
    words = RUNS / "stack-words"
    words.mkdir(exist_ok=True)
    for run, word in (("subject-word", "zxqtok"), ("vangogh-word", "vgword"), ("gris-word", "grword")):
        shutil.copy(RUNS / run / f"{word}.safetensors", words)
    out = M / "out/style/stack"
    out.mkdir(parents=True, exist_ok=True)
    args = [str(x) for x in MODELS["sdxl"]] + ["--embd-dir", str(words)]
    for name, style in STYLES.items():
        for weight in (1.0, 0.7):
            for scene, text in SCENES.items():
                for seed in (1, 2):
                    prompt = f"({style}:{weight}) style, zxqtok man {text}"
                    subprocess.run([str(SD), *args, "-p", prompt, "-s", str(seed), "-H", "1024", "-W", "1024",
                                    "--vae-tiling", "-o", str(out / f"{name}{weight}-{scene}-{seed}.png")],
                                   check=True, capture_output=True, timeout=300)
                    print(name, weight, scene, seed, flush=True)


if __name__ == "__main__":
    main()
