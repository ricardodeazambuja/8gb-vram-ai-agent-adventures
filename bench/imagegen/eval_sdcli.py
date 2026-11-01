#!/usr/bin/env python3
"""Score a trained run through sd-cli, the way we would ship it, on the same prompts and seeds as train_style.

Usage: eval_sdcli.py --run subject-lora --trigger zxq    (LoRA, baked.safetensors or word file, found in the run)
Pictures go to models/image-gen/out/style/<run>-sdcli/.
"""
import argparse
import subprocess

from run_gen import MODELS, SD
from train_style import M, RUNS, SUBJECT_TEST, TEST


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--trigger", required=True)
    a = ap.parse_args()
    run = RUNS / a.run
    out = M / "out/style" / f"{a.run}-sdcli"
    out.mkdir(parents=True, exist_ok=True)
    args = [str(x) for x in MODELS["sdxl"]]
    tag = ""
    if (run / "baked.safetensors").exists():
        args[1] = str(run / "baked.safetensors")
    elif (run / "pytorch_lora_weights.safetensors").exists():
        args += ["--lora-model-dir", str(run)]
        tag = "<lora:pytorch_lora_weights:1>"
    else:
        args += ["--embd-dir", str(run)]
    subject = "subject" in a.run
    tests = {k: v.format(t=a.trigger) for k, v in SUBJECT_TEST.items()} if subject else \
        {k: f"{a.trigger} style, {v}" for k, v in TEST.items()}
    tests["no-trigger"] = "photo of a man walking on a beach" if subject else TEST["lighthouse"]
    for k, p in tests.items():
        for seed in (1, 2):
            subprocess.run([str(SD), *args, "-p", p + tag, "-s", str(seed), "-H", "1024", "-W", "1024", "--vae-tiling",
                            "-o", str(out / f"{k}-{seed}.png")], check=True, capture_output=True, timeout=300)
            print(k, seed, flush=True)


if __name__ == "__main__":
    main()
