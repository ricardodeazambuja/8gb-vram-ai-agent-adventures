#!/usr/bin/env python3
"""Image generators on 8 GB through stable-diffusion.cpp: fixed prompts and seeds, time and peak VRAM.

Usage: run_gen.py klein iris sdxl [--size 1024] [--tag NAME]
Images go to models/image-gen/out/ (git-ignored); results to bench/imagegen/results/.
"""
import argparse
import json
import re
import subprocess
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SD = ROOT.parent / "stable-diffusion.cpp/build/bin/sd-cli"
M = ROOT / "models/image-gen"
OUT = M / "out"
RES = Path(__file__).parent / "results"
PROMPTS = {
    "photo": "a photograph of an old fisherman mending a net on a wooden dock at sunrise, natural light",
    "hands": "close-up photograph of two hands tying a red shoelace on a muddy hiking boot",
    "sign": 'a shop window with a hand-painted sign that says "OPEN LATE", street at night, rain',
    "scene": "a tiny robot watering a bonsai tree on a cluttered desk, watercolour illustration",
}
SEEDS = (1, 2)
CTE = ["--backend", "te=cpu"]
MODELS = {
    "klein": ["--diffusion-model", M / "flux2-klein/flux-2-klein-4b-Q4_0.gguf",
              "--vae", M / "flux2-klein/split_files/vae/flux2-vae.safetensors",
              "--llm", M / "flux2-klein/Qwen3-4B-Q4_K_M.gguf",
              "--cfg-scale", "1.0", "--steps", "4", *CTE],
    "iris": ["--diffusion-model", M / "iris-3b/turbo/model.safetensors",
             "--llm", M / "iris-3b/Qwen3VL-4B-Instruct-Q4_K_M.gguf",
             "--cfg-scale", "1.0", "--steps", "4", *CTE],
    "sdxl": ["-m", M / "sdxl/sd_xl_base_1.0.safetensors", "--vae", M / "sdxl/sdxl_vae.safetensors",
             "--cfg-scale", "7.0", "--steps", "25", "--sampling-method", "dpm++2m", *CTE],
}


def vram():
    r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, check=False)
    return int(r.stdout.strip() or 0)


def one(model, label, prompt, seed, size, tag):
    out = OUT / f"{model}{tag}-{label}-{seed}.png"
    cmd = [SD, *MODELS[model], "-p", prompt, "-s", seed, "-H", size, "-W", size, "--vae-tiling", "-o", out]
    base, peak, stop = vram(), [0], threading.Event()

    def watch():
        while not stop.wait(0.5):
            peak[0] = max(peak[0], vram())

    th = threading.Thread(target=watch)
    th.start()
    t = time.time()
    try:
        r = subprocess.run(list(map(str, cmd)), capture_output=True, text=True, check=False, timeout=1500)
        rc, log = r.returncode, r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        rc, log = -1, "timeout"
    secs = time.time() - t
    stop.set()
    th.join()
    (OUT / f"{model}{tag}.log").write_text(log[-20000:])
    m = re.findall(r"(?:sampling completed|generate_image completed).*?([\d.]+)s", log)
    return {"model": model, "prompt": label, "seed": seed, "size": size, "rc": rc, "secs": round(secs, 1),
            "base_mib": base, "peak_mib": peak[0], "ok": out.exists(), "timings": m[:3],
            "tail": "" if rc == 0 else log[-400:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", choices=list(MODELS))
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--tag", default="")
    ap.add_argument("--quick", action="store_true", help="one prompt, one seed")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    RES.mkdir(exist_ok=True)
    tag = f"-{a.tag}" if a.tag else ""
    for model in a.models:
        rows = []
        for label, prompt in PROMPTS.items():
            for seed in SEEDS:
                r = one(model, label, prompt, seed, a.size, tag)
                rows.append(r)
                print(f"== {model} {label} {seed}: rc={r['rc']} {r['secs']}s peak {r['peak_mib']} MiB", flush=True)
                if r["rc"]:
                    print(r["tail"], flush=True)
                    break
                if a.quick:
                    break
            if rows[-1]["rc"] or a.quick:
                break
        (RES / f"{model}{tag}-{a.size}.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
