#!/usr/bin/env python3
"""Teach SDXL a style or a subject on 8 GB: LoRA or norm-only, with text and latents cached up front.

Usage:
  train_style.py train --set vangogh --method lora --trigger vgstyl [--steps 1000] [--size 768] [--fit]
  train_style.py eval  --run vangogh-lora --trigger vgstyl
Sets are folders under models/image-gen/train/. Adapters and pictures stay under models/image-gen/.
"""
import argparse
import json
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionXLPipeline
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
M = ROOT / "models/image-gen"
CKPT = M / "sdxl/sd_xl_base_1.0.safetensors"
VAE = M / "sdxl/sdxl_vae.safetensors"
RUNS = M / "style-runs"
DATA = Path(__file__).parent / "data"
RES = Path(__file__).parent / "results"
TEST = {"lighthouse": "a lighthouse on a cliff above the sea", "bicycle": "a bicycle leaning against a brick wall",
        "cat": "a cat asleep on a wooden chair", "street": "a busy city street at night",
        "woman": "portrait of a woman in a green coat"}
SUBJECT_TEST = {"beach": "{t} man walking on a beach", "astronaut": "{t} man as an astronaut on the moon",
                "watercolour": "watercolour painting of {t} man", "chef": "{t} man in a chef's hat, photograph"}


def pipe_load():
    vae = AutoencoderKL.from_single_file(VAE, torch_dtype=torch.float16)
    vae.register_to_config(scaling_factor=0.13025)  # a lone VAE file loads with SD 1.5's 0.18215
    return StableDiffusionXLPipeline.from_single_file(CKPT, vae=vae, torch_dtype=torch.float16)


def captions(name, trigger):
    d = M / "train" / name
    if name == "subject":
        rows = json.loads((d / "captions.json").read_text())
        return [(d / r["file"], f"photo of {trigger} man, {r['caption']}") for r in rows if not r["held_out"]]
    rows = json.loads((DATA / f"{name}.json").read_text())
    out = []
    for r in rows:
        t = re.sub(r"\.\w+$", "", r["title"].removeprefix("File:"))
        t = re.sub(r"[-_,]+|\b(s\d+V\d+|Van Gogh Museum|Google Art Project|by|Vincent van Gogh)\b", " ", t)
        out.append((d / r["file"], f"{trigger} style, {' '.join(t.split())[:80]}"))
    return out


def crops(path, size, n):
    im = Image.open(path).convert("RGB")
    s = max(size / min(im.size), 0.0) * 1.15
    im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    for _ in range(n):
        x, y = random.randint(0, im.width - size), random.randint(0, im.height - size)
        yield im.crop((x, y, x + size, y + size))


@torch.no_grad()
def cache(pipe, rows, size, n_crops):
    pipe.vae.to("cuda")
    pipe.text_encoder.to("cuda")
    pipe.text_encoder_2.to("cuda")
    items = []
    for path, cap in rows:
        pe, _, pool, _ = pipe.encode_prompt(cap, device="cuda", do_classifier_free_guidance=False)
        for c in crops(path, size, n_crops):
            x = torch.from_numpy(np.asarray(c).copy()).float().permute(2, 0, 1)[None] / 127.5 - 1
            lat = pipe.vae.encode(x.to("cuda", torch.float16)).latent_dist.mean * pipe.vae.config.scaling_factor
            items.append((lat.cpu(), pe.cpu(), pool.cpu()))
    pipe.vae.to("cpu")
    pipe.text_encoder.to("cpu")
    pipe.text_encoder_2.to("cpu")
    torch.cuda.empty_cache()
    return items


def setup(unet, method):
    unet.requires_grad_(False)
    if method == "lora":
        unet.add_adapter(LoraConfig(r=8, lora_alpha=8, init_lora_weights="gaussian",
                                    target_modules=["to_k", "to_q", "to_v", "to_out.0"]))
    params = []
    for n, p in unet.named_parameters():
        if method == "lora" and "lora" in n or method == "norm" and re.search(r"norm\w*\.(weight|bias)$", n):
            p.requires_grad_(True)
        else:
            continue
        p.data = p.data.float()
        params.append(p)
    return params


def train(a):
    seed = 0
    random.seed(seed)
    torch.manual_seed(seed)
    run = RUNS / f"{a.set}-{a.method}{a.tag}"
    run.mkdir(parents=True, exist_ok=True)
    pipe = pipe_load()
    rows = captions(a.set, a.trigger)
    items = cache(pipe, rows, a.size, 4)
    unet = pipe.unet.to("cuda")
    unet.enable_gradient_checkpointing()
    unet.train()
    params = setup(unet, a.method)
    n_train = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=1e-4 if a.method == "lora" else 1e-3, weight_decay=0)
    scaler = torch.amp.GradScaler("cuda")
    sched = DDPMScheduler.from_config(pipe.scheduler.config)
    ids = torch.tensor([[a.size, a.size, 0, 0, a.size, a.size]], device="cuda", dtype=torch.float16)
    steps = 5 if a.fit else a.steps
    t0, losses = time.time(), []
    torch.cuda.reset_peak_memory_stats()
    for step in range(steps):
        lat, pe, pool = random.choice(items)
        lat, pe, pool = lat.to("cuda"), pe.to("cuda"), pool.to("cuda")
        noise = torch.randn_like(lat) + a.noise_offset * torch.randn(*lat.shape[:2], 1, 1, device="cuda", dtype=lat.dtype)
        t = torch.randint(0, 1000, (1,), device="cuda")
        noisy = sched.add_noise(lat, noise, t)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            pred = unet(noisy, t, pe, added_cond_kwargs={"text_embeds": pool, "time_ids": ids}).sample
            loss = torch.nn.functional.mse_loss(pred.float(), noise.float())
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        losses.append(loss.item())
        if step % 100 == 0:
            print(step, round(sum(losses[-100:]) / len(losses[-100:]), 4), round(time.time() - t0), flush=True)
    secs = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"done: {n_train} params, {steps} steps, {secs:.0f}s, peak {peak:.2f} GiB", flush=True)
    if a.fit:
        return
    if a.method == "lora":
        sd = get_peft_model_state_dict(unet)
        StableDiffusionXLPipeline.save_lora_weights(run, unet_lora_layers=sd, safe_serialization=True)
    else:
        torch.save({n: p.detach().cpu() for n, p in unet.named_parameters() if p.requires_grad}, run / "norms.pt")
    (run / "train.json").write_text(json.dumps({"set": a.set, "method": a.method, "trainable": n_train, "steps": steps,
                                                "size": a.size, "secs": round(secs), "peak_gib": round(peak, 2),
                                                "images": len(rows), "trigger": a.trigger}))


def eval_(a):
    run = RUNS / a.run
    out = M / "out/style" / a.run
    out.mkdir(parents=True, exist_ok=True)
    pipe = pipe_load()
    if a.run.startswith("base"):
        pass
    elif (run / "norms.pt").exists():
        sd = torch.load(run / "norms.pt", map_location="cpu", weights_only=True)
        pipe.unet.load_state_dict({k: v.half() for k, v in sd.items()}, strict=False)
    else:
        pipe.load_lora_weights(run)
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()
    subject = "subject" in a.run
    tests = {k: v.format(t=a.trigger) for k, v in SUBJECT_TEST.items()} if subject else \
        {k: (f"{a.trigger} style, {v}" if a.trigger else v + a.suffix) for k, v in TEST.items()}
    if not subject and a.trigger:
        tests["no-trigger"] = TEST["lighthouse"]
    elif subject:
        tests["no-trigger"] = "photo of a man walking on a beach"
    for k, p in tests.items():
        for seed in (1, 2):
            g = torch.Generator("cpu").manual_seed(seed)
            pipe(p, num_inference_steps=25, guidance_scale=7, height=1024, width=1024, generator=g).images[0]\
                .save(out / f"{k}-{seed}.png")
            print(k, seed, flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--set", required=True)
    t.add_argument("--method", choices=["lora", "norm"], required=True)
    t.add_argument("--trigger", required=True)
    t.add_argument("--steps", type=int, default=1000)
    t.add_argument("--size", type=int, default=768)
    t.add_argument("--fit", action="store_true")
    t.add_argument("--tag", default="")
    t.add_argument("--noise-offset", type=float, default=0.0)
    e = sub.add_parser("eval")
    e.add_argument("--run", required=True)
    e.add_argument("--trigger", default="")
    e.add_argument("--suffix", default="")
    a = ap.parse_args()
    train(a) if a.cmd == "train" else eval_(a)


if __name__ == "__main__":
    main()
