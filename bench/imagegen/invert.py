#!/usr/bin/env python3
"""Textual Inversion on SDXL: learn one new word in both text encoders, the rest of the model stays frozen.

Usage: invert.py --set subject --word zxqtok [--steps 1000] [--fit]
Writes models/image-gen/style-runs/<set>-word/<word>.safetensors, which sd-cli loads with --embd-dir.
"""
import argparse
import json
import random
import time

import numpy as np
import torch
from diffusers import DDPMScheduler
from safetensors.torch import save_file
from train_style import RUNS, cache, captions, pipe_load


def add_word(tok, enc, word, like):
    tok.add_tokens(word)
    enc.resize_token_embeddings(len(tok))
    i = tok.convert_tokens_to_ids(word)
    e = enc.get_input_embeddings()
    e.weight.data = e.weight.data.float()
    with torch.no_grad():
        e.weight[i] = e.weight[tok(like, add_special_tokens=False).input_ids[0]]
    e.weight.requires_grad_(True)
    return e, i, e.weight.detach().clone()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--word", required=True)
    ap.add_argument("--like", default="person")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--size", type=int, default=768)
    ap.add_argument("--fit", action="store_true")
    a = ap.parse_args()
    random.seed(0)
    torch.manual_seed(0)
    pipe = pipe_load()
    rows = captions(a.set, a.word)
    latents = cache(pipe, rows, a.size, 4)
    items = [(latents[i][0], rows[i // 4][1]) for i in range(len(latents))]
    for m in (pipe.unet, pipe.text_encoder, pipe.text_encoder_2):
        m.requires_grad_(False)
    pipe.unet.to("cuda")
    pipe.text_encoder.to("cpu", torch.float32)  # both encoders plus the U-Net don't fit in 8 GB
    pipe.text_encoder_2.to("cpu", torch.float32)
    pipe.unet.enable_gradient_checkpointing()
    e1, i1, keep1 = add_word(pipe.tokenizer, pipe.text_encoder, a.word, a.like)
    e2, i2, keep2 = add_word(pipe.tokenizer_2, pipe.text_encoder_2, a.word, a.like)
    opt = torch.optim.AdamW([e1.weight, e2.weight], lr=5e-3, weight_decay=0)
    scaler = torch.amp.GradScaler("cuda")
    sched = DDPMScheduler.from_config(pipe.scheduler.config)
    ids = torch.tensor([[a.size, a.size, 0, 0, a.size, a.size]], device="cuda", dtype=torch.float16)
    steps = 5 if a.fit else a.steps
    t0, losses = time.time(), []
    torch.cuda.reset_peak_memory_stats()
    for step in range(steps):
        lat, cap = random.choice(items)
        lat = lat.to("cuda")
        noise = torch.randn_like(lat)
        t = torch.randint(0, 1000, (1,), device="cuda")
        pe, _, pool, _ = pipe.encode_prompt(cap, device="cpu", do_classifier_free_guidance=False)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            pred = pipe.unet(sched.add_noise(lat, noise, t), t, pe.to("cuda"),
                             added_cond_kwargs={"text_embeds": pool.to("cuda"), "time_ids": ids}).sample
            loss = torch.nn.functional.mse_loss(pred.float(), noise.float())
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        with torch.no_grad():  # only the new word's row may move
            for e, i, keep in ((e1, i1, keep1), (e2, i2, keep2)):
                row = e.weight[i].clone()
                e.weight.copy_(keep)
                e.weight[i] = row
        losses.append(loss.item())
        if step % 100 == 0:
            print(step, round(float(np.mean(losses[-100:])), 4), round(time.time() - t0), flush=True)
    peak = torch.cuda.max_memory_allocated() / 2**30
    secs = time.time() - t0
    print(f"done: {steps} steps, {secs:.0f}s, peak {peak:.2f} GiB", flush=True)
    if a.fit:
        return
    run = RUNS / f"{a.set}-word"
    run.mkdir(parents=True, exist_ok=True)
    save_file({"clip_l": e1.weight[i1][None].detach().cpu().float().clone(),
               "clip_g": e2.weight[i2][None].detach().cpu().float().clone()}, run / f"{a.word}.safetensors")
    (run / "train.json").write_text(json.dumps({"set": a.set, "method": "word", "trainable": e1.embedding_dim + e2.embedding_dim,
                                                "steps": steps, "size": a.size, "secs": round(secs),
                                                "peak_gib": round(peak, 2), "trigger": a.word}))


if __name__ == "__main__":
    main()
