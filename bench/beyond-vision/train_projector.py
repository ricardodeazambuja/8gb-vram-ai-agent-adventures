#!/usr/bin/env python3
"""Teach LFM3-1.7B to read VL's vision features: train only the projector, on the glance scenes.

Runs only inside bench/finetune/sandbox.sh. LFM3-1.7B's text weights (from gguf_to_state.py) replace VL's;
the text model and the vision tower stay frozen. Saves norms.pt, which train_glance.py eval loads as text
weights plus the new projector, so scoring uses the same code as every other adapter.
"""
import argparse
import json
import os
import random
import sys
import time

import torch

sys.path.insert(0, "/bench/finetune")
import train_glance as tg  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="/model")
ap.add_argument("--text", default="/out/lfm3-1.7b-text.pt")
ap.add_argument("--out", default="/out/run")
ap.add_argument("--n-train", type=int, default=300, help="scenes per kind")
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--accum", type=int, default=8)
ap.add_argument("--fresh", action="store_true", help="start from random projector weights, not VL's")
ap.add_argument("--keep-vl-text", action="store_true", help="control: VL's own text weights")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--max-secs", type=int, default=2700)
a = ap.parse_args()
assert os.environ.get("ADV_SANDBOX") == "1", "run this through sandbox.sh"

torch.manual_seed(0)
proc, model = tg.load(a.model)
params = dict(model.named_parameters())
text = {}
if not a.keep_vl_text:
    state = torch.load(a.text, weights_only=True)
    text = {k: v for k, v in state.items() if k in params}
    print("text tensors loaded:", len(text), "skipped:", len(state) - len(text), flush=True)
    assert len(text) > 200
    model.load_state_dict(text, strict=False)
for p in model.parameters():
    p.requires_grad = False
proj = model.model.multi_modal_projector
if a.fresh:
    for m in proj.modules():
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()
for p in proj.parameters():
    p.requires_grad = True
    p.data = p.data.float()
trained = list(proj.parameters())
print("trainable", sum(p.numel() for p in trained), flush=True)
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.train()
data = [(name, i) for name in tg.rc.SCENES for i in range(a.n_train)]
random.Random(0).shuffle(data)
if a.limit:
    data = data[:a.limit]
opt = torch.optim.AdamW(trained, lr=a.lr, weight_decay=0.0)
scaler = torch.amp.GradScaler("cuda")
t0, run = time.time(), 0.0
for step, (name, i) in enumerate(data):
    if time.time() - t0 > a.max_secs:
        print("time limit at", step, flush=True)
        break
    img, prompt, tools, _, truth = tg.scene(name, f"train-{name}-{i}", i)
    enc = tg.encode(proc, img, prompt, tools, tg.target(name, truth, tools)).to("cuda")
    enc["pixel_values"] = enc["pixel_values"].to(torch.float16)
    with torch.autocast("cuda", dtype=torch.float16):
        loss = model(**enc, use_cache=False).loss / a.accum
    scaler.scale(loss).backward()
    run += loss.item()
    if (step + 1) % a.accum == 0:
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(trained, 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        if ((step + 1) // a.accum) % 10 == 0 or a.limit:
            print(f"step {(step + 1) // a.accum}/{len(data) // a.accum} loss {run / (1 if a.limit else 10):.4f} "
                  f"{time.time() - t0:.0f}s peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
            run = 0.0
if not a.limit:
    os.makedirs(a.out, exist_ok=True)
    out = {k: v.half().cpu() for k, v in text.items()}
    out.update({n: p.detach().half().cpu() for n, p in model.named_parameters() if "multi_modal_projector" in n})
    torch.save(out, os.path.join(a.out, "norms.pt"))
    with open(os.path.join(a.out, "train.json"), "w") as f:
        json.dump(vars(a) | {"secs": round(time.time() - t0)}, f)
    print("saved", a.out)
