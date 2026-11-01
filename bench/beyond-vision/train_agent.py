#!/usr/bin/env python3
"""Train a LoRA on LFM3-VL's language model from data/agent.json. Runs only inside bench/finetune/sandbox.sh."""
import argparse
import json
import os
import random
import time

import torch
import transformers.utils

transformers.utils.is_triton_available = lambda: False
from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="/model")
ap.add_argument("--out", default="/out/agent-lora")
ap.add_argument("--epochs", type=int, default=2)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--accum", type=int, default=8)
ap.add_argument("--limit", type=int, default=0, help="stop after this many examples, for a fit test")
ap.add_argument("--max-secs", type=int, default=3000)
a = ap.parse_args()
assert os.environ.get("ADV_SANDBOX") == "1", "run this through sandbox.sh"

with open("/bench/beyond-vision/data/agent.json") as f:
    rows = json.load(f)
proc = AutoProcessor.from_pretrained(a.model, trust_remote_code=True)
tok = proc.tokenizer
model = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.float16, trust_remote_code=True).to("cuda")
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                         target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|out_proj|in_proj|w1|w2|w3)"))
for p in model.parameters():
    if p.requires_grad:
        p.data = p.data.float()
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model.train()
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
scaler = torch.amp.GradScaler("cuda")
order = [r for e in range(a.epochs) for r in random.Random(e).sample(rows, len(rows))]
if a.limit:
    order = order[:a.limit]
t0, total = time.time(), 0.0
for step, r in enumerate(order):
    if time.time() - t0 > a.max_secs:
        print("time limit at example", step, flush=True)
        break
    head = tok(r["head"], add_special_tokens=False, return_tensors="pt").input_ids
    tail = tok(r["tail"], add_special_tokens=False, return_tensors="pt").input_ids
    ids = torch.cat([head, tail], 1).to("cuda")
    n = tail.shape[1]
    with torch.autocast("cuda", dtype=torch.float16):
        logits = model(input_ids=ids, logits_to_keep=n + 1).logits  # the whole sequence would be 1.75 GiB
        loss = torch.nn.functional.cross_entropy(logits[0, :-1], ids[0, -n:]) / a.accum
    scaler.scale(loss).backward()
    total += loss.item()
    if (step + 1) % a.accum == 0:
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        print(f"step {(step + 1) // a.accum}/{len(order) // a.accum} loss {total:.4f} {time.time() - t0:.0f}s "
              f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
        total = 0.0
if not a.limit:
    model.save_pretrained(a.out)
    print("saved", a.out)
