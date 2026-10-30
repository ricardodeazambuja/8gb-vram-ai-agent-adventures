#!/usr/bin/env python3
"""Fine-tune a small vision model on the visual-control glance scenes, and score it before and after.

    python3 bench/finetune/train_glance.py eval   --model DIR [--adapter DIR] [--n 30] --key KEY
    python3 bench/finetune/train_glance.py train  --model DIR --out DIR [--method lora|ia3|norm] [--n-train 400]
    python3 bench/finetune/train_glance.py fit    --model DIR       # one training step, peak VRAM

Runs in the `vlft` mamba env (torch, transformers 5.22, peft). Only models that transformers supports
natively: no trust_remote_code, ever.

Data: the five glance scenes of bench/visual-control/run_control.py. Training uses seeds
"train-{scene}-{i}"; evaluation uses the probe's own seeds "{scene}-{i}", so no test picture is
ever trained on (the drawing code is the same, so this measures learning our scenes, not the world).
The target answer is the right tool call, rendered through the model's own chat template; scenes
that need no call get a one-line refusal.

Safety: the model's answers are parsed with ast (call names and literal arguments only) and scored;
nothing it writes is run.
"""
import argparse
import ast
import json
import os
import random
import re
import sys
import time

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "visual-control"))
import run_control as rc

RES = os.path.join(HERE, "results")
CALL_RE = re.compile(r"<\|tool_call_start\|>(.*?)<\|tool_call_end\|>", re.DOTALL)


# ---------------------------------------------------------------- the right answer for a scene
def target(name, truth, tools):
    """(tool_calls, text) the scene wants."""
    def call(fn, **args):
        return {"type": "function", "function": {"name": fn, "arguments": args}}
    if name == "tanks":
        if not truth["want"]:
            return [], "No tank is below 30%, so nothing needs a refill."
        return [call("refill", tank=t) for t in truth["want"]], ""
    if name == "click":
        if truth["absent"]:
            return [], f"There is no '{truth['target']}' button in this dialog, so I won't click anything."
        b = truth["boxes"][truth["target"]]
        return [call("click", x=round((b[0] + b[2]) / 2 * 1000 / 800), y=round((b[1] + b[3]) / 2 * 1000 / 500))], ""
    if name == "pick":
        t = truth["target"]
        return [call("pick", x=round(t["x"] * 1000 / 600), y=round(t["y"] * 1000 / 600))], ""
    if name == "signal":
        return [call(truth["want"])], ""
    if name == "gauge":
        if truth["want"] == "report":
            return [call("report", value=truth["value"])], ""
        return [call(truth["want"])], ""
    raise ValueError(name)


def scene(name, seed, i):
    img, prompt, tools, score, truth = rc.SCENES[name](random.Random(seed), i)
    return img, prompt, tools, score, truth


# ---------------------------------------------------------------- parsing the model's answer
def parse_calls(text):
    """'<|tool_call_start|>[f(a=1, b="x")]<|tool_call_end|>' -> [{"name", "args"}], literals only."""
    calls = []
    for block in CALL_RE.findall(text):
        try:
            tree = ast.parse(block.strip(), mode="eval")
        except SyntaxError:
            calls.append({"name": "?", "args": None})
            continue
        items = tree.body.elts if isinstance(tree.body, ast.List) else [tree.body]
        for node in items:
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                calls.append({"name": "?", "args": None})
                continue
            try:
                args = {k.arg: ast.literal_eval(k.value) for k in node.keywords}
            except ValueError:
                args = None
            calls.append({"name": node.func.id, "args": args})
    return calls


# ---------------------------------------------------------------- model plumbing
def load(model_dir, adapter=None):
    proc = AutoProcessor.from_pretrained(model_dir)
    model = AutoModelForImageTextToText.from_pretrained(model_dir, dtype=torch.float16).to("cuda")
    if adapter and os.path.exists(os.path.join(adapter, "norms.pt")):  # norm/vnorm: new gains, same shapes
        state = torch.load(os.path.join(adapter, "norms.pt"), weights_only=True)
        missing = set(state) - {n for n, _ in model.named_parameters()}
        assert not missing, f"unknown parameters in norms.pt: {sorted(missing)[:3]}"
        model.load_state_dict(state, strict=False)
    elif adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    return proc, model


def chat(img, prompt, tools, answer=None):
    msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
    if answer is not None:
        calls, text = answer
        m = {"role": "assistant", "content": text}
        if calls:
            m["tool_calls"] = calls
        msgs.append(m)
    return msgs


def encode(proc, img, prompt, tools, answer):
    """input_ids, pixel inputs and labels with the prompt masked out."""
    full = proc.apply_chat_template(chat(img, prompt, tools, answer), tools=tools, tokenize=False)
    if not full.rstrip().endswith("<|im_end|>"):
        full += "<|im_end|>\n"  # a tool-call-only turn: the template adds no end marker after an empty content
    head = proc.apply_chat_template(chat(img, prompt, tools), tools=tools, tokenize=False, add_generation_prompt=True)
    enc = proc(text=[full], images=[[img]], return_tensors="pt")
    n_head = proc(text=[head], images=[[img]], return_tensors="pt")["input_ids"].shape[1]
    labels = enc["input_ids"].clone()
    labels[:, :n_head] = -100
    enc["labels"] = labels
    return enc


@torch.no_grad()
def answer(proc, model, img, prompt, tools, max_new=120):
    text = proc.apply_chat_template(chat(img, prompt, tools), tools=tools, tokenize=False, add_generation_prompt=True)
    enc = proc(text=[text], images=[[img]], return_tensors="pt").to("cuda")
    enc["pixel_values"] = enc["pixel_values"].to(torch.float16)
    out = model.generate(**enc, max_new_tokens=max_new, do_sample=False)
    return proc.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=False)


# ---------------------------------------------------------------- commands
def shifted(img, i):
    """The same scene as a worse camera would see it: smaller, darker or brighter, JPEG-compressed.
    Uniform scaling keeps the 0-1000 coordinates valid, so the scorers don't change."""
    import io

    from PIL import ImageEnhance
    w, h = img.size
    img = img.resize((int(w * 0.7), int(h * 0.7)))
    img = ImageEnhance.Brightness(img).enhance(0.75 if i % 2 else 1.2)
    img = ImageEnhance.Contrast(img).enhance(0.8)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=35)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def cmd_eval(a):
    proc, model = load(a.model, a.adapter)
    model.eval()
    res = {"test": "glance-hf", "key": a.key, "model": a.model, "adapter": a.adapter, "shift": a.shift,
           "items": [], "scores": {}}
    for name in rc.SCENES:
        ok = 0
        for i in range(a.n):
            img, prompt, tools, score, _ = scene(name, f"{name}-{i}", i)
            if a.shift:
                img = shifted(img, i)
            t = time.time()
            raw = answer(proc, model, img, prompt, tools)
            calls = parse_calls(raw)
            good = bool(score({"calls": calls}))
            ok += good
            res["items"].append({"scene": name, "i": i, "ok": good, "calls": calls, "raw": raw[:300],
                                 "secs": round(time.time() - t, 2)})
        res["scores"][name] = ok
        print(f"== {name}: {ok}/{a.n}", flush=True)
    total = sum(res["scores"].values())
    print(f"== all: {total}/{a.n * len(rc.SCENES)}")
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, f"eval-{a.key}.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)


def trainable(model, method):
    if method == "lora":
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                         target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|out_proj|in_proj|w1|w2|w3)")
        return get_peft_model(model, cfg)
    if method == "ia3":
        from peft import IA3Config, get_peft_model
        cfg = IA3Config(target_modules=r".*language_model.*\.(k_proj|v_proj|w2)",
                        feedforward_modules=r".*language_model.*\.w2")
        return get_peft_model(model, cfg)
    if method in ("norm", "vnorm"):  # norm: the language model's RMSNorm gains; vnorm: the vision side's norms
        side = "language_model" if method == "norm" else ("vision_tower", "multi_modal_projector")
        for n, p in model.named_parameters():
            p.requires_grad = (n.split(".")[1] in side if method == "vnorm" else side in n) and "norm" in n
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.float()
        return model
    raise ValueError(method)


def cmd_train(a, steps_only=None):
    torch.manual_seed(a.seed)
    proc, model = load(a.model)
    model = trainable(model, a.method)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"method {a.method}: {n_train:,} trainable of {n_all:,} ({100 * n_train / n_all:.3f}%)", flush=True)
    for p in model.parameters():  # fp32 master weights for what we train (Turing: fp16 autocast, no bf16)
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.train()
    data = [(name, i) for name in rc.SCENES for i in range(a.n_train)]
    random.Random(a.seed).shuffle(data)
    if steps_only:
        data = data[:steps_only * a.accum]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda")
    torch.cuda.reset_peak_memory_stats()
    t0, run_loss = time.time(), 0.0
    for step, (name, i) in enumerate(data):
        img, prompt, tools, _, truth = scene(name, f"train-{name}-{i}", i)
        enc = encode(proc, img, prompt, tools, target(name, truth, tools)).to("cuda")
        enc["pixel_values"] = enc["pixel_values"].to(torch.float16)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = model(**enc).loss / a.accum
        scaler.scale(loss).backward()
        run_loss += loss.item()
        if (step + 1) % a.accum == 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            if ((step + 1) // a.accum) % 10 == 0 or steps_only:
                print(f"step {(step + 1) // a.accum}/{len(data) // a.accum} loss {run_loss / 10 if not steps_only else run_loss:.4f} "
                      f"{time.time() - t0:.0f}s peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
                run_loss = 0.0
    print(f"done: {len(data)} examples in {time.time() - t0:.0f}s, peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    if steps_only:
        return
    os.makedirs(a.out, exist_ok=True)
    if a.method in ("norm", "vnorm"):
        state = {n: p.detach().half().cpu() for n, p in model.named_parameters() if p.requires_grad}
        torch.save(state, os.path.join(a.out, "norms.pt"))
    else:
        model.save_pretrained(a.out)
    with open(os.path.join(a.out, "train.json"), "w") as f:
        json.dump({"method": a.method, "seed": a.seed, "n_train": a.n_train, "lr": a.lr, "accum": a.accum,
                   "trainable": n_train, "secs": round(time.time() - t0)}, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["eval", "train", "fit"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--out")
    ap.add_argument("--key", default="model")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--n-train", type=int, default=400)
    ap.add_argument("--method", choices=["lora", "ia3", "norm", "vnorm"], default="lora")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--shift", action="store_true", help="eval on degraded pictures (smaller, JPEG, brightness)")
    a = ap.parse_args()
    if a.cmd == "eval":
        cmd_eval(a)
    elif a.cmd == "fit":
        cmd_train(a, steps_only=2)
    else:
        cmd_train(a)


if __name__ == "__main__":
    main()
