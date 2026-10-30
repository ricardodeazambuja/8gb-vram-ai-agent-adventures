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


# ---------------------------------------------------------------- grid: steer the robot, one fresh picture per step
def grid_prompt(history):
    recent = ", ".join(history[-3:]) or "none yet"
    return f"{rc.GRID_RULES}\nYour last moves, oldest first: {recent}."


def best_move(walls, robot, goal):
    """A move on a shortest path (the first in up, down, left, right order)."""
    d0 = rc.bfs(walls, robot, goal)
    for name, (dx, dy) in rc.MOVES.items():
        n = (robot[0] + dx, robot[1] + dy)
        if 0 <= n[0] < rc.N and 0 <= n[1] < rc.N and n not in walls and rc.bfs(walls, n, goal) == d0 - 1:
            return name
    return None


def grid_examples(n_episodes, prefix, recover=0):
    """Every step of an optimal walk through n grids (half with walls): (walls, robot, goal, history, move).
    recover=k adds k states per grid at random free cells, where possible just after bumping into a wall
    or the edge ("left (blocked)"), so the model also sees how to get going again off the shortest path."""
    out = []
    for j in range(n_episodes):
        rng = random.Random(f"{prefix}-{j % 2 == 1}-{j}")
        walls, robot, goal, _ = rc.make_grid(rng, j % 2 == 1)
        history = []
        while robot != goal:
            mv = best_move(walls, robot, goal)
            out.append((walls, robot, goal, list(history), mv))
            dx, dy = rc.MOVES[mv]
            robot = (robot[0] + dx, robot[1] + dy)
            history.append(f"{mv} (moved)")
        free = [(x, y) for x in range(rc.N) for y in range(rc.N)
                if (x, y) not in walls and (x, y) != goal and rc.bfs(walls, (x, y), goal)]
        for _ in range(recover):
            cell = rng.choice(free)
            bumps = [m for m, (dx, dy) in rc.MOVES.items()
                     if not (0 <= cell[0] + dx < rc.N and 0 <= cell[1] + dy < rc.N) or (cell[0] + dx, cell[1] + dy) in walls]
            hist = [f"{rng.choice(list(rc.MOVES))} (moved)"]
            if bumps and rng.random() < 0.7:
                hist.append(f"{rng.choice(bumps)} (blocked)")
            out.append((walls, cell, goal, hist, best_move(walls, cell, goal)))
    return out


def cmd_eval_grid(a):
    proc, model = load(a.model, a.adapter)
    model.eval()
    res = {"test": "grid-hf", "key": a.key, "adapter": a.adapter, "episodes": []}
    for walled in (False, True):
        ok = 0
        for i in range(8):
            walls, robot, goal, dist = rc.make_grid(random.Random(f"grid-{walled}-{i}"), walled)
            cap, steps, blocked, nocall, history = 2 * dist + 8, 0, 0, 0, []
            while steps < cap and robot != goal:
                calls = parse_calls(answer(proc, model, rc.draw_grid(walls, robot, goal), grid_prompt(history),
                                           rc.GRID_TOOLS, max_new=40))
                steps += 1
                mv = (calls[0]["args"] or {}).get("direction") if calls and calls[0]["name"] == "move" else None
                if mv not in rc.MOVES:
                    nocall += 1
                    history.append("no valid move")
                    continue
                nxt = (robot[0] + rc.MOVES[mv][0], robot[1] + rc.MOVES[mv][1])
                moved = 0 <= nxt[0] < rc.N and 0 <= nxt[1] < rc.N and nxt not in walls
                blocked += not moved
                robot = nxt if moved else robot
                history.append(f"{mv} ({'moved' if moved else 'blocked'})")
            ok += robot == goal
            res["episodes"].append({"walled": walled, "i": i, "success": robot == goal, "steps": steps,
                                    "optimal": dist, "blocked": blocked, "no_call": nocall})
            print(f"  {'walls' if walled else 'open '} {i} {'OK  ' if robot == goal else 'FAIL'} steps {steps} "
                  f"(best {dist}) blocked {blocked} no-call {nocall}", flush=True)
        print(f"== {'walls' if walled else 'open'}: {ok}/8", flush=True)
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, f"grid-{a.key}.json"), "w") as f:
        json.dump(res, f, indent=1)


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
    if a.task == "grid":
        data = [("grid", ex) for ex in grid_examples(a.n_train, "train-grid", a.recover)]
    else:
        data = [(name, i) for name in rc.SCENES for i in range(a.n_train)]
    random.Random(a.seed).shuffle(data)
    if steps_only:
        data = data[:steps_only * a.accum]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda")
    torch.cuda.reset_peak_memory_stats()
    t0, run_loss = time.time(), 0.0
    for step, (name, i) in enumerate(data):
        if name == "grid":
            walls, robot, goal, history, mv = i
            img, prompt, tools = rc.draw_grid(walls, robot, goal), grid_prompt(history), rc.GRID_TOOLS
            ans = ([{"type": "function", "function": {"name": "move", "arguments": {"direction": mv}}}], "")
        else:
            img, prompt, tools, _, truth = scene(name, f"train-{name}-{i}", i)
            ans = target(name, truth, tools)
        enc = encode(proc, img, prompt, tools, ans).to("cuda")
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
        json.dump({"task": a.task, "recover": a.recover, "method": a.method, "seed": a.seed, "n_train": a.n_train, "lr": a.lr, "accum": a.accum,
                   "trainable": n_train, "secs": round(time.time() - t0)}, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["eval", "train", "fit"])
    ap.add_argument("--task", choices=["glance", "grid"], default="glance",
                    help="grid: train on optimal walks (--n-train = grids), eval closed-loop on the probe's 16 grids")
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--out")
    ap.add_argument("--key", default="model")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--n-train", type=int, default=400)
    ap.add_argument("--method", choices=["lora", "ia3", "norm", "vnorm"], default="lora")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--recover", type=int, default=0, help="grid: extra off-path states per grid (after a bump)")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--shift", action="store_true", help="eval on degraded pictures (smaller, JPEG, brightness)")
    a = ap.parse_args()
    if a.cmd == "eval":
        cmd_eval_grid(a) if a.task == "grid" else cmd_eval(a)
    elif a.cmd == "fit":
        cmd_train(a, steps_only=2)
    else:
        cmd_train(a)


if __name__ == "__main__":
    main()
