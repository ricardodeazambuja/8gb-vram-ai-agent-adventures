#!/usr/bin/env python3
"""Visual control probe: can a vision model look at a picture, decide, and call the right tool?

    python3 bench/visual-control/run_control.py glance KEY [--think on|off] [--n 12]
    python3 bench/visual-control/run_control.py grid KEY [--think on|off] [--memory stateless|history] [--n 8]
    python3 bench/visual-control/run_control.py examples      # save one picture of each scene

Needs a llama-server with the vision projector on 127.0.0.1:8080 (start it yourself, e.g.
`systemd-run --user --scope --unit=llama-lfm3vl -E QMODEL=lfm3vl bash scripts/run-qwen.sh --image`).
KEY only names the results file.

  glance  one picture, one decision: five kinds of scene (tank levels, a dialog's buttons, objects on
          a table, a traffic light, a pressure gauge), each with its own tools. Some scenes need no
          call at all (no tank is low, the button isn't there).
  grid    a closed loop: the model steers a robot to a goal on a 7x7 grid with walls, one move() per
          step, seeing only the picture. "stateless" sends a fresh context every step (the picture,
          the goal, the last three moves as text); "history" keeps the whole conversation.

Every scene is drawn here from a seed, so the answers are known and runs are repeatable.

Safety: the model only ever produces tool calls. They are parsed with json.loads, checked against
the tool's schema, and applied to the pure-Python simulation below: nothing the model writes is
run, evaluated, written to a path it chose, or sent anywhere. The only network traffic is to the
local server.
"""
import argparse
import base64
import collections
import io
import itertools
import json
import os
import random
import sys
import time
import urllib.request

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
URL = "http://127.0.0.1:8080/v1/chat/completions"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def font(size, bold=False):
    return ImageFont.truetype(FONT_B if bold else FONT, size)


def png_b64(img):
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def tool(name, desc, props=None, required=None):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props or {}, "required": required or []}}}


POINT = {"x": {"type": "integer", "description": "0-1000, left to right"},
         "y": {"type": "integer", "description": "0-1000, top to bottom"}}


# ---------------------------------------------------------------- the model, through llama-server
TOOL_CHOICE = "auto"  # "required": llama-server constrains the answer to a tool call (--tool-choice)
# With a call forced, the model needs a way to decline: every request then also offers no_action, and
# calling it counts as making no call (so the scorers are the same in both modes).
NO_ACTION = tool("no_action", "Call this when none of the other tools should be used",
                 {"reason": {"type": "string"}}, ["reason"])


def chat(messages, tools, think, max_tokens=None):
    if TOOL_CHOICE == "required":
        tools = tools + [NO_ACTION]
    body = {"messages": messages, "tools": tools, "parallel_tool_calls": True, "tool_choice": TOOL_CHOICE,
            "max_tokens": max_tokens or (6000 if think else 600),
            "chat_template_kwargs": {"enable_thinking": think}}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = time.time()
    try:
        d = json.load(urllib.request.urlopen(req, timeout=300))
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}", "calls": [], "text": "", "secs": time.time() - t}
    m = d["choices"][0]["message"]
    calls = []
    for c in m.get("tool_calls") or []:
        try:
            args = json.loads(c["function"].get("arguments") or "{}")
        except (json.JSONDecodeError, TypeError):
            args = None
        calls.append({"name": c["function"]["name"], "args": args, "id": c.get("id", "")})
    declined = [c for c in calls if c["name"] == "no_action"]
    calls = [c for c in calls if c["name"] != "no_action"]
    return {"calls": calls, "declined": [c["args"] for c in declined], "text": (m.get("content") or "").strip(),
            "reasoning": (m.get("reasoning_content") or "").strip(), "raw": m,
            "secs": time.time() - t, "tokens": d.get("usage", {}).get("completion_tokens")}


def valid(call, tools):
    """The call names one of the tools and its arguments match that tool's schema."""
    spec = next((t["function"] for t in tools if t["function"]["name"] == call["name"]), None)
    if spec is None or not isinstance(call["args"], dict):
        return False
    props, req = spec["parameters"]["properties"], spec["parameters"]["required"]
    if any(k not in call["args"] for k in req) or any(k not in props for k in call["args"]):
        return False
    for k, v in call["args"].items():
        p = props[k]
        is_int = (isinstance(v, int) and not isinstance(v, bool)) or (isinstance(v, float) and v.is_integer())
        if p["type"] == "integer" and not is_int:
            return False
        if p["type"] == "number" and not isinstance(v, (int, float)):
            return False
        if p["type"] == "string" and not isinstance(v, str):
            return False
        if "enum" in p and v not in p["enum"]:
            return False
    return True


# ---------------------------------------------------------------- glance: five kinds of scene
def scene_tanks(rng, i):
    img = Image.new("RGB", (640, 400), "white")
    d = ImageDraw.Draw(img)
    none_low = i % 4 == 0  # a quarter of the scenes need no call
    levels = {}
    for k, name in enumerate("ABC"):
        low = (not none_low) and rng.random() < 0.45
        levels[name] = rng.randint(5, 22) if low else rng.randint(38, 95)
    if not none_low and all(v >= 30 for v in levels.values()):
        levels[rng.choice("ABC")] = rng.randint(5, 22)
    for k, name in enumerate("ABC"):
        x0, y0, x1, y1 = 90 + k * 190, 60, 190 + k * 190, 340
        fill_top = y1 - (y1 - y0) * levels[name] / 100
        d.rectangle([x0, fill_top, x1, y1], fill=(70, 130, 220))
        d.rectangle([x0, y0, x1, y1], outline="black", width=3)
        for pct in (0, 25, 50, 75, 100):
            yy = y1 - (y1 - y0) * pct / 100
            d.line([x0 - 10, yy, x0, yy], fill="black", width=2)
            d.text((x0 - 52, yy - 9), f"{pct}%", fill="black", font=font(15))
        d.text(((x0 + x1) / 2 - 10, 352), name, fill="black", font=font(26, True))
    want = sorted(n for n, v in levels.items() if v < 30)
    tools = [tool("refill", "Start refilling one tank", {"tank": {"type": "string", "enum": ["A", "B", "C"]}}, ["tank"])]
    prompt = ("This picture shows three water tanks, A, B and C, with a scale in percent. Refill every tank "
              "that is below 30% full (one refill call per tank). If no tank is below 30%, don't call "
              "anything and say so.")

    def score(r):
        got = sorted(c["args"].get("tank") for c in r["calls"] if c["name"] == "refill" and isinstance(c["args"], dict))
        return got == want
    return img, prompt, tools, score, {"levels": levels, "want": want}


BUTTONS = ["Submit", "Cancel", "Delete", "Save", "Help", "Settings", "Export", "Login", "Print", "Share"]


def scene_click(rng, i):
    W, H = 800, 500
    img = Image.new("RGB", (W, H), (236, 236, 240))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 44], fill=(60, 64, 80))
    d.text((16, 10), "Document properties", fill="white", font=font(20, True))
    labels = rng.sample(BUTTONS, rng.randint(4, 6))
    absent = i % 5 == 0  # one in five asks for a button that isn't there
    target = rng.choice([b for b in BUTTONS if b not in labels]) if absent else rng.choice(labels)
    rects, boxes = [], {}
    for lab in labels:
        for _ in range(200):
            w, h = 130, 46
            x, y = rng.randint(20, W - w - 20), rng.randint(70, H - h - 20)
            r = (x, y, x + w, y + h)
            if all(r[2] + 15 < q[0] or q[2] + 15 < r[0] or r[3] + 15 < q[1] or q[3] + 15 < r[1] for q in rects):
                break
        rects.append(r)
        boxes[lab] = r
        d.rounded_rectangle(r, 8, fill=(250, 250, 252), outline=(90, 90, 110), width=2)
        tw = d.textlength(lab, font=font(18))
        d.text(((r[0] + r[2] - tw) / 2, r[1] + 12), lab, fill="black", font=font(18))
    tools = [tool("click", "Click at a point of the screen", POINT, ["x", "y"]),
             tool("type_text", "Type text into the focused field", {"text": {"type": "string"}}, ["text"])]
    prompt = (f"This is a screenshot of a dialog. Click the '{target}' button. Give the point in 0-1000 "
              f"coordinates of the image. If there is no such button, don't click anything and say so.")

    def score(r):
        clicks = [c for c in r["calls"] if c["name"] == "click" and isinstance(c["args"], dict)]
        if absent:
            return not r["calls"]
        if len(r["calls"]) != 1 or not clicks:
            return False
        try:
            x, y = clicks[0]["args"]["x"] * W / 1000, clicks[0]["args"]["y"] * H / 1000
        except (KeyError, TypeError):
            return False
        b = boxes[target]
        return b[0] <= x <= b[2] and b[1] <= y <= b[3]
    return img, prompt, tools, score, {"target": target, "absent": absent, "boxes": boxes}


COLORS = {"red": (220, 40, 40), "blue": (40, 90, 220), "green": (40, 170, 70), "yellow": (240, 200, 30),
          "purple": (140, 60, 180), "orange": (245, 130, 30)}
SHAPES = ["circle", "square", "triangle"]


def scene_pick(rng, i):
    S = 600
    img = Image.new("RGB", (S, S), (205, 180, 140))  # a wooden table
    d = ImageDraw.Draw(img)
    combos = rng.sample([(c, s) for c in COLORS for s in SHAPES], rng.randint(4, 6))
    objs = []
    for c, s in combos:
        for _ in range(500):
            r = rng.randint(32, 48)
            x, y = rng.randint(r + 15, S - r - 15), rng.randint(r + 15, S - r - 15)
            if all(abs(x - o["x"]) > r + o["r"] + 20 or abs(y - o["y"]) > r + o["r"] + 20 for o in objs):
                break
        objs.append({"color": c, "shape": s, "x": x, "y": y, "r": r})
        box = [x - r, y - r, x + r, y + r]
        if s == "circle":
            d.ellipse(box, fill=COLORS[c], outline="black", width=2)
        elif s == "square":
            d.rectangle(box, fill=COLORS[c], outline="black", width=2)
        else:
            d.polygon([(x, y - r), (x - r, y + r), (x + r, y + r)], fill=COLORS[c], outline="black")
    if i % 2 == 0:
        t = rng.choice(objs)
        ask = f"the {t['color']} {t['shape']}"
    else:
        side, key, rev = rng.choice([("leftmost", "x", False), ("rightmost", "x", True),
                                     ("topmost", "y", False), ("bottommost", "y", True)])
        t = sorted(objs, key=lambda o: o[key], reverse=rev)[0]
        ask = f"the {side} object"
    tools = [tool("pick", "Move the robot gripper to a point of the picture and grasp", POINT, ["x", "y"])]
    prompt = (f"You control a robot arm above this table (camera view from the top). Pick up {ask}. Give "
              f"the point in 0-1000 coordinates of the image.")

    def score(r):
        p = [c for c in r["calls"] if c["name"] == "pick" and isinstance(c["args"], dict)]
        if len(r["calls"]) != 1 or not p:
            return False
        try:
            x, y = p[0]["args"]["x"] * S / 1000, p[0]["args"]["y"] * S / 1000
        except (KeyError, TypeError):
            return False
        return abs(x - t["x"]) <= t["r"] + 4 and abs(y - t["y"]) <= t["r"] + 4
    return img, prompt, tools, score, {"ask": ask, "target": t, "objects": objs}


def scene_signal(rng, i):
    img = Image.new("RGB", (640, 420), (150, 200, 240))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 260, 640, 420], fill=(80, 80, 85))  # the road
    for k in range(8):  # a crosswalk
        d.rectangle([180 + k * 40, 300, 200 + k * 40, 400], fill="white")
    light = rng.choice(["red", "yellow", "green"])
    person = rng.random() < 0.35
    d.rectangle([520, 40, 580, 210], fill=(30, 30, 30))
    d.rectangle([545, 210, 555, 300], fill=(30, 30, 30))
    for k, (name, col) in enumerate([("red", (230, 30, 30)), ("yellow", (245, 200, 20)), ("green", (30, 200, 60))]):
        d.ellipse([530, 48 + k * 54, 570, 88 + k * 54], fill=col if name == light else (70, 70, 70))
    if person:
        px = rng.randint(220, 460)
        d.ellipse([px - 14, 250, px + 14, 278], fill=(250, 210, 170), outline="black")
        d.line([px, 278, px, 330], fill="black", width=6)
        d.line([px, 292, px - 20, 315], fill="black", width=5)
        d.line([px, 292, px + 20, 315], fill="black", width=5)
        d.line([px, 330, px - 16, 370], fill="black", width=5)
        d.line([px, 330, px + 16, 370], fill="black", width=5)
    want = "stop" if light == "red" or person else ("slow_down" if light == "yellow" else "go")
    tools = [tool("go", "Drive on"), tool("stop", "Stop the car"), tool("slow_down", "Slow down")]
    prompt = ("You drive the car and this is the view ahead. Rules: red light, stop; yellow light, slow down; "
              "green light, go; but if a person is on the crosswalk, stop whatever the light. Call exactly "
              "one tool.")

    def score(r):
        return len(r["calls"]) == 1 and r["calls"][0]["name"] == want
    return img, prompt, tools, score, {"light": light, "person": person, "want": want}


def scene_gauge(rng, i):
    import math
    img = Image.new("RGB", (500, 420), "white")
    d = ImageDraw.Draw(img)
    cx, cy, R = 250, 260, 190
    kind = i % 3  # high, low, in between
    val = rng.randint(76, 97) if kind == 0 else rng.randint(3, 14) if kind == 1 else rng.randint(26, 64)
    d.arc([cx - R, cy - R, cx + R, cy + R], 180, 360, fill="black", width=4)
    for v in range(0, 101, 5):
        a = math.pi * (1 - v / 100)
        r0 = R - (24 if v % 10 == 0 else 12)
        d.line([cx + r0 * math.cos(a), cy - r0 * math.sin(a), cx + R * math.cos(a), cy - R * math.sin(a)],
               fill="black", width=3 if v % 10 == 0 else 1)
        if v % 10 == 0:
            tx, ty = cx + (R - 46) * math.cos(a), cy - (R - 46) * math.sin(a)
            d.text((tx - 12, ty - 9), str(v), fill="black", font=font(17))
    a = math.pi * (1 - val / 100)
    d.line([cx, cy, cx + (R - 30) * math.cos(a), cy - (R - 30) * math.sin(a)], fill=(200, 20, 20), width=5)
    d.ellipse([cx - 10, cy - 10, cx + 10, cy + 10], fill="black")
    d.text((cx - 50, cy + 30), "PRESSURE", fill="black", font=font(20, True))
    want = "open_valve" if val > 70 else "close_valve" if val < 20 else "report"
    tools = [tool("open_valve", "Open the relief valve"), tool("close_valve", "Close the relief valve"),
             tool("report", "Log the current reading", {"value": {"type": "number"}}, ["value"])]
    prompt = ("Read the pressure gauge. If the pressure is above 70, call open_valve. If it is below 20, call "
              "close_valve. Otherwise call report with the reading. Call exactly one tool.")

    def score(r):
        if len(r["calls"]) != 1 or r["calls"][0]["name"] != want:
            return False
        if want == "report":
            try:
                return abs(float(r["calls"][0]["args"]["value"]) - val) <= 5
            except (KeyError, TypeError, ValueError):
                return False
        return True
    return img, prompt, tools, score, {"value": val, "want": want}


SCENES = {"tanks": scene_tanks, "click": scene_click, "pick": scene_pick, "signal": scene_signal,
          "gauge": scene_gauge}


def run_glance(key, think, n):
    out = {"test": "glance", "key": key, "think": think, "tool_choice": TOOL_CHOICE, "items": []}
    for name, gen in SCENES.items():
        ok = 0
        for i in range(n):
            rng = random.Random(f"{name}-{i}")
            img, prompt, tools, score, truth = gen(rng, i)
            msgs = [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{png_b64(img)}"}},
                {"type": "text", "text": prompt}]}]
            r = chat(msgs, tools, think)
            good = bool(not r.get("error") and score(r))
            ok += good
            bad = [c for c in r["calls"] if not valid(c, tools)]
            out["items"].append({"scene": name, "i": i, "ok": good, "calls": r["calls"], "declined": r.get("declined"),
                                 "text": r["text"][:300], "reasoning": r.get("reasoning", "")[-1200:],
                                 "invalid_calls": len(bad), "secs": round(r["secs"], 2), "tokens": r.get("tokens"),
                                 "error": r.get("error"), "truth": json.loads(json.dumps(truth, default=str))})
            print(f"  {name:6s} {i:2d} {'ok ' if good else 'MISS'} {r['secs']:5.1f}s "
                  f"calls={[(c['name'], c['args']) for c in r['calls']]} {r.get('error') or ''}", flush=True)
        print(f"== {name}: {ok}/{n}", flush=True)
    return out


# ---------------------------------------------------------------- grid: a closed loop
N = 7
CELL = 64
MOVES = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}


def bfs(walls, a, b):
    seen, q = {a: 0}, collections.deque([a])
    while q:
        p = q.popleft()
        if p == b:
            return seen[p]
        for dx, dy in MOVES.values():
            n = (p[0] + dx, p[1] + dy)
            if 0 <= n[0] < N and 0 <= n[1] < N and n not in walls and n not in seen:
                seen[n] = seen[p] + 1
                q.append(n)
    return None


def make_grid(rng, walled):
    while True:
        cells = [(x, y) for x in range(N) for y in range(N)]
        walls = set(rng.sample(cells, 11)) if walled else set()
        free = [c for c in cells if c not in walls]
        a, b = rng.sample(free, 2)
        dist = bfs(walls, a, b)
        manh = abs(a[0] - b[0]) + abs(a[1] - b[1])
        if dist and 5 <= dist <= 10 and (not walled or dist > manh):
            return walls, a, b, dist


def draw_grid(walls, robot, goal):
    img = Image.new("RGB", (N * CELL, N * CELL), "white")
    d = ImageDraw.Draw(img)
    for (x, y) in walls:
        d.rectangle([x * CELL, y * CELL, (x + 1) * CELL, (y + 1) * CELL], fill=(90, 90, 90))
    gx, gy = goal
    d.rectangle([gx * CELL + 6, gy * CELL + 6, (gx + 1) * CELL - 6, (gy + 1) * CELL - 6], fill=(40, 180, 70))
    rx, ry = robot
    d.ellipse([rx * CELL + 8, ry * CELL + 8, (rx + 1) * CELL - 8, (ry + 1) * CELL - 8], fill=(40, 90, 230))
    for k in range(N + 1):
        d.line([k * CELL, 0, k * CELL, N * CELL], fill=(170, 170, 170), width=1)
        d.line([0, k * CELL, N * CELL, k * CELL], fill=(170, 170, 170), width=1)
    return img


GRID_TOOLS = [tool("move", "Move the robot one cell", {"direction": {"type": "string", "enum": list(MOVES)}}, ["direction"])]
GRID_RULES = ("You steer the blue robot on a grid seen from above. Get it onto the green square. Dark gray "
              "cells are walls: the robot can't enter them or leave the grid. Call move with one direction "
              "(up, down, left or right); you'll see the new picture after each move.")


def image_part(img):
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{png_b64(img)}"}}


def longest_same_move(log):
    """Longest run of the same direction over the steps that made a move (no-call steps are skipped)."""
    dirs = [e["dir"] for e in log if e.get("dir")]
    return max((len(list(g)) for _, g in itertools.groupby(dirs)), default=0)


def run_episode(rng, walled, think, memory):
    walls, robot, goal, dist = make_grid(rng, walled)
    cap = 2 * dist + 8
    log, history, steps, blocked, nocall, invalid, secs = [], [], 0, 0, 0, 0, 0.0
    msgs = []
    while steps < cap and robot != goal:
        img = draw_grid(walls, robot, goal)
        if memory == "stateless":
            recent = ", ".join(history[-3:]) or "none yet"
            msgs = [{"role": "user", "content": [image_part(img), {"type": "text", "text":
                     f"{GRID_RULES}\nYour last moves, oldest first: {recent}."}]}]
        elif not msgs:
            msgs = [{"role": "user", "content": [image_part(img), {"type": "text", "text": GRID_RULES}]}]
        r = chat(msgs, GRID_TOOLS, think)
        secs += r["secs"]
        steps += 1
        call = r["calls"][0] if r["calls"] else None
        if call is None or not valid(call, GRID_TOOLS):
            nocall += call is None
            invalid += call is not None
            history.append("no valid move")
            log.append({"pos": robot, "call": call, "declined": r.get("declined"), "text": r["text"][:200],
                        "reasoning": r.get("reasoning", "")[-600:],
                        "error": r.get("error")})
            if memory == "history":
                msgs += [{"role": "assistant", "content": r["text"] or "(no call)"},
                         {"role": "user", "content": [image_part(img), {"type": "text", "text":
                          "No valid move was made. Call move with up, down, left or right."}]}]
            continue
        dx, dy = MOVES[call["args"]["direction"]]
        nxt = (robot[0] + dx, robot[1] + dy)
        ok = 0 <= nxt[0] < N and 0 <= nxt[1] < N and nxt not in walls
        blocked += not ok
        robot = nxt if ok else robot
        history.append(f"{call['args']['direction']} ({'moved' if ok else 'blocked'})")
        log.append({"pos": robot, "dir": call["args"]["direction"], "moved": ok,
                    "reasoning": r.get("reasoning", "")[-600:]})
        if memory == "history":
            raw = r["raw"]
            msgs += [{"role": "assistant", "content": raw.get("content") or "",
                      "tool_calls": [{"id": call["id"] or f"c{steps}", "type": "function", "function": {
                          "name": "move", "arguments": json.dumps(call["args"])}}]},
                     {"role": "tool", "tool_call_id": call["id"] or f"c{steps}",
                      "content": "moved" if ok else "blocked: a wall or the edge is there"},
                     {"role": "user", "content": [image_part(draw_grid(walls, robot, goal)),
                                                  {"type": "text", "text": "The new picture."}]}]
    longest = longest_same_move(log)
    return {"walled": walled, "optimal": dist, "cap": cap, "success": robot == goal, "steps": steps,
            "blocked": blocked, "no_call": nocall, "invalid": invalid, "longest_same_move": longest,
            "secs": round(secs, 1), "log": log}


def run_grid(key, think, memory, n):
    out = {"test": "grid", "key": key, "think": think, "memory": memory, "tool_choice": TOOL_CHOICE,
           "episodes": []}
    for walled in (False, True):
        for i in range(n):
            ep = run_episode(random.Random(f"grid-{walled}-{i}"), walled, think, memory)
            ep["i"] = i
            out["episodes"].append(ep)
            print(f"  {'walls' if walled else 'open '} {i} {'OK  ' if ep['success'] else 'FAIL'} steps {ep['steps']:2d}"
                  f" (best {ep['optimal']}, cap {ep['cap']}) blocked {ep['blocked']} no-call {ep['no_call']}"
                  f" same-move run {ep['longest_same_move']} {ep['secs']:.0f}s", flush=True)
    return out


def summary():
    import glob
    import statistics
    lines = ["# Visual control probe: results", "", "## glance: one picture, one decision", "",
             "| Model | Thinking | Tool call | " + " | ".join(SCENES) + " | All | Median time | Invalid calls |",
             "|---|---|---|" + "---:|" * (len(SCENES) + 3)]
    for p in sorted(glob.glob(os.path.join(RES, "glance-*.json"))):
        with open(p) as f:
            d = json.load(f)
        it = d["items"]
        cells = []
        for s in SCENES:
            xs = [x for x in it if x["scene"] == s]
            cells.append(f"{sum(x['ok'] for x in xs)}/{len(xs)}")
        lines.append(f"| {d['key']} | {'on' if d['think'] else 'off'} | {d.get('tool_choice', 'auto')} | "
                     + " | ".join(cells) +
                     f" | **{sum(x['ok'] for x in it)}/{len(it)}** | {statistics.median(x['secs'] for x in it):.1f} s"
                     f" | {sum(x['invalid_calls'] for x in it)} |")
    lines += ["", "## grid: steer a robot to the goal, one move per step", "",
              ("| Model | Thinking | Tool call | Memory | Open grid | With walls | Steps / best (solved) | Steps without a move | Blocked moves | "
               "Longest run of one move | Time per step |"), "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for p in sorted(glob.glob(os.path.join(RES, "grid-*.json"))):
        with open(p) as f:
            d = json.load(f)
        eps = d["episodes"]
        op = [e for e in eps if not e["walled"]]
        wa = [e for e in eps if e["walled"]]
        solved = [e["steps"] / e["optimal"] for e in eps if e["success"]]
        ratio = f"{statistics.median(solved):.2f}" if solved else "—"
        lines.append(f"| {d['key']} | {'on' if d['think'] else 'off'} | {d.get('tool_choice', 'auto')} | {d['memory']} | "
                     f"{sum(e['success'] for e in op)}/{len(op)} | {sum(e['success'] for e in wa)}/{len(wa)} | "
                     f"{ratio} | {sum(e['no_call'] + e['invalid'] for e in eps)}/{sum(e['steps'] for e in eps)} | "
                     f"{sum(e['blocked'] for e in eps)} | {max(longest_same_move(e['log']) for e in eps)} | "
                     f"{sum(e['secs'] for e in eps) / max(1, sum(e['steps'] for e in eps)):.1f} s |")
    with open(os.path.join(RES, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("test", choices=["glance", "grid", "examples", "summary"])
    ap.add_argument("key", nargs="?", default="model")
    ap.add_argument("--think", choices=["on", "off"], default="on")
    ap.add_argument("--memory", choices=["stateless", "history"], default="stateless")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--tool-choice", choices=["auto", "required"], default="auto",
                    help="required: the server only accepts a tool call as the answer")
    a = ap.parse_args()
    global TOOL_CHOICE
    TOOL_CHOICE = a.tool_choice
    req = "-required" if a.tool_choice == "required" else ""
    if a.test == "summary":
        return summary()
    if a.test == "examples":
        os.makedirs(os.path.join(HERE, "examples"), exist_ok=True)
        for name, gen in SCENES.items():
            gen(random.Random(f"{name}-1"), 1)[0].save(os.path.join(HERE, "examples", f"{name}.png"))
        w, r, g, _ = make_grid(random.Random("grid-True-0"), True)
        draw_grid(w, r, g).save(os.path.join(HERE, "examples", "grid.png"))
        return
    think = a.think == "on"
    os.makedirs(RES, exist_ok=True)
    if a.test == "glance":
        out = run_glance(a.key, think, a.n or 12)
        tag = f"glance-{a.key}-think-{a.think}{req}"
    else:
        out = run_grid(a.key, think, a.memory, a.n or 8)
        tag = f"grid-{a.key}-think-{a.think}-{a.memory}{req}"
    with open(os.path.join(RES, tag + ".json"), "w") as f:
        json.dump(out, f, indent=1)
    print("saved", os.path.join(RES, tag + ".json"))


if __name__ == "__main__":
    sys.exit(main())
