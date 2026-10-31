"""The five glance scenes, same rules, prompts and tools, drawn differently from anything in training.

Used by `train_glance.py eval --new`: if a tuned model only memorised run_control.py's drawing code,
this is where it shows. Layouts, sizes, palettes and most button names all change; nothing here is
ever trained on.
"""
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "visual-control"))
import run_control as rc

MORE_BUTTONS = ["Open", "Close", "Apply", "Reset", "Retry", "Next", "Back", "Search", "Upload", "Rename"]
DARK = (24, 28, 36)
SHADES = {"red": (250, 90, 110), "blue": (70, 180, 250), "green": (130, 220, 90), "yellow": (250, 240, 110),
          "purple": (190, 120, 240), "orange": (255, 170, 80)}


def scene_tanks(rng, i):
    """Horizontal bars stacked on a dark background, filling left to right, in order C, A, B."""
    img = Image.new("RGB", (760, 460), DARK)
    d = ImageDraw.Draw(img)
    none_low = i % 4 == 0
    levels = {}
    for name in "ABC":
        low = (not none_low) and rng.random() < 0.45
        levels[name] = rng.randint(5, 22) if low else rng.randint(38, 95)
    if not none_low and all(v >= 30 for v in levels.values()):
        levels[rng.choice("ABC")] = rng.randint(5, 22)
    for k, name in enumerate("CAB"):
        x0, x1, y0, y1 = 110, 700, 50 + k * 135, 130 + k * 135
        d.rectangle([x0, y0, x0 + (x1 - x0) * levels[name] / 100, y1], fill=(240, 150, 60))
        d.rectangle([x0, y0, x1, y1], outline="white", width=2)
        d.text((40, y0 + 22), name, fill="white", font=rc.font(30, True))
        for pct in (0, 25, 50, 75, 100):
            xx = x0 + (x1 - x0) * pct / 100
            d.line([xx, y1, xx, y1 + 8], fill="white", width=2)
            d.text((xx - 16, y1 + 10), f"{pct}%", fill="white", font=rc.font(14))
    _, prompt, tools, _, _ = rc.scene_tanks(rng, i)
    want = sorted(n for n, v in levels.items() if v < 30)

    def score(r):
        return sorted(c["args"].get("tank") for c in r["calls"]
                      if c["name"] == "refill" and isinstance(c["args"], dict)) == want
    return img, prompt, tools, score, {"levels": levels, "want": want}


def scene_click(rng, i):
    """A dark toolbar-style window: pill buttons of varied width in a loose grid, with other names."""
    W, H = 960, 540
    img = Image.new("RGB", (W, H), DARK)
    d = ImageDraw.Draw(img)
    d.text((24, 16), "Preferences", fill=(160, 170, 190), font=rc.font(16))
    names = rc.BUTTONS + MORE_BUTTONS
    labels = rng.sample(names, rng.randint(4, 7))
    absent = i % 5 == 0
    target = rng.choice([b for b in names if b not in labels]) if absent else rng.choice(labels)
    cells = rng.sample([(c, r) for c in range(4) for r in range(3)], len(labels))
    boxes = {}
    for lab, (c, r) in zip(labels, cells, strict=True):
        w, h = rng.randint(120, 190), rng.randint(40, 60)
        x, y = 40 + c * 225 + rng.randint(0, 20), 80 + r * 150 + rng.randint(0, 50)
        box = (x, y, x + w, y + h)
        boxes[lab] = box
        d.rounded_rectangle(box, h // 2, fill=rng.choice([(50, 110, 220), (60, 150, 110), (150, 70, 160)]))
        tw = d.textlength(lab, font=rc.font(19, True))
        d.text(((box[0] + box[2] - tw) / 2, y + h / 2 - 12), lab, fill="white", font=rc.font(19, True))
    tools = [rc.tool("click", "Click at a point of the screen", rc.POINT, ["x", "y"]),
             rc.tool("type_text", "Type text into the focused field", {"text": {"type": "string"}}, ["text"])]
    prompt = (f"This is a screenshot of a dialog. Use the click tool to click the '{target}' button (x and y "
              f"are 0-1000 coordinates of the image). If there is no such button, don't click anything and "
              f"say so.")

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


def scene_pick(rng, i):
    """A wide slate table, lighter shades, a dotted texture, no outlines."""
    W, H = 900, 520
    img = Image.new("RGB", (W, H), (58, 66, 78))
    d = ImageDraw.Draw(img)
    for x in range(0, W, 30):
        for y in range(0, H, 30):
            d.point((x, y), fill=(95, 105, 120))
    extreme = None if i % 2 == 0 else rng.choice([("leftmost", "x", False), ("rightmost", "x", True),
                                                  ("topmost", "y", False), ("bottommost", "y", True)])
    while True:
        combos = rng.sample([(c, s) for c in SHADES for s in rc.SHAPES], rng.randint(4, 6))
        objs = []
        for c, s in combos:
            for _ in range(500):
                r = rng.randint(24, 58)
                x, y = rng.randint(r + 15, W - r - 15), rng.randint(r + 15, H - r - 15)
                if all(abs(x - o["x"]) > r + o["r"] + 20 or abs(y - o["y"]) > r + o["r"] + 20 for o in objs):
                    break
            objs.append({"color": c, "shape": s, "x": x, "y": y, "r": r})
        if extreme is None:
            break
        first, second = sorted(objs, key=lambda o: o[extreme[1]], reverse=extreme[2])[:2]
        if abs(first[extreme[1]] - second[extreme[1]]) >= 60:
            break
    for o in objs:
        x, y, r = o["x"], o["y"], o["r"]
        box = [x - r, y - r, x + r, y + r]
        if o["shape"] == "circle":
            d.ellipse(box, fill=SHADES[o["color"]])
        elif o["shape"] == "square":
            d.rectangle(box, fill=SHADES[o["color"]])
        else:
            d.polygon([(x, y - r), (x - r, y + r), (x + r, y + r)], fill=SHADES[o["color"]])
    t = rng.choice(objs) if extreme is None else first
    ask = f"the {t['color']} {t['shape']}" if extreme is None else f"the {extreme[0]} object"
    tools = [rc.tool("pick", "Move the robot gripper to a point of the picture and grasp", rc.POINT, ["x", "y"])]
    prompt = (f"You control a robot arm above this table (camera view from the top). Use the pick tool to pick "
              f"up {ask} (x and y are 0-1000 coordinates of the image).")

    def score(r):
        p = [c for c in r["calls"] if c["name"] == "pick" and isinstance(c["args"], dict)]
        if len(r["calls"]) != 1 or not p:
            return False
        try:
            x, y = p[0]["args"]["x"] * W / 1000, p[0]["args"]["y"] * H / 1000
        except (KeyError, TypeError):
            return False
        return abs(x - t["x"]) <= t["r"] + 4 and abs(y - t["y"]) <= t["r"] + 4
    return img, prompt, tools, score, {"ask": ask, "target": t, "objects": objs}


def scene_signal(rng, i):
    """Night, a horizontal light hung on the left, a person in another colour and on either side."""
    img = Image.new("RGB", (760, 460), (20, 24, 50))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 290, 760, 460], fill=(45, 45, 55))
    for k in range(9):
        d.rectangle([120 + k * 60, 340, 150 + k * 60, 440], fill=(200, 200, 200))
    light = rng.choice(["red", "yellow", "green"])
    person = rng.random() < 0.35
    d.rectangle([40, 50, 230, 115], fill=(90, 90, 100))
    d.rectangle([40, 0, 50, 50], fill=(90, 90, 100))
    for k, (name, col) in enumerate([("red", (255, 50, 50)), ("yellow", (255, 220, 40)), ("green", (40, 240, 90))]):
        d.ellipse([52 + k * 60, 60, 98 + k * 60, 106], fill=col if name == light else (35, 35, 40))
    if person:
        px, c = rng.randint(160, 640), (240, 240, 120)
        d.ellipse([px - 16, 262, px + 16, 294], fill=c)
        d.line([px, 294, px, 360], fill=c, width=8)
        d.line([px, 310, px - 24, 340], fill=c, width=6)
        d.line([px, 310, px + 24, 340], fill=c, width=6)
        d.line([px, 360, px - 18, 410], fill=c, width=6)
        d.line([px, 360, px + 18, 410], fill=c, width=6)
    want = "stop" if light == "red" or person else ("slow_down" if light == "yellow" else "go")
    _, prompt, tools, _, _ = rc.scene_signal(rng, i)

    def score(r):
        return len(r["calls"]) == 1 and r["calls"][0]["name"] == want
    return img, prompt, tools, score, {"light": light, "person": person, "want": want}


def scene_gauge(rng, i):
    """A dark round dial with a 270-degree sweep, instead of a white half circle."""
    S = 480
    img = Image.new("RGB", (S, S), (235, 230, 215))
    d = ImageDraw.Draw(img)
    cx, cy, R = S / 2, S / 2, 205
    kind = i % 3
    val = rng.randint(76, 97) if kind == 0 else rng.randint(3, 14) if kind == 1 else rng.randint(26, 64)
    d.ellipse([cx - R - 12, cy - R - 12, cx + R + 12, cy + R + 12], fill=(30, 34, 44))

    def angle(v):
        return math.radians(225 - 270 * v / 100)
    for v in range(0, 101, 5):
        a, r0 = angle(v), R - (22 if v % 10 == 0 else 10)
        d.line([cx + r0 * math.cos(a), cy - r0 * math.sin(a), cx + R * math.cos(a), cy - R * math.sin(a)],
               fill="white", width=3 if v % 10 == 0 else 1)
        if v % 10 == 0:
            d.text((cx + (R - 44) * math.cos(a) - 12, cy - (R - 44) * math.sin(a) - 9), str(v), fill="white",
                   font=rc.font(16))
    a = angle(val)
    d.line([cx, cy, cx + (R - 34) * math.cos(a), cy - (R - 34) * math.sin(a)], fill=(255, 190, 40), width=6)
    d.ellipse([cx - 12, cy - 12, cx + 12, cy + 12], fill=(255, 190, 40))
    d.text((cx - 36, cy + 110), "kPa x10", fill="white", font=rc.font(16, True))
    want = "open_valve" if val > 70 else "close_valve" if val < 20 else "report"
    _, prompt, tools, _, _ = rc.scene_gauge(rng, i)

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
