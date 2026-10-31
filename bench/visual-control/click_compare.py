#!/usr/bin/env python3
"""Who clicks better: two-step LocateAnything or tuned LFM3-VL? Same dialogs, same real pages.

  python3 click_compare.py locate CAPTURE_DIR    # detect-all first, then the single-instance box
  python3 click_compare.py lfm3vl CAPTURE_DIR    # the click tool, thinking off

Start the matching llama-server first. Score: the click lands on the element (within 4 px), and no
click when the target isn't there. CAPTURE_DIR comes from `locate_probe.py capture`.
"""
import base64
import json
import random
import re
import statistics
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "finetune"))
import locate_probe as lp
import run_control as rc
import scenes_new

ALL = "Locate all the instances that matches the following description: {}."


def ask(png, prompt):
    body = {"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}},
        {"type": "text", "text": prompt}]}], "max_tokens": 96, "temperature": 0}
    req = urllib.request.Request(rc.URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def click_locate(png, name, kind):
    """(x, y) in 0-1000, or None to decline."""
    if not lp.BOX.search(ask(png, ALL.format(f"{name} {kind}"))):
        return None
    box, _, _ = lp.locate(png, f"the '{name}' {kind}")
    return None if box is None else ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def click_lfm3(png, name, kind):
    tools = [rc.tool("click", "Click at a point of the screen", rc.POINT, ["x", "y"]),
             rc.tool("type_text", "Type text into the focused field", {"text": {"type": "string"}}, ["text"])]
    prompt = (f"This is a screenshot of a dialog. Use the click tool to click the '{name}' {kind} (x and y are "
              f"0-1000 coordinates of the image). If there is no such {kind}, don't click anything and say so.")
    msgs = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}},
        {"type": "text", "text": prompt}]}]
    calls = [c for c in rc.chat(msgs, tools, False)["calls"] if c["name"] == "click" and isinstance(c["args"], dict)]
    try:
        return float(calls[0]["args"]["x"]), float(calls[0]["args"]["y"])
    except (IndexError, KeyError, TypeError, ValueError):
        return None


def main():
    who, folder = sys.argv[1], sys.argv[2]
    click = click_locate if who == "locate" else click_lfm3
    items, secs = [], []
    for setname, gen in (("dialog", rc.scene_click), ("dialog-new", scenes_new.scene_click)):
        for i in range(30):
            img, _, _, score, truth = gen(random.Random(f"click-{i}"), i)
            png = base64.b64decode(rc.png_b64(img))
            t = time.time()
            p = click(png, truth["target"], "button")
            secs.append(time.time() - t)
            calls = [] if p is None else [{"name": "click", "args": {"x": p[0], "y": p[1]}}]
            items.append({"set": setname, "absent": truth["absent"], "ok": bool(score({"calls": calls})), "point": p})
    for name in lp.PAGES:
        path = Path(folder, f"{name}.json")
        if not path.exists():
            continue
        els = json.loads(path.read_text())["elements"]
        png = Path(folder, f"{name}.png").read_bytes()
        rng = random.Random(name)  # the same targets as locate_probe.py web
        for e in rng.sample(els, min(8, len(els))):
            t = time.time()
            p = click(png, re.sub(r"\s+", " ", e["text"]), "link or button")
            secs.append(time.time() - t)
            ok = False
            if p:
                x, y = p[0] * lp.VIEW["width"] / 1000, p[1] * lp.VIEW["height"] / 1000
                ok = e["box"][0] - 4 <= x <= e["box"][2] + 4 and e["box"][1] - 4 <= y <= e["box"][3] + 4
            items.append({"set": "web", "page": name, "absent": False, "ok": ok, "point": p, "query": e["text"]})
        for what in rng.sample(lp.ABSENT, 2):
            t = time.time()
            p = click(png, what, "link or button")
            secs.append(time.time() - t)
            items.append({"set": "web", "page": name, "absent": True, "ok": p is None, "point": p, "query": what})
    for s in ("dialog", "dialog-new", "web"):
        for absent in (False, True):
            xs = [i["ok"] for i in items if i["set"] == s and i["absent"] == absent]
            print(f"== {who} {s} {'absent ' if absent else 'present'}: {sum(xs)}/{len(xs)}")
    print(f"== {who} median {statistics.median(secs):.2f} s")
    Path(rc.RES, f"click-{who}.json").write_text(json.dumps(items, indent=1))


if __name__ == "__main__":
    main()
