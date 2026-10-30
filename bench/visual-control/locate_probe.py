#!/usr/bin/env python3
"""LocateAnything-3B on the click and pick scenes, and on real web pages, scored as points.

It has no tool calls: it answers with a box, so the "click" is the box's centre, fed to the same
scorers as run_control.py (same seeds). <box>None</box> counts as not clicking.

  python3 locate_probe.py capture DIR      # render the real pages, take element boxes from the DOM
  python3 locate_probe.py scenes [N]       # click + pick, N per kind (default 30)
  python3 locate_probe.py web DIR          # the pages captured in DIR

Needs llama-server on :8080 (QMODEL=locate bash scripts/run-qwen.sh --image). Screenshots stay out
of the repo; only scores are saved.
"""
import base64
import json
import os
import random
import re
import sys
import time
import urllib.request
from pathlib import Path

import run_control as rc

PAGES = {"wikipedia": "https://en.wikipedia.org/wiki/Laptop",
         "hackernews": "https://news.ycombinator.com/",
         "github": "https://github.com/ggml-org/llama.cpp",
         "huggingface": "https://huggingface.co/nvidia/LocateAnything-3B",
         "pypi": "https://pypi.org/project/pillow/",
         "python-docs": "https://docs.python.org/3/library/json.html"}
VIEW = {"width": 1280, "height": 800}
ABSENT = ["Checkout", "Unsubscribe", "Add to cart", "Print label", "Reboot now"]
BOX = re.compile(r"<box>((?:<\d+>){4})</box>")


def locate(png_bytes, what):
    body = {"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png_bytes).decode()}},
        {"type": "text", "text": f"Locate a single instance that matches the following description: {what}."}]}],
        "max_tokens": 64, "temperature": 0}
    req = urllib.request.Request(rc.URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        text = json.load(r)["choices"][0]["message"]["content"]
    found = [[int(n) for n in re.findall(r"\d+", b)] for b in BOX.findall(text)]
    return (found[0] if found else None), text, time.time() - t


def run_scenes(n):
    out = {"key": "locate", "items": []}
    for name, gen, tool_name, query in (
            ("click", rc.scene_click, "click", lambda t: f"the '{t['target']}' button"),
            ("pick", rc.scene_pick, "pick", lambda t: t["ask"])):
        ok = 0
        for i in range(n):
            img, _, _, score, truth = gen(random.Random(f"{name}-{i}"), i)
            box, text, secs = locate(base64.b64decode(rc.png_b64(img)), query(truth))
            calls = [] if box is None else [{"name": tool_name, "args": {"x": (box[0] + box[2]) / 2,
                                                                         "y": (box[1] + box[3]) / 2}}]
            good = bool(score({"calls": calls}))
            ok += good
            out["items"].append({"scene": name, "i": i, "ok": good, "query": query(truth), "box": box,
                                 "text": text[:120], "secs": round(secs, 2)})
            print(f"  {name} {i:2d} {'ok  ' if good else 'MISS'} {secs:.1f}s {query(truth)!r} {box}", flush=True)
        print(f"== {name}: {ok}/{n}", flush=True)
    return out


def capture(folder):
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright
    js = """() => [...document.querySelectorAll('a,button,[role=button],input[type=submit]')].map(e => {
        const r = e.getBoundingClientRect(), s = getComputedStyle(e);
        return {text: (e.innerText || e.value || '').trim(), kind: e.tagName.toLowerCase(),
                box: [r.left, r.top, r.right, r.bottom],
                shown: s.visibility === 'visible' && s.display !== 'none' && r.width > 12 && r.height > 8};
    })"""
    os.makedirs(folder, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for name, url in PAGES.items():
            page = browser.new_page(viewport=VIEW)
            try:
                page.goto(url, timeout=45000, wait_until="load")
                page.wait_for_timeout(1500)
            except PlaywrightError as e:
                print(f"{name}: skipped, {type(e).__name__}", flush=True)
                continue
            els = [e for e in page.evaluate(js) if e["shown"] and e["box"][0] >= 0 and e["box"][1] >= 0
                   and e["box"][2] <= VIEW["width"] and e["box"][3] <= VIEW["height"]
                   and 2 <= len(e["text"]) <= 40 and "\n" not in e["text"]]
            names = [e["text"] for e in page.evaluate(js) if e["shown"]]
            els = [e for e in els if names.count(e["text"]) == 1]  # visible and named once
            page.screenshot(path=f"{folder}/{name}.png")
            Path(f"{folder}/{name}.json").write_text(json.dumps({"url": url, "elements": els}, indent=1))
            print(f"{name}: {len(els)} elements", flush=True)
            page.close()
        browser.close()


def run_web(folder):
    out = {"key": "locate", "items": []}
    hits = total = absent_ok = absent_n = 0
    for name in PAGES:
        if not os.path.exists(f"{folder}/{name}.json"):
            continue
        data = json.loads(Path(f"{folder}/{name}.json").read_text())
        png = Path(f"{folder}/{name}.png").read_bytes()
        rng = random.Random(name)
        els = rng.sample(data["elements"], min(8, len(data["elements"])))
        for e in els:
            box, _, secs = locate(png, f"the '{e['text']}' link or button")
            ok = False
            if box:
                x = (box[0] + box[2]) / 2 * VIEW["width"] / 1000
                y = (box[1] + box[3]) / 2 * VIEW["height"] / 1000
                ok = e["box"][0] - 4 <= x <= e["box"][2] + 4 and e["box"][1] - 4 <= y <= e["box"][3] + 4
            hits += ok
            total += 1
            out["items"].append({"page": name, "query": e["text"], "ok": ok, "box": box, "secs": round(secs, 2)})
            print(f"  {name} {'ok  ' if ok else 'MISS'} {secs:.1f}s {e['text']!r} {box}", flush=True)
        for what in rng.sample(ABSENT, 2):
            box, _, secs = locate(png, f"the '{what}' link or button")
            absent_ok += box is None
            absent_n += 1
            out["items"].append({"page": name, "query": what, "absent": True, "ok": box is None, "box": box,
                                 "secs": round(secs, 2)})
            print(f"  {name} {'ok  ' if box is None else 'MISS'} absent {what!r} {box}", flush=True)
    print(f"== present: {hits}/{total}; absent declined: {absent_ok}/{absent_n}", flush=True)
    return out


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "capture":
        capture(sys.argv[2])
    else:
        res = run_scenes(int(sys.argv[2]) if len(sys.argv) > 2 else 30) if cmd == "scenes" else run_web(sys.argv[2])
        Path(rc.RES, f"locate-{cmd}.json").write_text(json.dumps(res, indent=1))
