#!/usr/bin/env python3
"""LocateAnything on the vision probe's three shape images: one query per shape, then one detect-all.

    locate_shapes.py   # needs QMODEL=locate with --image on :8080
"""
import base64
import json
import pathlib
import re
import time
import urllib.request

REPO = str(pathlib.Path(__file__).resolve().parents[2])
with open(f"{REPO}/bench/vision/items.json") as f:
    ITEMS = [i for i in json.load(f) if i["scorer"] == "bbox"]
W, H = 1024, 768
BOX = re.compile(r"<ref>([^<]*)</ref>((?:<box>(?:<\d+>){4}</box>)+)")


def ask(img, prompt):
    with open(f"{REPO}/bench/vision/items/{img}", "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    body = {"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        {"type": "text", "text": prompt}]}], "max_tokens": 512, "temperature": 0}
    req = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        out = json.load(r)
    return out["choices"][0]["message"]["content"], time.time() - t


def boxes(text):
    found = []
    for label, blk in BOX.findall(text):
        for b in re.findall(r"<box>((?:<\d+>){4})</box>", blk):
            x1, y1, x2, y2 = (int(n) for n in re.findall(r"<(\d+)>", b))
            found.append((label, [x1 * W / 1000, y1 * H / 1000, x2 * W / 1000, y2 * H / 1000]))
    return found


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


single, secs = [], []
for it in ITEMS:
    for obj in it["truth"]:
        q = f"{obj['color']} {obj['label']}"
        text, dt = ask(it["image"], f"Locate a single instance that matches the following description: {q}.")
        got = boxes(text)
        v = max((iou(b, obj["bbox"]) for _, b in got), default=0.0)
        single.append(v)
        secs.append(dt)
        print(f"{it['id']} {q:16s} iou={v:.2f} {dt:.2f}s {text[:90]!r}", flush=True)
    text, dt = ask(it["image"], "Locate all the instances that matches the following description: "
                                "circle</c>square</c>triangle</c>star.")
    got = boxes(text)
    hits = sum(any(iou(b, o["bbox"]) >= 0.5 and lab == o["label"] for lab, b in got) for o in it["truth"])
    print(f"{it['id']} DETECT-ALL {hits}/{len(it['truth'])} found, {len(got)} boxes, {dt:.2f}s {text[:200]!r}")
print(f"== single: {sum(v >= 0.5 for v in single)}/{len(single)} at IoU>=0.5, mean IoU {sum(single) / len(single):.3f}, "
      f"median {sorted(secs)[len(secs) // 2]:.2f}s")
