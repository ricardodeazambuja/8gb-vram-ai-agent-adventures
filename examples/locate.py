#!/usr/bin/env python3
"""Find things in a picture with LocateAnything-3B: a box (in pixels) for what you describe.

    python3 examples/locate.py photo.jpg "the red mug"                 one box: the best match, ALWAYS
                                                                       (even if there is no red mug)
    python3 examples/locate.py photo.jpg "person" --all                every match, or none: use this
                                                                       when the thing may be absent
    python3 examples/locate.py photo.jpg "cat,dog" --all --draw out.png    several things, boxes drawn

The model answers in its own tokens: <ref>label</ref><box><x1><y1><x2><y2></box>, on a 0-1000 scale;
this converts them to pixels. --draw needs Pillow. docs/models/locateanything-3b.md has its scores.
"""
import argparse
import re
import time
from pathlib import Path

import _server

REF = re.compile(r"<ref>([^<]*)</ref>((?:<box>(?:<\d+>){4}</box>|<box>None</box>)+)")
BOX = re.compile(r"<box>((?:<\d+>){4})</box>")


def image_size(path):
    try:
        from PIL import Image
        with Image.open(path) as im:
            return im.size
    except ImportError:
        import struct
        head = Path(path).read_bytes()[:24]
        if head[:8] == b"\x89PNG\r\n\x1a\n":
            return struct.unpack(">II", head[16:24])
        raise SystemExit("can't read the picture's size without Pillow (pip install pillow), except for PNG") from None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", type=Path)
    ap.add_argument("what", help="a description; with --all, several separated by commas")
    ap.add_argument("--all", action="store_true", help="every instance, not just the best one")
    ap.add_argument("--draw", type=Path, help="save a copy of the picture with the boxes drawn")
    _server.add_server_args(ap)
    a = ap.parse_args()
    w, h = image_size(a.image)
    _server.ensure("locate", a)

    if a.all:
        prompt = ("Locate all the instances that matches the following description: "
                  + "</c>".join(s.strip() for s in a.what.split(",")) + ".")
    else:
        prompt = f"Locate a single instance that matches the following description: {a.what}."
    t = time.perf_counter()
    r = _server.post("/v1/chat/completions", {"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": _server.data_url(a.image)}},
        {"type": "text", "text": prompt}]}], "max_tokens": 512, "temperature": 0})
    text = r["choices"][0]["message"].get("content") or ""
    found = []
    for label, boxes in REF.findall(text):
        for b in BOX.findall(boxes):
            x1, y1, x2, y2 = (int(v) for v in re.findall(r"\d+", b))
            found.append((label, round(x1 * w / 1000), round(y1 * h / 1000), round(x2 * w / 1000), round(y2 * h / 1000)))
    for label, *box in found:
        print(f"{label}: x1={box[0]} y1={box[1]} x2={box[2]} y2={box[3]}")
    if not found:
        print(f"nothing found ({text.strip()[:200] or 'empty answer'})")
    print(f"[{time.perf_counter() - t:.1f} s, picture {w}x{h}]")
    if a.draw and found:
        from PIL import Image, ImageDraw
        with Image.open(a.image) as im:
            im = im.convert("RGB")
            d = ImageDraw.Draw(im)
            for label, *box in found:
                d.rectangle(box, outline=(255, 40, 40), width=max(2, w // 300))
                d.text((box[0] + 4, box[1] + 2), label, fill=(255, 40, 40))
            im.save(a.draw)
        print(a.draw)


if __name__ == "__main__":
    main()
