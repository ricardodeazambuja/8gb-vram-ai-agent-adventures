#!/usr/bin/env python3
"""Public-domain training pictures from Wikimedia Commons, with a manifest of where each came from.

Usage: fetch_commons.py [--n 24]
Pictures go to models/image-gen/train/<style>/ (git-ignored). With a manifest in bench/imagegen/data/, it
fetches exactly the pictures listed and checks their sha256, so a clone gets our training set. Delete a
manifest to choose a new set.
"""
import argparse
import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "models/image-gen/train"
DATA = Path(__file__).parent / "data"
API = "https://commons.wikimedia.org/w/api.php"
UA = {"User-Agent": "8gb-vram-adventures-bench/0.1"}
STYLES = {
    "vangogh": "Category:Featured pictures of paintings by Vincent van Gogh",
    "gris": '"Juan Gris" painting filetype:bitmap',
    "atget": "Category:Photographs by Eugène Atget",
}


def get(url):
    req = urllib.request.Request(url, headers=UA)
    for wait in (30, 60, 120, 240):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code != 429:
                raise
            time.sleep(wait)
    raise RuntimeError(f"still rate limited: {url}")


def members(source):
    q = {"action": "query", "format": "json", "prop": "imageinfo",
         "iiprop": "url|size|mime|extmetadata", "iiurlwidth": "1280"}
    if source.startswith("Category:"):
        q |= {"generator": "categorymembers", "gcmtitle": source, "gcmtype": "file", "gcmlimit": "500"}
    else:
        q |= {"generator": "search", "gsrsearch": source, "gsrnamespace": "6", "gsrlimit": "200"}
    return json.loads(get(f"{API}?{urllib.parse.urlencode(q)}"))["query"]["pages"].values()


def restore(style, rows):
    """Fetch the pictures a manifest lists. Commons can re-render a thumbnail, so a changed file is reported."""
    out = TRAIN / style
    out.mkdir(parents=True, exist_ok=True)
    changed = 0
    for r in rows:
        path = out / r["file"]
        if not path.exists():
            q = {"action": "query", "format": "json", "prop": "imageinfo", "iiprop": "url",
                 "iiurlwidth": "1280", "titles": r["title"]}
            page = next(iter(json.loads(get(f"{API}?{urllib.parse.urlencode(q)}"))["query"]["pages"].values()))
            path.write_bytes(get(page["imageinfo"][0]["thumburl"]))
            time.sleep(6)
        if hashlib.sha256(path.read_bytes()).hexdigest() != r["sha256"]:
            changed += 1
            print("differs from our copy:", style, r["file"], r["title"], flush=True)
    print(style, len(rows), "pictures,", changed, "differ", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    a = ap.parse_args()
    DATA.mkdir(exist_ok=True)
    for style, cat in STYLES.items():
        manifest = DATA / f"{style}.json"
        if manifest.exists():
            restore(style, json.loads(manifest.read_text()))
            continue
        out = TRAIN / style
        out.mkdir(parents=True, exist_ok=True)
        rows, usable = [], []
        for p in sorted(members(cat), key=lambda x: x["title"]):
            info = p.get("imageinfo", [{}])[0]
            licence = info.get("extmetadata", {}).get("LicenseShortName", {}).get("value", "")
            if info.get("mime") != "image/jpeg" or min(info["width"], info["height"]) < 700:
                continue
            if style == "gris" and not re.search(r"gris", p["title"], re.IGNORECASE):
                continue
            if re.search(r"public domain|^PD|CC0", licence, re.IGNORECASE):
                usable.append((p, info, licence))
        # evenly spaced: an alphabetical list gives 24 details of one church
        for p, info, licence in usable[:: max(1, len(usable) // a.n)]:
            name = f"{len(rows):02d}.jpg"
            (out / name).write_bytes(get(info["thumburl"]))
            rows.append({"file": name, "title": p["title"], "licence": licence, "page": info["descriptionurl"],
                         "sha256": hashlib.sha256((out / name).read_bytes()).hexdigest()})
            time.sleep(6)
            if len(rows) == a.n:
                break
        (DATA / f"{style}.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))
        print(style, len(rows), flush=True)


if __name__ == "__main__":
    main()
