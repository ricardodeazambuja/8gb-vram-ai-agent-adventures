#!/usr/bin/env python3
"""Generate the vision probe: synthetic images with exact ground truth.

    python3 bench/vision/gen_images.py        # -> bench/vision/items/*.png + items.json

Four categories, three images each, every one drawn from a fixed seed so the set is identical on
every run: OCR (invoices), bounding boxes (coloured shapes), understanding (bar chart, counting,
spatial layout) and CAD (dimensioned drawings of plates with holes).
"""
import json
import math
import os
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402
from PIL import Image, ImageDraw, ImageFilter, ImageFont  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "items")
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_M = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
JSON_ONLY = "Answer with JSON only, no other text."
items = []


def font(path, size):
    return ImageFont.truetype(path, size)


# --- OCR: invoices -------------------------------------------------------------------------------
PRODUCTS = ["Hex bolt M8x40", "Steel washer 8mm", "Hinge, brass 75mm", "Cable tie 200mm (pack)",
            "Wood screw 4x30", "Wall plug 6mm", "Drill bit HSS 5mm", "Sandpaper P120 (sheet)",
            "Masking tape 25mm", "Pipe clamp 22mm"]
COMPANIES = ["Northwind Hardware Ltd.", "Bluegate Supplies GmbH", "Oakridge Tools & Co."]


def gen_invoice(i):
    rng = random.Random(100 + i)
    w, h = 900, 1100
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    inv = f"INV-{rng.randint(10000, 99999)}"
    date = f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
    d.text((60, 50), COMPANIES[i], font=font(FONT_B, 34), fill="black")
    d.text((60, 100), "12 Industrial Way, Springfield", font=font(FONT, 20), fill="#444")
    d.text((560, 170), "INVOICE", font=font(FONT_B, 40), fill="#1a3d7c")
    d.text((560, 230), f"Invoice no: {inv}", font=font(FONT, 22), fill="black")
    d.text((560, 262), f"Date: {date}", font=font(FONT, 22), fill="black")
    y = 360
    d.rectangle([60, y, 840, y + 40], fill="#e8eef8")
    for x, label in ((70, "Description"), (520, "Qty"), (600, "Unit"), (730, "Amount")):
        d.text((x, y + 8), label, font=font(FONT_B, 20), fill="black")
    y += 55
    lines, total = [], 0
    for p in rng.sample(PRODUCTS, rng.randint(3, 5)):
        qty = rng.randint(1, 40)
        unit = rng.choice([0.15, 0.35, 1.2, 2.75, 4.5, 8.9, 12.0])
        amount = round(qty * unit, 2)
        total = round(total + amount, 2)
        lines.append({"description": p, "qty": qty, "amount": amount})
        d.text((70, y), p, font=font(FONT, 20), fill="black")
        d.text((520, y), str(qty), font=font(FONT_M, 20), fill="black")
        d.text((600, y), f"{unit:.2f}", font=font(FONT_M, 20), fill="black")
        d.text((730, y), f"{amount:.2f}", font=font(FONT_M, 20), fill="black")
        y += 42
    d.line([60, y + 10, 840, y + 10], fill="black", width=2)
    d.text((560, y + 30), "TOTAL (EUR)", font=font(FONT_B, 24), fill="black")
    d.text((730, y + 30), f"{total:.2f}", font=font(FONT_B, 24), fill="black")
    d.text((60, h - 80), "Payment due within 30 days. Thank you for your business!", font=font(FONT, 18), fill="#666")
    if i == 2:  # the third one is a "scan": slightly rotated, blurred and grey
        img = img.rotate(1.5, resample=Image.BICUBIC, expand=False, fillcolor="white")
        img = img.filter(ImageFilter.GaussianBlur(0.8)).convert("L").convert("RGB")
    path = f"ocr-invoice-{i + 1}.png"
    img.save(os.path.join(OUT, path))
    items.append({
        "id": f"ocr-invoice-{i + 1}", "category": "ocr", "image": path, "scorer": "invoice",
        "prompt": ("Read this invoice. Return a JSON object with the keys invoice_number, date "
                   "(YYYY-MM-DD), total (a number) and items: a list of objects with description, "
                   "qty (an integer) and amount (a number), in the order they appear. " + JSON_ONLY),
        "truth": {"invoice_number": inv, "date": date, "total": total, "items": lines},
    })


# --- bounding boxes: coloured shapes ----------------------------------------------------------------
COLORS = {"red": (220, 40, 40), "blue": (40, 90, 220), "green": (30, 160, 70), "orange": (245, 150, 20),
          "purple": (140, 60, 180), "black": (20, 20, 20)}
SHAPES = ["circle", "square", "triangle", "star"]


def star_points(cx, cy, r):
    pts = []
    for k in range(10):
        rr = r if k % 2 == 0 else r * 0.45
        a = -math.pi / 2 + k * math.pi / 5
        pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    return pts


def gen_shapes(i):
    rng = random.Random(200 + i)
    w, h = 1024, 768
    img = Image.new("RGB", (w, h), (245, 243, 236))
    d = ImageDraw.Draw(img)
    objs, placed = [], []
    n = [4, 5, 6][i]
    colors = rng.sample(list(COLORS), n)
    while len(objs) < n:
        shape = rng.choice(SHAPES)
        r = rng.randint(40, 95)
        cx, cy = rng.randint(r + 10, w - r - 10), rng.randint(r + 10, h - r - 10)
        if any(math.hypot(cx - px, cy - py) < r + pr + 25 for px, py, pr in placed):
            continue
        placed.append((cx, cy, r))
        col = colors[len(objs)]
        if shape == "circle":
            box = [cx - r, cy - r, cx + r, cy + r]
            d.ellipse(box, fill=COLORS[col])
        elif shape == "square":
            box = [cx - r, cy - r, cx + r, cy + r]
            d.rectangle(box, fill=COLORS[col])
        elif shape == "triangle":
            pts = [(cx, cy - r), (cx - r, cy + r * 0.8), (cx + r, cy + r * 0.8)]
            d.polygon(pts, fill=COLORS[col])
            box = [cx - r, cy - r, cx + r, cy + r * 0.8]
        else:
            pts = star_points(cx, cy, r)
            d.polygon(pts, fill=COLORS[col])
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            box = [min(xs), min(ys), max(xs), max(ys)]
        objs.append({"label": shape, "color": col, "bbox": [round(v) for v in box]})
    path = f"bbox-shapes-{i + 1}.png"
    img.save(os.path.join(OUT, path))
    items.append({
        "id": f"bbox-shapes-{i + 1}", "category": "bbox", "image": path, "scorer": "bbox",
        "size": [w, h],
        "prompt": (f"The image is {w}x{h} pixels and shows several coloured shapes (circle, square, "
                   "triangle or star). For every shape return an object with label (the shape), color, "
                   "and bbox: [x_min, y_min, x_max, y_max] in pixel coordinates of this image, with "
                   "(0, 0) at the top-left. Return a JSON list. " + JSON_ONLY),
        "truth": objs,
    })


# --- understanding: chart, counting, spatial layout --------------------------------------------------
def gen_chart():
    rng = random.Random(301)
    cats = ["Lisbon", "Oslo", "Quito", "Hanoi", "Perth", "Dakar"]
    vals = [rng.randint(12, 95) for _ in cats]
    fig, ax = plt.subplots(figsize=(9, 6), dpi=110)
    ax.bar(cats, vals, color="#4a7bd0")
    ax.set_ylabel("Rainy days per year (fictional)")
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 10))
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Rainy days by city")
    path = "understand-chart.png"
    fig.savefig(os.path.join(OUT, path), bbox_inches="tight")
    plt.close(fig)
    top = cats[vals.index(max(vals))]
    above50 = sum(v > 50 for v in vals)
    items.append({
        "id": "understand-chart", "category": "understanding", "image": path, "scorer": "fields",
        "prompt": ("Look at the bar chart. Return a JSON object with: highest (the city with the most "
                   "rainy days), quito (Quito's value, read from the chart), above_50 (how many cities "
                   "have more than 50), lowest (the city with the fewest). " + JSON_ONLY),
        "truth": {"highest": top, "quito": vals[cats.index("Quito")], "above_50": above50,
                  "lowest": cats[vals.index(min(vals))]},
        "tolerance": {"quito": 2},
    })


def gen_count():
    rng = random.Random(302)
    w, h = 1000, 700
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    n_red, n_blue = 7, 11
    placed = []
    for col, n in (("red", n_red), ("blue", n_blue)):
        k = 0
        while k < n:
            x, y = rng.randint(40, w - 40), rng.randint(40, h - 40)
            if any(math.hypot(x - px, y - py) < 70 for px, py in placed):
                continue
            placed.append((x, y))
            d.ellipse([x - 22, y - 22, x + 22, y + 22], fill=COLORS[col])
            k += 1
    path = "understand-count.png"
    img.save(os.path.join(OUT, path))
    items.append({
        "id": "understand-count", "category": "understanding", "image": path, "scorer": "fields",
        "prompt": ("Count the dots. Return a JSON object with red (number of red dots), blue (number of "
                   "blue dots) and total. " + JSON_ONLY),
        "truth": {"red": n_red, "blue": n_blue, "total": n_red + n_blue},
    })


def gen_spatial():
    # a 3x3 grid of labelled objects
    grid = [["red circle", "blue square", "green triangle"],
            ["orange star", "purple circle", "black square"],
            ["blue triangle", "green circle", "red square"]]
    w, h = 900, 900
    img = Image.new("RGB", (w, h), (250, 250, 250))
    d = ImageDraw.Draw(img)
    for r, row in enumerate(grid):
        for c, obj in enumerate(row):
            col, shape = obj.split()
            cx, cy, s = 150 + c * 300, 150 + r * 300, 80
            if shape == "circle":
                d.ellipse([cx - s, cy - s, cx + s, cy + s], fill=COLORS[col])
            elif shape == "square":
                d.rectangle([cx - s, cy - s, cx + s, cy + s], fill=COLORS[col])
            elif shape == "triangle":
                d.polygon([(cx, cy - s), (cx - s, cy + s), (cx + s, cy + s)], fill=COLORS[col])
            else:
                d.polygon(star_points(cx, cy, s), fill=COLORS[col])
    path = "understand-spatial.png"
    img.save(os.path.join(OUT, path))
    items.append({
        "id": "understand-spatial", "category": "understanding", "image": path, "scorer": "fields",
        "prompt": ("The image shows a 3x3 grid of coloured shapes. Describe objects as '<color> <shape>' "
                   "in lowercase, e.g. 'red circle'. Return a JSON object with: top_right (the object in "
                   "the top-right cell), left_of_black_square (the object directly to the left of the "
                   "black square), below_green_triangle (the object directly below the green triangle), "
                   "circles (how many circles there are). " + JSON_ONLY),
        "truth": {"top_right": "green triangle", "left_of_black_square": "purple circle",
                  "below_green_triangle": "black square", "circles": 3},
    })


# --- CAD: dimensioned drawings ---------------------------------------------------------------------
def dim(ax, x1, y1, x2, y2, text, off, vertical=False):
    """Draw a dimension line with extension lines and a centred label."""
    if vertical:
        ax.plot([x1, x1 + off], [y1, y1], "k-", lw=0.6)
        ax.plot([x2, x2 + off], [y2, y2], "k-", lw=0.6)
        ax.annotate("", (x1 + off, y1), (x2 + off, y2), arrowprops=dict(arrowstyle="<->", lw=0.8))
        ax.text(x1 + off - 2, (y1 + y2) / 2, text, rotation=90, ha="right", va="center", fontsize=11)
    else:
        ax.plot([x1, x1], [y1, y1 + off], "k-", lw=0.6)
        ax.plot([x2, x2], [y2, y2 + off], "k-", lw=0.6)
        ax.annotate("", (x1, y1 + off), (x2, y2 + off), arrowprops=dict(arrowstyle="<->", lw=0.8))
        ax.text((x1 + x2) / 2, y1 + off + 1.5, text, ha="center", va="bottom", fontsize=11)


def gen_cad(i):
    W, H, D, n, x0, pitch, y0, part, mat = [
        (120, 60, 10, 4, 20, 26.67, 30, "PL-1042", "Aluminium 6061-T6"),
        (150, 80, 8, 5, 15, 30, 25, "PL-2077", "Steel S235"),
        (100, 50, 6, 3, 20, 30, 25, "PL-3310", "Stainless 304"),
    ][i]
    pitch = round(pitch, 2) if i else 80 / 3
    fig, ax = plt.subplots(figsize=(11, 7.5), dpi=110)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), W, H, fill=False, lw=1.6))
    for k in range(n):
        ax.add_patch(Circle((x0 + k * pitch, y0), D / 2, fill=False, lw=1.2))
        ax.plot([x0 + k * pitch - D * 0.8, x0 + k * pitch + D * 0.8], [y0, y0], "k-.", lw=0.4)
        ax.plot([x0 + k * pitch, x0 + k * pitch], [y0 - D * 0.8, y0 + D * 0.8], "k-.", lw=0.4)
    dim(ax, 0, H, W, H, f"{W}", 12)
    dim(ax, 0, 0, 0, H, f"{H}", -12, vertical=True)
    dim(ax, 0, 0, x0, 0, f"{x0}", -10)
    pitch_txt = f"{n - 1}x {pitch:g}" if i else f"{n - 1}x 26.67"
    dim(ax, x0, 0, x0 + pitch, 0, pitch_txt, -20)
    dim(ax, W, 0, W, y0, f"{y0}", 10, vertical=True)
    ax.annotate(f"{n}x Ø{D} THRU", (x0 + (n - 1) * pitch + D / 2 * 0.7, y0 + D / 2 * 0.7),
                (x0 + (n - 1) * pitch + 12, y0 + 20), fontsize=11, arrowprops=dict(arrowstyle="-", lw=0.6))
    # title block
    tb = ax.table(cellText=[["PART", part], ["MATERIAL", mat], ["THICKNESS", "5 mm"], ["UNITS", "mm"]],
                  colWidths=[0.12, 0.22], loc="lower right", cellLoc="left", bbox=[0.66, -0.32, 0.34, 0.2])
    tb.set_fontsize(10)
    ax.set_xlim(-25, W + 25)
    ax.set_ylim(-60, H + 25)
    path = f"cad-plate-{i + 1}.png"
    fig.savefig(os.path.join(OUT, path), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    items.append({
        "id": f"cad-plate-{i + 1}", "category": "cad", "image": path, "scorer": "fields",
        "prompt": ("This is a dimensioned engineering drawing of a plate, in millimetres. Return a JSON "
                   "object with: width, height, hole_count, hole_diameter, hole_pitch (centre-to-centre "
                   "distance between neighbouring holes), first_hole_x (distance from the left edge to "
                   "the first hole centre), hole_y (distance from the bottom edge to the hole centres), "
                   "part_number, material. Numbers as numbers. " + JSON_ONLY),
        "truth": {"width": W, "height": H, "hole_count": n, "hole_diameter": D,
                  "hole_pitch": round(pitch, 2), "first_hole_x": x0, "hole_y": y0,
                  "part_number": part, "material": mat},
        "tolerance": {"hole_pitch": 0.02},
    })


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for k in range(3):
        gen_invoice(k)
    for k in range(3):
        gen_shapes(k)
    gen_chart()
    gen_count()
    gen_spatial()
    for k in range(3):
        gen_cad(k)
    json.dump(items, open(os.path.join(HERE, "items.json"), "w"), indent=1, ensure_ascii=False)
    print(f"{len(items)} items -> {OUT}")
