#!/usr/bin/env python3
"""Run the vision probe: every image-capable model answers every item in bench/vision/items.json.

    python3 bench/vision/run_vision.py                     # all models
    python3 bench/vision/run_vision.py qwen3vl gemma26b    # some of them
    python3 bench/vision/run_vision.py --score-only        # re-score saved answers

No agent and no tools: the image and the question go straight to llama-server's chat API, and the
JSON answer is scored against the generator's ground truth. Answers are saved per model/item under
bench/vision/results/raw/ (resumable: an item with a saved answer is not asked again), box overlays
under results/overlays/, the table in results/summary.md.
"""
import base64
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request

from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
RES = os.path.join(HERE, "results")
URL = "http://127.0.0.1:8080"
ITEMS = json.load(open(os.path.join(HERE, "items.json"), encoding="utf-8"))
MODELS = {  # key -> (launch env, launch script) ; every one with its vision projector
    "qwen3vl": ({"QMODEL": "qwen3vl"}, "scripts/run-qwen.sh"),
    "gemma26b": ({"BACKEND": "cuda"}, "scripts/run-server.sh"),
    "qwen36": ({"QMODEL": "qwen36"}, "scripts/run-qwen.sh"),
    "distill": ({"QMODEL": "distill"}, "scripts/run-qwen.sh"),
    "bonsai2": ({"QMODEL": "bonsai2"}, "scripts/run-qwen.sh"),
    "bolt": ({"QMODEL": "bolt"}, "scripts/run-qwen.sh"),
}
MAX_TOKENS = 12000
TIMEOUT = 900


# --- server ------------------------------------------------------------------------------------------
def healthy():
    try:
        return urllib.request.urlopen(URL + "/health", timeout=2).status == 200
    except OSError:
        return False


def start(key):
    if healthy():
        sys.exit("ERROR: something is already serving on port 8080; stop it first (scripts/stop-server.sh)")
    env, script = MODELS[key]
    log = open(os.path.join(RES, "server.log"), "a")
    proc = subprocess.Popen(["bash", os.path.join(REPO, script), "--image"], cwd=REPO,
                            env={**os.environ, **env}, stdout=log, stderr=log, start_new_session=True)
    for _ in range(300):
        if healthy():
            return proc
        if proc.poll() is not None:
            sys.exit(f"ERROR: {key} server exited; see {RES}/server.log")
        time.sleep(2)
    stop(proc)
    sys.exit(f"ERROR: {key} did not come up")


def stop(proc):
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=30)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    time.sleep(3)


def ask(item):
    b64 = base64.b64encode(open(os.path.join(HERE, "items", item["image"]), "rb").read()).decode()
    body = {"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        {"type": "text", "text": item["prompt"]}]}], "max_tokens": MAX_TOKENS}
    req = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        d = json.load(r)
    d["_wall_s"] = round(time.time() - t, 1)
    return d


# --- scoring -------------------------------------------------------------------------------------------
def parse_json(text):
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1)
    starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
    if not starts:
        return None
    s = min(starts)
    e = max(text.rfind("}"), text.rfind("]"))
    try:
        return json.loads(text[s:e + 1])
    except json.JSONDecodeError:
        return None


def norm(v):
    return " ".join(str(v).lower().replace("ø", "").split())


def same(got, want, tol=0.0):
    if isinstance(want, (int, float)) and not isinstance(want, bool):
        try:
            g = float(str(got).replace("Ø", "").replace("mm", "").replace(",", "").strip())
        except ValueError:
            return False
        return abs(g - want) <= max(tol, 1e-6)
    return norm(got) == norm(want)


def score_fields(ans, item):
    tol = item.get("tolerance", {})
    if not isinstance(ans, dict):
        return 0.0, {}
    ok = {k: same(ans.get(k), v, tol.get(k, 0)) for k, v in item["truth"].items()}
    return sum(ok.values()) / len(ok), ok


def score_invoice(ans, item):
    t = item["truth"]
    if not isinstance(ans, dict):
        return 0.0, {}
    ok = {"invoice_number": same(ans.get("invoice_number"), t["invoice_number"]),
          "date": same(ans.get("date"), t["date"]),
          "total": same(ans.get("total"), t["total"], 0.01)}
    got = ans.get("items") if isinstance(ans.get("items"), list) else []
    for i, line in enumerate(t["items"]):
        g = got[i] if i < len(got) and isinstance(got[i], dict) else {}
        ok[f"item{i}.description"] = same(g.get("description"), line["description"])
        ok[f"item{i}.qty"] = same(g.get("qty"), line["qty"])
        ok[f"item{i}.amount"] = same(g.get("amount"), line["amount"], 0.01)
    return sum(ok.values()) / len(ok), ok


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def convert(box, conv, w, h):
    x1, y1, x2, y2 = box
    if conv == "pixel":
        return [x1, y1, x2, y2]
    if conv == "norm1000_xyxy":
        return [x1 * w / 1000, y1 * h / 1000, x2 * w / 1000, y2 * h / 1000]
    if conv == "norm1000_yxyx":
        return [y1 * w / 1000, x1 * h / 1000, y2 * w / 1000, x2 * h / 1000]
    return [x1 * w, y1 * h, x2 * w, y2 * h]  # norm1


def score_bbox(ans, item):
    """Share of shapes found: matched box with IoU >= 0.5 and the right shape and colour.

    Scored in pixels as asked, and also in the best of three other common coordinate conventions
    (0-1000 xyxy, 0-1000 yxyx, 0-1), to separate "can't localize" from "used another format"."""
    w, h = item["size"]
    truth = item["truth"]
    preds = []
    for p in ans if isinstance(ans, list) else []:
        try:
            # Accept the models' native key names (Gemma: box_2d, Qwen: bbox_2d) as well as the
            # requested "bbox", and a shape named inside a longer label ("blue circle").
            key = next(k for k in ("bbox", "bbox_2d", "box_2d", "box") if k in p)
            b = [float(v) for v in p[key]][:4]
            label = norm(p.get("label"))
            label = next((sh for sh in ("circle", "square", "triangle", "star") if sh in label.split()), label)
            if len(b) == 4:
                preds.append((label, norm(p.get("color")), b))
        except (StopIteration, TypeError, ValueError, AttributeError):
            continue
    out = {}
    for conv in ("pixel", "norm1000_xyxy", "norm1000_yxyx", "norm1"):
        if not preds:
            out[conv] = {"score": 0.0, "mean_iou": 0.0, "boxes": []}
            continue
        boxes = [convert(b, conv, w, h) for _, _, b in preds]
        cost = [[-iou(t["bbox"], b) for b in boxes] for t in truth]
        rows, cols = linear_sum_assignment(cost)
        hits, ious = 0, []
        for r, c in zip(rows, cols):
            v = -cost[r][c]
            ious.append(v)
            if v >= 0.5 and preds[c][0] == truth[r]["label"] and preds[c][1] == truth[r]["color"]:
                hits += 1
        out[conv] = {"score": hits / len(truth), "mean_iou": sum(ious) / len(truth), "boxes": boxes}
    best = max(out, key=lambda k: (out[k]["mean_iou"], out[k]["score"]))
    return out["pixel"]["score"], {"pixel": {k: v for k, v in out["pixel"].items() if k != "boxes"},
                                   "best_convention": best,
                                   "best": {k: v for k, v in out[best].items() if k != "boxes"},
                                   "_boxes": out[best]["boxes"], "_count": len(preds)}


SCORERS = {"fields": score_fields, "invoice": score_invoice, "bbox": score_bbox}


def overlay(item, detail, path):
    img = Image.open(os.path.join(HERE, "items", item["image"])).convert("RGB")
    d = ImageDraw.Draw(img)
    for t in item["truth"]:
        d.rectangle(t["bbox"], outline=(0, 200, 0), width=3)
    for b in detail.get("_boxes", []):
        d.rectangle([min(b[0], b[2]), min(b[1], b[3]), max(b[0], b[2]), max(b[1], b[3])],
                    outline=(255, 0, 255), width=3)
    img.save(path)


def score_all():
    rows = {}
    os.makedirs(os.path.join(RES, "overlays"), exist_ok=True)
    for key in MODELS:
        rdir = os.path.join(RES, "raw", key)
        if not os.path.isdir(rdir):
            continue
        res = []
        for item in ITEMS:
            p = os.path.join(rdir, item["id"] + ".json")
            if not os.path.exists(p):
                continue
            d = json.load(open(p))
            msg = d["choices"][0]["message"] if "choices" in d else {}
            ans = parse_json(msg.get("content", ""))
            s, detail = SCORERS[item["scorer"]](ans, item)
            if item["scorer"] == "bbox":
                overlay(item, detail, os.path.join(RES, "overlays", f"{key}-{item['id']}.png"))
                detail = {k: v for k, v in detail.items() if not k.startswith("_")} | {"boxes": detail["_count"]}
            res.append({"id": item["id"], "category": item["category"], "score": s, "parsed": ans is not None,
                        "detail": detail, "wall_s": d.get("_wall_s"),
                        "tokens": (d.get("usage") or {}).get("completion_tokens", 0)})
        json.dump(res, open(os.path.join(RES, f"scores-{key}.json"), "w"), indent=1, default=str)
        rows[key] = res
    cats = ["ocr", "bbox", "understanding", "cad"]
    lines = ["| Model | OCR | Boxes (pixels, as asked) | Boxes (any format) | Understanding | CAD | Overall (any format) | Median time | Median tokens |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for key, res in rows.items():
        def mean(xs):
            return sum(xs) / len(xs) if xs else float("nan")
        by = {c: [r["score"] for r in res if r["category"] == c] for c in cats}
        best = [r["detail"]["best"]["score"] for r in res if r["category"] == "bbox"]
        # Overall = what the model can see: boxes count in whichever coordinate format it used.
        seen = [r["detail"]["best"]["score"] if r["category"] == "bbox" else r["score"] for r in res]
        conv = sorted({r["detail"]["best_convention"] for r in res if r["category"] == "bbox"})
        times = sorted(r["wall_s"] or 0 for r in res)
        toks = sorted(r["tokens"] for r in res)
        lines.append(f"| {key} | {mean(by['ocr']):.0%} | {mean(by['bbox']):.0%} | {mean(best):.0%} ({', '.join(conv)}) | "
                     f"{mean(by['understanding']):.0%} | {mean(by['cad']):.0%} | **{mean(seen):.0%}** | "
                     f"{times[len(times) // 2]:.0f} s | {toks[len(toks) // 2]:,} |")
    open(os.path.join(RES, "summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    args = sys.argv[1:]
    if args == ["--score-only"]:
        return score_all()
    os.makedirs(RES, exist_ok=True)
    for key in args or list(MODELS):
        rdir = os.path.join(RES, "raw", key)
        os.makedirs(rdir, exist_ok=True)
        todo = [it for it in ITEMS if not os.path.exists(os.path.join(rdir, it["id"] + ".json"))]
        if not todo:
            print(f"skip {key} (done)")
            continue
        print(f"[{time.strftime('%H:%M:%S')}] starting {key}", flush=True)
        proc = start(key)
        try:
            for item in todo:
                try:
                    d = ask(item)
                except Exception as e:  # noqa: BLE001 — record the failure as the answer
                    d = {"error": str(e), "_wall_s": TIMEOUT}
                json.dump(d, open(os.path.join(rdir, item["id"] + ".json"), "w"), indent=1)
                c = (d.get("choices") or [{}])[0].get("message", {}).get("content", "")
                print(f"[{time.strftime('%H:%M:%S')}]   {key} {item['id']}: {d.get('_wall_s')} s, "
                      f"{(d.get('usage') or {}).get('completion_tokens', 0)} tok, "
                      f"{'json' if parse_json(c) is not None else 'NO JSON'}", flush=True)
        finally:
            stop(proc)
    score_all()


if __name__ == "__main__":
    main()
