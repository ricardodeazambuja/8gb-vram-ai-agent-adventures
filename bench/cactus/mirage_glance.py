"""Mirage on the visual-control glance scenes. Runs only inside sandbox.sh (no network).

    bash bench/cactus/sandbox.sh -I /bench/cactus/mirage_glance.py [N]

Mirage can't call tools, so it reports what it sees (multiple-choice form fields, the text it reads,
the objects it finds) and a few lines here apply each scene's rule and turn that into the call the
probe expects. The scenes, their seeds and the scorers are the probe's own (run_control.py), so the
right answers are the same 150 that LFM3-VL-2B was scored on. The split (the model looks, code
decides) is not the probe's test, where the model applies the rule itself.
"""
import json
import os
import sys
import time

sys.path.insert(0, "/cx")
sys.path.insert(0, "/bench/visual-control")
import cactus
import run_control as rc

N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
OUT = os.environ.get("ADV_RES", "/out")
m = cactus.Mirage()


def call(name, **args):
    return {"name": name, "args": args}


def pick_value(field):
    return field.get("value")


def ask_tanks(img, truth):
    form = {f"tank_{t}_below_30_percent_full": {"enum": ["yes", "no"]} for t in "ABC"}
    r = m.extract(img, form)["form"]
    calls = [call("refill", tank=t) for t in "ABC" if pick_value(r[f"tank_{t}_below_30_percent_full"]) == "yes"]
    return calls, r


def ask_signal(img, truth):
    form = {"traffic_light_color": {"enum": ["red", "yellow", "green"]},
            "person_on_crosswalk": {"enum": ["yes", "no"]}}
    r = m.extract(img, form)["form"]
    seen = [str(o.get("name", "")).lower() for o in m.describe(img, sections=["objects"]).get("objects", [])]
    r["objects_seen"] = seen
    light = pick_value(r["traffic_light_color"])
    person = pick_value(r["person_on_crosswalk"]) == "yes" or any(
        w in n for n in seen for w in ("person", "pedestrian", "man", "woman", "child", "people", "figure"))
    name = "stop" if light == "red" or person else "slow_down" if light == "yellow" else "go" if light == "green" else None
    return ([call(name)] if name else []), r


TICKS = [str(v) for v in range(0, 101, 10)]


def ask_gauge(img, truth):
    r = m.extract(img, {"value_the_red_needle_points_at": {"enum": TICKS}})["form"]
    v = pick_value(r["value_the_red_needle_points_at"])
    if v is None:
        return [], r
    v = int(v)
    return [call("open_valve") if v > 70 else call("close_valve") if v < 20 else call("report", value=v)], r


def ask_click(img, truth):
    r = m.describe(img, sections=["text"])
    target = truth["target"].lower()
    hits = [t for t in r.get("text", []) if t.get("string", "").strip().lower() == target]
    if not hits:
        return [], r.get("text")
    b = max(hits, key=lambda t: t.get("score", 0))["bbox"]  # Mirage's boxes are already 0-1000
    return [call("click", x=(b[0] + b[2]) / 2, y=(b[1] + b[3]) / 2)], r.get("text")


def ask_pick(img, truth):
    r = m.describe(img, sections=["objects"], fields=["bbox"])
    objs = [o for o in r.get("objects", []) if "bbox" in o
            and (o["bbox"][2] - o["bbox"][0]) * (o["bbox"][3] - o["bbox"][1]) < 0.25 * 1000 * 1000]  # not the table
    slim = [{"name": o.get("name"), "description": o.get("description"), "bbox": o["bbox"], "score": o.get("score")}
            for o in objs]
    ask = truth["ask"].split(" ", 1)[1]  # "green circle" or "leftmost object"
    words = ask.split()
    if words[-1] == "object":
        if not objs:
            return [], slim
        key = {"leftmost": lambda o: o["bbox"][0] + o["bbox"][2], "rightmost": lambda o: -(o["bbox"][0] + o["bbox"][2]),
               "topmost": lambda o: o["bbox"][1] + o["bbox"][3], "bottommost": lambda o: -(o["bbox"][1] + o["bbox"][3])}[words[0]]
        best = min(objs, key=key)
    else:
        color, shape = words

        def text(o):
            return (str(o.get("name", "")) + " " + json.dumps(o.get("description", {}))).lower()
        match = [o for o in objs if color in text(o) and shape in text(o)]
        if not match:
            return [], slim
        best = max(match, key=lambda o: o.get("score", 0))
    b = best["bbox"]
    return [call("pick", x=(b[0] + b[2]) / 2, y=(b[1] + b[3]) / 2)], slim


out = {"test": "glance", "version": 2, "model": "mirage.cact (Cactus-Compute/mirage@76c1e9e)", "n": N, "items": []}
total = 0
for name, gen in rc.SCENES.items():
    ok = 0
    for i in range(N):
        rng = rc.random.Random(f"{name}-{i}")
        img, _prompt, _tools, score, truth = gen(rng, i)
        t = time.perf_counter()
        if name == "click":
            calls, raw = ask_click(img, truth)
        elif name == "pick":
            calls, raw = ask_pick(img, truth)
        else:
            calls, raw = {"tanks": ask_tanks, "signal": ask_signal, "gauge": ask_gauge}[name](img, truth)
        secs = time.perf_counter() - t
        good = bool(score({"calls": calls}))
        ok += good
        out["items"].append({"scene": name, "i": i, "ok": good, "calls": calls, "secs": round(secs, 2),
                             "mirage": json.loads(json.dumps(raw, default=str)),
                             "truth": json.loads(json.dumps(truth, default=str))})
        print(f"  {name:6s} {i:2d} {'ok ' if good else 'MISS'} {secs:4.1f}s {[(c['name'], c['args']) for c in calls]}",
              flush=True)
    print(f"== {name}: {ok}/{N}", flush=True)
    total += ok
print(f"== all: {total}/{N * len(rc.SCENES)}")
with open(os.path.join(OUT, f"mirage-glance-{N}.json"), "w") as f:
    json.dump(out, f, indent=1)
