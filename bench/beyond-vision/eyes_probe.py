#!/usr/bin/env python3
"""Two models, one job: LFM3-VL looks, LFM3-1.7B decides. Same scenes and scoring as the glance probe.

Needs the driver on 127.0.0.1:8080 and the eye, with its projector, on 127.0.0.1:8081.
Writes results/eyes.json. The driver never sees the picture.
"""
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "visual-control"))
import run_control as rc  # noqa: E402

DRIVER = "http://127.0.0.1:8080/v1/chat/completions"
EYE = "http://127.0.0.1:8081/v1/chat/completions"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 12
ASK = ("You are the eyes for another agent that can't see this picture. It was asked:\n\n{prompt}\n\n"
       "Report only what it needs: exact readings, colours, labels and positions, with x and y from 0 to 1000 "
       "for anything it may have to point at. Don't take any action.")
items = []
for name, gen in rc.SCENES.items():
    ok = 0
    for i in range(N):
        img, prompt, tools, score, truth = gen(random.Random(f"{name}-{i}"), i)
        rc.URL = EYE
        seen = rc.chat([{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{rc.png_b64(img)}"}},
            {"type": "text", "text": ASK.format(prompt=prompt)}]}], [], True, 3000)
        rc.URL = DRIVER
        t = time.time()
        r = rc.chat([{"role": "user", "content": f"{prompt}\n\nYou can't see the picture. Your vision tool reports:\n\n"
                                                 f"{seen['text']}"}], tools, True)
        good = bool(not r.get("error") and score(r))
        ok += good
        items.append({"scene": name, "i": i, "ok": good, "report": seen["text"][:400], "calls": r["calls"],
                      "eye_secs": round(seen["secs"], 1), "driver_secs": round(time.time() - t, 1)})
        print(name, i, "ok" if good else "MISS", items[-1]["eye_secs"], items[-1]["driver_secs"], flush=True)
    print(f"== {name}: {ok}/{N}", flush=True)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results/eyes.json"), "w") as f:
        json.dump(items, f, indent=1, default=str)
