"""First look at Mirage: describe pictures, fill one form. Runs only inside sandbox.sh."""
import json
import sys
import time

import cactus


def slim(x):
    if isinstance(x, dict):
        return {k: slim(v) for k, v in x.items() if k != "mask"}
    if isinstance(x, list):
        return [slim(v) for v in x]
    return x


PICS = sys.argv[1:] or ["/bench/visual-control/examples/gauge.png", "/images/xmas-star-2.png"]
m = cactus.Mirage()
for p in PICS:
    t = time.perf_counter()
    r = slim(m.describe(p))
    dt = time.perf_counter() - t
    objs = [(o.get("name"), o.get("score"), o.get("position"), o.get("description")) for o in r.get("objects", [])]
    rest = {k: v for k, v in r.items() if k != "objects"}
    print(f"\n== {p} ({dt:.2f} s)\n{json.dumps(rest)[:1500]}")
    for o in objs:
        print("  ", o)
