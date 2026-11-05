import json
import time
import urllib.request

body = {"prompt": "<|im_start|>user\nWrite a complete, well-documented Python implementation of a B-tree with insert, delete, search, range queries and pretty printing, plus extensive pytest tests.<|im_end|>\n<|im_start|>assistant\n", "n_predict": 2500, "temperature": 0, "ignore_eos": True, "stream": True, "cache_prompt": False}
r = urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8090/completion", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=3600)
n, t0, marks = 0, None, []
for line in r:
    if not line.startswith(b"data: "): continue
    d = json.loads(line[6:]); n += 1
    if t0 is None: t0 = time.time()
    if n % 500 == 0: marks.append(time.time())
    if d.get("stop"): tm = d.get("timings", {}); break
prev, out = t0, []
for m in marks: out.append(f"{500/(m-prev):5.1f}"); prev = m
print("   t/s per 500-token window:", " ".join(out), f"| overall {tm.get('predicted_per_second', 0):.2f}")
