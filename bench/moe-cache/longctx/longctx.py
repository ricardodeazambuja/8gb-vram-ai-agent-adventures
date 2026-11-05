#!/usr/bin/env python3
"""Long-context decode check against a running llama-server on PORT (default 8090).
Builds a ~30K-token prompt from the adventures docs (same text as bench/moe-cache/fitcheck.sh),
then sends it twice with n_predict 600 (greedy, ignore_eos), streaming, and prints decode t/s per
100-token window. Request 1 starts with a cold expert cache after a long prefill; request 2 reuses the
prompt cache, so its decode runs with the expert cache warmed by request 1.
Also writes the templated prompt to OUT_PROMPT (for llama-completion -f) and logs CPU temperature."""
import glob
import json
import os
import subprocess
import time
import urllib.request

PORT = os.environ.get("PORT", "8090")
U = f"http://127.0.0.1:{PORT}"
DOCS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "docs", "*.md")
OUT_PROMPT = os.environ.get("OUT_PROMPT")

def post(path, body, stream=False):
    r = urllib.request.Request(U + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(r, timeout=7200) if stream else json.load(urllib.request.urlopen(r, timeout=7200))

def temp():
    try:
        out = subprocess.run(["sensors"], capture_output=True, text=True, check=False).stdout
        return next((l.split()[3] for l in out.splitlines() if l.startswith("Package id 0")), "?")
    except (OSError, IndexError):
        return "?"

txt = ""
for f in sorted(glob.glob(DOCS)):
    with open(f) as fh:
        txt += fh.read()
txt = txt[:100000]
prompt = post("/apply-template", {"messages": [{"role": "user", "content": txt + "\n\nSummarize the above in 3 sentences."}]})["prompt"]
if OUT_PROMPT:
    with open(OUT_PROMPT, "w") as fh:
        fh.write(prompt)

for k in (1, 2):
    t_before = temp()
    body = {"prompt": prompt, "n_predict": 600, "temperature": 0, "ignore_eos": True, "stream": True, "cache_prompt": True}
    n, t0, marks, tm = 0, None, [], {}
    for line in post("/completion", body, stream=True):
        if not line.startswith(b"data: "):
            continue
        d = json.loads(line[6:])
        if d.get("stop"):
            tm = d.get("timings", {})
            break
        n += 1
        if t0 is None:
            t0 = time.time()
        if n % 100 == 0:
            marks.append(time.time())
    prev, w = t0, []
    for m in marks:
        w.append(f"{100/(m-prev):5.1f}")
        prev = m
    print(f"req{k}: prompt {tm.get('prompt_n')} tok ({tm.get('cache_n')} cached) {tm.get('prompt_per_second', 0):.1f} t/s | "
          f"decode {tm.get('predicted_n')} tok {tm.get('predicted_per_second', 0):.2f} t/s | per 100: {' '.join(w)} | CPU {t_before} -> {temp()}", flush=True)
