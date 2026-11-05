import json
import os
import sys
import urllib.request

label, vram, runs = sys.argv[1], sys.argv[2], int(sys.argv[3])
def post(body):
    r = urllib.request.Request("http://127.0.0.1:8090/completion", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=3600))
# The long prompt is src/llama-moe-cache.cpp of our fork (as at moe-offload 3a72c57be), first 22,000 chars.
with open(os.path.join(os.environ.get('ADV_LLAMA', os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../llama.cpp')), 'src/llama-moe-cache.cpp')) as f:
    src = f.read()[:22000]
for k in range(runs):
    long = post({"prompt": f"<|im_start|>user\nRun {k}. Review this C++ file and list its bugs:\n{src}<|im_end|>\n<|im_start|>assistant\n", "n_predict": 1, "cache_prompt": False})
    gen = post({"prompt": "<|im_start|>user\nWrite a Python module that implements an LRU cache with TTL expiry, thread safety, and statistics. Include docstrings and unit tests.<|im_end|>\n<|im_start|>assistant\n", "n_predict": 400, "temperature": 0, "ignore_eos": True, "cache_prompt": False})
    t1, t2 = long["timings"], gen["timings"]
    print(f"{label:<28} vram {vram:>9}  prefill {t1['prompt_n']:>5} tok {t1['prompt_per_second']:7.1f} t/s   decode {t2['predicted_n']} tok {t2['predicted_per_second']:6.2f} t/s", flush=True)
