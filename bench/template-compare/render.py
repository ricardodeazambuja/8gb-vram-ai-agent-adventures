"""POST every captured pi request to llama-server's /apply-template; write the rendered prompts as JSON.
Usage: render.py PORT OUT.json REQ.json..."""
import json, sys, urllib.request
port, out, reqs = sys.argv[1], sys.argv[2], sys.argv[3:]
res = {}
def post(body):
    rq = urllib.request.Request(f"http://127.0.0.1:{port}/apply-template", json.dumps(body).encode(),
                                {"Content-Type": "application/json"})
    try:
        return json.load(urllib.request.urlopen(rq, timeout=60)).get("prompt")
    except urllib.error.HTTPError as e:
        return "HTTP %d: %s" % (e.code, e.read().decode()[:300])
for p in reqs:
    r = json.load(open(p))
    body = {k: r[k] for k in ("messages", "tools", "reasoning_effort") if k in r}
    res[p] = post(body)
    for eff in ("high", "low", "xhigh"):
        res[f"{p}#effort={eff}"] = post({**body, "reasoning_effort": eff})
json.dump(res, open(out, "w"), indent=0)
print(len(res), "rendered;", sum(1 for v in res.values() if v and v.startswith("HTTP")), "errors")
