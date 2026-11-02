"""Fill most of a running server's context with this repo's docs and ask one question about them.

    python3 bench/bonsai-ctx/longprompt.py 28000     # about 28,000 prompt tokens

Run from the repo root, against a llama-server on 127.0.0.1:8080. The question has one right
answer (the Bonsai 2 page calls big web pages its "kryptonite"), so a model that lost the start
of the prompt shows it. Prints the prompt size, both speeds and the end of the answer.
"""
import glob
import json
import sys
import urllib.request

URL = "http://127.0.0.1:8080"


def post(path, body, timeout=60):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def main():
    target = int(sys.argv[1])
    text = ""
    for f in sorted(glob.glob("docs/**/*.md", recursive=True)):
        with open(f) as fh:
            text += fh.read()
    n = len(post("/tokenize", {"content": text})["tokens"])
    text = text[: int(len(text) * target / n)]
    q = text + ("\n\nQuestion: which model page above mentions 'kryptonite', and what is it? "
                "Answer in one sentence.")
    d = post("/v1/chat/completions", {"messages": [{"role": "user", "content": q}], "max_tokens": 3000,
                                      "chat_template_kwargs": {"enable_thinking": False}}, timeout=1800)
    t = d.get("timings", {})
    print(f"prompt tokens {d['usage']['prompt_tokens']} | prefill {t.get('prompt_per_second', 0):.0f} t/s"
          f" | gen {t.get('predicted_per_second', 0):.1f} t/s")
    answer = (d["choices"][0]["message"].get("content") or "").replace("\n", " ")
    print("answer:", answer[-300:])


if __name__ == "__main__":
    main()
