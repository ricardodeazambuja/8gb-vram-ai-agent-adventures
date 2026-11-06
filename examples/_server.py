"""Shared by the examples: start a model with scripts/llm if it isn't serving, call its API, stop it after.

Standard library only. Every example takes --keep (leave the server running for the next call) and
--port. A server that was already running is never stopped, only one an example started itself.
"""
import atexit
import base64
import json
import mimetypes
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LLM = REPO / "scripts" / "llm"
PORT = 8080


def add_server_args(ap):
    ap.add_argument("--keep", action="store_true", help="leave the server running afterwards")
    ap.add_argument("--port", type=int, default=8080)


def ensure(name, args, extra=()):
    """Make sure `name` serves on args.port; stop it at exit if we started it (unless --keep)."""
    global PORT
    PORT = args.port
    env_port = {"PORT": str(PORT)}
    r = subprocess.run([str(LLM), name, "serve", *extra], capture_output=True, text=True, check=False,
                       env={**os.environ, **env_port})
    if r.returncode != 0:
        sys.exit((r.stderr or r.stdout).strip())
    started = "already running" not in r.stdout
    if started:
        print(f"[started {name}; " + ("left running (--keep): scripts/llm stop]" if args.keep else
                                      "it stops when this script ends; --keep leaves it running]"), file=sys.stderr)
        if not args.keep:
            atexit.register(lambda: subprocess.run([str(LLM), "stop"], capture_output=True, check=False,
                                                   env={**os.environ, **env_port}))


def post(path, payload, timeout=600):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", json.dumps(payload).encode(),
                                 {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as e:
        sys.exit(f"server error {e.code}: {e.read().decode(errors='replace')[:500]}")


def data_url(path):
    """A picture as a data: URL for image_url content."""
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(Path(path).read_bytes()).decode()}"


def b64(path):
    return base64.b64encode(Path(path).read_bytes()).decode()
