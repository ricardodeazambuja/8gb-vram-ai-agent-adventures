#!/usr/bin/env python3
"""A pass-through HTTP proxy that saves every request and response, for looking at what pi sends a model server
and what comes back.

    UPSTREAM=http://127.0.0.1:8081 HOST=172.30.0.1 PORT=8080 LOG_DIR=/path python3 bench/gemma12-probe/logproxy.py

Each POST is saved as NNNN-req.json (the body pi sent) and NNNN-resp.txt (the raw bytes the server sent back,
SSE included). Responses are streamed through unchanged, so the client sees exactly what it would without the proxy.
"""
import http.client
import itertools
import os
import sys
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UP = urllib.parse.urlparse(os.environ.get("UPSTREAM", "http://127.0.0.1:8081"))
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8080"))
LOG_DIR = os.environ["LOG_DIR"]
_count = itertools.count(1)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"   # the response ends when the connection closes, so streaming needs no framing

    def log_message(self, *a):
        pass

    def _forward(self, method):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0) or 0)) if method == "POST" else None
        n = next(_count) if method == "POST" else None
        if n:
            with open(os.path.join(LOG_DIR, f"{n:04d}-req.json"), "wb") as f:
                f.write(body)
        conn = http.client.HTTPConnection(UP.hostname, UP.port, timeout=1800)
        headers = {k: v for k, v in self.headers.items() if k.lower() not in ("host", "connection")}
        try:
            conn.request(method, self.path, body=body, headers=headers)
            r = conn.getresponse()
        except OSError as e:
            self.send_error(502, str(e))
            return
        self.send_response(r.status)
        for k, v in r.getheaders():
            if k.lower() not in ("transfer-encoding", "connection", "content-length"):
                self.send_header(k, v)
        self.send_header("Connection", "close")
        self.end_headers()
        path = os.path.join(LOG_DIR, f"{n:04d}-resp.txt") if n else os.devnull
        try:
            with open(path, "wb") as log:
                while True:
                    chunk = r.read1(65536)
                    if not chunk:
                        break
                    log.write(chunk)
                    log.flush()
                    self.wfile.write(chunk)
                    self.wfile.flush()
        finally:
            conn.close()

    def do_GET(self):
        self._forward("GET")

    def do_POST(self):
        self._forward("POST")


if __name__ == "__main__":
    os.makedirs(LOG_DIR, exist_ok=True)
    print(f"[logproxy] http://{HOST}:{PORT} -> {UP.geturl()}, logging to {LOG_DIR}", file=sys.stderr, flush=True)
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
