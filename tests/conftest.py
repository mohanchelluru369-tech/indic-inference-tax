"""A fake OpenAI-compatible streaming server, so the client, the runner and the
report can be tested end to end without a model or a GPU.

It imitates the one property the project is about: non-English prompts cost
more tokens. Each UTF-8 byte of input counts as a "prompt token" and the reply
is a fixed script-appropriate string streamed one chunk per "token".
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

REPLIES = {
    "telugu": ["ఇది ", "ఒక ", "పరీక్ష ", "సమాధానం", "."],
    "devanagari": ["यह ", "एक ", "परीक्षण ", "उत्तर ", "है", "।"],
    "latin": ["This ", "is ", "a ", "test."],
}


def _script(text: str) -> str:
    for ch in text:
        if 0x0C00 <= ord(ch) <= 0x0C7F:
            return "telugu"
        if 0x0900 <= ord(ch) <= 0x097F:
            return "devanagari"
    return "latin"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # keep pytest output clean
        pass

    def do_GET(self):
        body = json.dumps({"data": [{"id": "fake-model"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append(req)
        user = req["messages"][-1]["content"]
        if "FAIL" in user:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b"boom")
            return
        pieces = REPLIES[_script(user)]
        if "ANSWER_B" in user:
            pieces = ["B"]
        pieces = pieces[: req["max_tokens"]]
        prompt_tokens = len(user.encode("utf-8"))

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()

        def send(obj):
            self.wfile.write(f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode())
            self.wfile.flush()

        # real engines open with a role-only chunk; it must not count as the first token
        send({"choices": [{"index": 0, "delta": {"role": "assistant", "content": None}}]})
        if "STREAM_ERROR" in user:  # llama.cpp reports e.g. context overflow inside a 200 stream
            send({"error": {"code": 500, "message": "context size exceeded", "type": "server_error"}})
            return
        if "DROP" in user:          # connection dies mid-answer: no finish_reason, no usage, no [DONE]
            send({"choices": [{"index": 0, "delta": {"content": "partial "}}]})
            return

        time.sleep(prompt_tokens * 0.00005)  # "prefill" grows with prompt tokens
        for p in pieces:
            time.sleep(0.004)
            send({"choices": [{"index": 0, "delta": {"content": p}, "finish_reason": None}]})
        cut = len(pieces) >= req["max_tokens"]
        send({"choices": [{"index": 0, "delta": {}, "finish_reason": "length" if cut else "stop"}]})
        send({"choices": [], "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": len(pieces)},
              "timings": {"prompt_per_second": 1000.0, "predicted_per_second": 250.0}})
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


@pytest.fixture()
def fake_server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    srv.requests = []
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield srv, f"http://127.0.0.1:{srv.server_address[1]}/v1"
    finally:
        srv.shutdown()
        srv.server_close()
