#!/usr/bin/env python3
"""
Minimal HTTP server that returns deterministic sham CLAP-like scores (same hash as benchmark.py).

Usage (from repo root):

  ./.venv/bin/python sham_clap_server.py --port 8765

Then:

  ./.venv/bin/python benchmark.py --endpoint http://127.0.0.1:8765/v1/similarity

Expects POST JSON with \"text\" and optional \"audio_filename\" (basename of the clip).

If your client sends audio_base64, the stem is inferred from audio_filename — no audio decoding needed for sham scores.
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse

from benchmark import emotion_to_prompt, sham_similarity, stem_from_annotation_filename


class Handler(BaseHTTPRequestHandler):
    server_version = "ShamCLAP/0.1"

    def log_message(self, format: str, *args: object) -> None:
        """Less noisy stderr."""
        return

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        # Accept POST on any path (the client URL may include /v1/similarity, etc.)

        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"

        try:
            body = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            self._send_json(400, {"error": f"invalid json: {exc}"})
            return

        text = body.get("text")
        audio_fn = body.get("audio_filename") or "unknown.mp3"

        stem = stem_from_annotation_filename(str(audio_fn))

        if isinstance(text, str):
            prompt = text
        else:
            queried = body.get("queried_emotion")
            if not isinstance(queried, str):
                self._send_json(
                    400,
                    {"error": "Provide either {\"text\": \"...\"} or {\"queried_emotion\": \"...\"}"},
                )
                return
            prompt = emotion_to_prompt(queried)

        sim = sham_similarity(stem, prompt)
        payload = {"similarity": sim, "path": parsed.path, "stem": stem}
        self._send_json(200, payload)

    def _send_json(self, status: int, payload: dict) -> None:
        buf = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(buf)))
        self.end_headers()
        self.wfile.write(buf)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()
    httpd = HTTPServer((args.host, args.port), Handler)
    print(f"Sham CLAP listening on http://{args.host}:{args.port} (POST JSON; any path)")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
