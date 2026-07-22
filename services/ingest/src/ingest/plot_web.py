"""Lightweight debug web page: a live CSI amplitude waterfall/heatmap.

No web framework — a stdlib `http.server` serves one static page and one
JSON endpoint that the page polls, so the only runtime cost when `--plot
web` is off is that this module is never imported.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ingest.buffer import RingBuffer

_STATIC_DIR = Path(__file__).resolve().parent / "static"


def _snapshot_to_json(buffer: RingBuffer) -> bytes:
    frames = buffer.snapshot()
    amplitude_matrix = [f["amplitude"] for f in frames]
    sequence_numbers = [f["sequence_number"] for f in frames]
    payload = {
        "capacity": buffer.capacity,
        "frame_count": len(frames),
        "subcarrier_count": len(amplitude_matrix[-1]) if amplitude_matrix else 0,
        "amplitude_matrix": amplitude_matrix,
        "sequence_numbers": sequence_numbers,
    }
    return json.dumps(payload).encode("utf-8")


def _make_handler(buffer: RingBuffer) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            pass  # quiet; the receiver loop already logs

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                self._serve_file("index.html", "text/html; charset=utf-8")
            elif self.path == "/data.json":
                self._serve_bytes(_snapshot_to_json(buffer), "application/json", no_store=True)
            else:
                self.send_error(404)

        def _serve_file(self, name: str, content_type: str) -> None:
            self._serve_bytes((_STATIC_DIR / name).read_bytes(), content_type)

        def _serve_bytes(self, body: bytes, content_type: str, *, no_store: bool = False) -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            if no_store:
                self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

    return Handler


def start_web_plot_server(
    buffer: RingBuffer, *, host: str, port: int
) -> tuple[ThreadingHTTPServer, threading.Thread]:
    """Start the debug web plot server in a background thread.

    Returns (server, thread); call `server.shutdown()` then `thread.join()`
    to stop it.
    """
    server = ThreadingHTTPServer((host, port), _make_handler(buffer))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread
