"""
Lightweight HTTP control API for runtime market management.

Runs in a daemon thread. Endpoints:

    GET  /status              → system overview (markets, alerts, uptime)
    POST /markets/add         → add a slug (body: {"slug": "some-slug"})
    POST /markets/remove      → remove a slug (body: {"slug": "some-slug"})
    GET  /markets             → list active market slugs

Usage from host:
    curl localhost:8585/status
    curl -X POST localhost:8585/markets/add -d '{"slug": "fed-decision-in-march"}'
    curl -X POST localhost:8585/markets/remove -d '{"slug": "fed-decision-in-march"}'
"""

import json
import logging
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from main import InsiderTradingDetector


class ControlHandler(BaseHTTPRequestHandler):
    """HTTP request handler. Accesses the detector via self.server.detector."""

    # Suppress per-request log lines (we log ourselves)
    def log_message(self, format, *args):
        pass

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data, indent=2).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw)

    def do_GET(self):
        detector: "InsiderTradingDetector" = self.server.detector

        if self.path == "/status":
            self._send_json(detector.get_status())

        elif self.path == "/markets":
            slugs = list(set(detector.metadata_manager.condition_to_slug.values()))
            self._send_json({"slugs": sorted(slugs), "count": len(slugs)})

        else:
            self._send_json({"error": "Not found"}, 404)

    def do_POST(self):
        detector: "InsiderTradingDetector" = self.server.detector

        if self.path == "/markets/add":
            try:
                body = self._read_body()
                slug = body.get("slug", "").strip()
                if not slug:
                    self._send_json({"error": "Missing 'slug' in body"}, 400)
                    return
                result = detector.add_market(slug)
                self._send_json(result, 200 if result["ok"] else 400)
            except Exception as e:
                logging.error(f"Control API error on /markets/add: {e}")
                self._send_json({"error": str(e)}, 500)

        elif self.path == "/markets/remove":
            try:
                body = self._read_body()
                slug = body.get("slug", "").strip()
                if not slug:
                    self._send_json({"error": "Missing 'slug' in body"}, 400)
                    return
                result = detector.remove_market(slug)
                self._send_json(result, 200 if result["ok"] else 400)
            except Exception as e:
                logging.error(f"Control API error on /markets/remove: {e}")
                self._send_json({"error": str(e)}, 500)

        else:
            self._send_json({"error": "Not found"}, 404)


def start_control_api(detector: "InsiderTradingDetector", port: int = 8585):
    """Start the HTTP control API in a daemon thread."""
    server = HTTPServer(("0.0.0.0", port), ControlHandler)
    server.detector = detector  # Attach reference for handlers

    thread = threading.Thread(target=server.serve_forever, daemon=True, name="control-api")
    thread.start()
    logging.info(f"Control API listening on http://0.0.0.0:{port}")
    return server