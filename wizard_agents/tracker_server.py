"""
tracker_server.py
Lightweight HTTP server on port 5151 for bet logging from model_report.html.
This is Agent 6's runtime — it handles POST /log_bet, POST /log_result, GET /stats.

Start manually: python tracker_server.py
Auto-started by main.py before report generation.

Endpoints:
  POST /log_bet     → append_bet()
  POST /log_result  → log_result()
  GET  /stats       → read_tracker_stats()
  GET  /health      → {"status": "ok"}

Accepted `model` values (strict): "ML", "Totals", "Runline", "F5", "NRFI".
Enforcement lives in AGENT6's tool schema (tools/schemas.py:AGENT6_TOOLS);
this handler delegates validation there rather than duplicating the enum.
Legacy labels ("MFull", "MF5i", "MF3i", "MF1i", "MBat") are no longer accepted.
"""
from __future__ import annotations

import json
import logging
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

from agents.definitions import AGENT6
from orchestrator.agent_loop import run_agent
from config.settings import TRACKER_PORT

# Single source of truth for the current market vocabulary.  Validation is
# enforced by AGENT6's schema, but we pre-check here so a bad POST returns
# a fast, explicit error instead of the generic tool-executor wrapping.
ALLOWED_MODELS = ("ML", "Totals", "Runline", "F5", "NRFI")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [TRACKER] %(message)s")
logger = logging.getLogger("wizard.tracker")


class TrackerHandler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        logger.info(f"{self.address_string()} {format % args}")

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        raw    = self.rfile.read(length)
        return json.loads(raw) if raw else {}

    def do_OPTIONS(self):
        """CORS preflight for browser requests from model_report.html."""
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        if self.path == "/health":
            self._send_json({"status": "ok", "port": TRACKER_PORT})
            return

        if self.path == "/stats":
            result = AGENT6.tool_executor("read_tracker_stats", {})
            try:
                self._send_json(json.loads(result))
            except Exception:
                self._send_json({"error": result}, 500)
            return

        self._send_json({"error": f"Unknown path: {self.path}"}, 404)

    def do_POST(self):
        try:
            body = self._read_body()
        except Exception as e:
            self._send_json({"status": "ERROR", "error": f"Invalid JSON body: {e}"}, 400)
            return

        if self.path == "/log_bet":
            required = ["date", "game", "model", "bet_type", "model_prob", "market_line", "book", "units"]
            missing  = [f for f in required if f not in body]
            if missing:
                self._send_json({"status": "REJECTED", "error": f"Missing fields: {missing}"}, 400)
                return
            model_val = body.get("model")
            if model_val not in ALLOWED_MODELS:
                self._send_json({
                    "status": "REJECTED",
                    "error":  f"Invalid model '{model_val}'. Allowed: {list(ALLOWED_MODELS)}.",
                }, 400)
                return
            result = AGENT6.tool_executor("append_bet", body)
            self._send_json(json.loads(result))
            return

        if self.path == "/log_result":
            required = ["bet_id", "result"]
            missing  = [f for f in required if f not in body]
            if missing:
                self._send_json({"status": "REJECTED", "error": f"Missing fields: {missing}"}, 400)
                return
            result = AGENT6.tool_executor("log_result", body)
            parsed = json.loads(result)
            status = 400 if parsed.get("status") == "REJECTED" else 200
            self._send_json(parsed, status)
            return

        self._send_json({"error": f"Unknown path: {self.path}"}, 404)


def start_tracker_server(port: int = TRACKER_PORT, block: bool = True):
    server = HTTPServer(("localhost", port), TrackerHandler)
    logger.info(f"Tracker server running on http://localhost:{port}")
    logger.info("  POST /log_bet    → log a new bet")
    logger.info("  POST /log_result → record WIN/LOSS/PUSH")
    logger.info("  GET  /stats      → aggregate performance stats")
    if block:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            logger.info("Tracker server stopped.")
    return server


if __name__ == "__main__":
    start_tracker_server(block=True)
