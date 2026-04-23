"""
tracker_server.py — MLB Bet Tracker Local Server (read-only observer)

Exposes GET /stats (and /bets, /ping) for the Auto-Reconciliation Engine and
the rendered HTML report's live stats bar. All manual-entry endpoints
(/log_bet, /log_result) have been removed — grading now flows through
auto_grade_historical_picks() against actuals_2026.parquet.

Usage:
    python tracker_server.py          # Starts on port 5151
    python tracker_server.py --port 5252
"""

import os
import csv
import json
import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
TRACKER_FILE = os.path.join(PIPELINE_DIR, "data", "raw", "bet_tracker.csv")

FIELDNAMES = [
    "id", "date", "game", "home", "away", "model", "bet_type",
    "model_prob", "market_line", "book", "units", "result",
    "actual_total", "profit_loss", "logged_at", "notes"
]

def ensure_tracker():
    if not os.path.exists(TRACKER_FILE):
        with open(TRACKER_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()

def load_bets():
    ensure_tracker()
    bets = []
    with open(TRACKER_FILE, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            bets.append(row)
    return bets

class TrackerHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Suppress default logging

    def send_json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/bets":
            self.send_json(load_bets())
        elif parsed.path == "/stats":
            self.send_json(compute_stats())
        elif parsed.path == "/ping":
            self.send_json({"status": "ok"})
        else:
            self.send_json({"error": "not found"}, 404)

def compute_stats():
    bets = load_bets()
    settled = [b for b in bets if b["result"] in ("WIN","LOSS","PUSH")]
    wins    = [b for b in settled if b["result"] == "WIN"]
    losses  = [b for b in settled if b["result"] == "LOSS"]
    pending = [b for b in bets if b["result"] == "PENDING"]

    total_pl = sum(float(b["profit_loss"]) for b in settled if b["profit_loss"])
    total_u  = sum(float(b["units"]) for b in settled if b["units"])

    by_model = {}
    for b in settled:
        m = b["model"]
        if m not in by_model:
            by_model[m] = {"wins":0,"losses":0,"pl":0.0}
        if b["result"] == "WIN":
            by_model[m]["wins"] += 1
        elif b["result"] == "LOSS":
            by_model[m]["losses"] += 1
        by_model[m]["pl"] += float(b["profit_loss"] or 0)

    return {
        "total_bets":    len(settled),
        "wins":          len(wins),
        "losses":        len(losses),
        "pending":       len(pending),
        "win_rate":      round(len(wins)/len(settled)*100, 1) if settled else 0,
        "total_units":   round(total_u, 1),
        "profit_loss":   round(total_pl, 2),
        "roi":           round(total_pl/total_u*100, 1) if total_u else 0,
        "by_model":      by_model,
        "all_bets":      bets,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5151)
    args = parser.parse_args()

    ensure_tracker()
    print(f"\n{'='*50}")
    print(f"  MLB Bet Tracker Server (read-only) — port {args.port}")
    print(f"  Tracker file: {TRACKER_FILE}")
    print(f"  Endpoints: GET /stats, /bets, /ping")
    print(f"  Ctrl+C to stop")
    print(f"{'='*50}\n")

    server = HTTPServer(("localhost", args.port), TrackerHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nTracker server stopped.")

if __name__ == "__main__":
    main()
