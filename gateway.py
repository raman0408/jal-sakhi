"""Local gateway for the Jal Sakhi network simulation (standard library only)."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock


class Gateway:
    def __init__(self, event_log: Path | None = None) -> None:
        self.event_log = event_log
        self.seen: set[tuple[str, int]] = set()
        self.packets: list[dict] = []
        self.alerts: list[dict] = []
        self._lock = Lock()

    def receive(self, packet: dict) -> tuple[dict, bool]:
        required = {"node_id", "seq", "timestamp", "verdict", "reason"}
        missing = required - packet.keys()
        if missing:
            raise ValueError("missing fields: " + ", ".join(sorted(missing)))
        key = (str(packet["node_id"]), int(packet["seq"]))
        with self._lock:
            if key in self.seen:
                return {"status": "duplicate", "seq": key[1]}, True
            self.seen.add(key)
            packet["received_at"] = datetime.now(timezone.utc).isoformat()
            self.packets.append(packet)
            if int(packet["verdict"]) == 2:
                ticket = {
                    "ticket_id": f"JS-{len(self.alerts) + 1:04d}",
                    "node_id": packet["node_id"],
                    "seq": packet["seq"],
                    "timestamp": packet["timestamp"],
                    "reason": packet["reason"],
                    "status": "open",
                }
                self.alerts.append(ticket)
            if self.event_log:
                with self.event_log.open("a", encoding="utf-8") as out:
                    out.write(json.dumps(packet) + "\n")
            return {"status": "accepted", "seq": key[1]}, False

    def summary(self) -> dict:
        with self._lock:
            return {
                "packets_received": len(self.packets),
                "open_alerts": len(self.alerts),
                "alerts": self.alerts,
            }


def make_handler(gateway: Gateway):
    class Handler(BaseHTTPRequestHandler):
        def _json(self, status: int, body: dict) -> None:
            data = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path in ("/health", "/summary"):
                self._json(HTTPStatus.OK, gateway.summary())
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path != "/packets":
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if size <= 0 or size > 16384:
                    raise ValueError("invalid body size")
                packet = json.loads(self.rfile.read(size))
                response, duplicate = gateway.receive(packet)
                self._json(HTTPStatus.OK if duplicate else HTTPStatus.CREATED, response)
            except (ValueError, json.JSONDecodeError) as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

        def log_message(self, _format: str, *_args: object) -> None:
            return

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Jal Sakhi simulated gateway")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--event-log", type=Path, default=Path("gateway_events.jsonl"))
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(Gateway(args.event_log)))
    print(f"Gateway listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nGateway stopped")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
