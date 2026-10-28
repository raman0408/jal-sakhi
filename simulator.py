"""Replay water readings through a lossy node-to-gateway network."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

MODEL_REASON, PH_REASON, TURB_REASON, TDS_REASON, LEVEL_REASON = 1, 2, 4, 8, 16
REASON_NAMES = {
    MODEL_REASON: "statistical anomaly", PH_REASON: "unsafe pH",
    TURB_REASON: "high turbidity", TDS_REASON: "high TDS", LEVEL_REASON: "low water level",
}


def model_from_header(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    def define(name: str) -> str:
        match = re.search(rf"#define\s+{name}\s+([^\s]+)", text)
        if not match:
            raise ValueError(f"{name} missing from {path}")
        return match.group(1).rstrip("f")
    node_id = define("NODE_ID").strip('"')
    threshold = float(define("SCORE_THRESHOLD"))
    mu_match = re.search(r"MU\[N_FEATURES\]\s*=\s*\{([^}]+)\}", text, re.S)
    matrix_match = re.search(r"LINV\[N_FEATURES\]\[N_FEATURES\]\s*=\s*\{(.*?)\n\};", text, re.S)
    if not mu_match or not matrix_match:
        raise ValueError("MU or LINV missing from model header")
    numbers = lambda value: [float(x.rstrip("f")) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?f?", value)]
    mu, flat = numbers(mu_match.group(1)), numbers(matrix_match.group(1))
    if len(mu) != 4 or len(flat) != 16:
        raise ValueError("expected a 4-feature model")
    return {"node_id": node_id, "threshold": threshold, "mu": mu, "linv": flat}


def verdict(row: dict[str, str], model: dict, history: int) -> tuple[int, int, float, int]:
    values = [float(row[name]) for name in ("tds", "turbidity", "ph", "temp_c")]
    transformed = []
    for i in range(4):
        transformed.append(sum(model["linv"][i * 4 + j] * (values[j] - model["mu"][j]) for j in range(i + 1)))
    score = math.sqrt(sum(v * v for v in transformed))
    tds, turbidity, ph, _temp = values
    level = float(row["level_pct"])
    reason = 0
    if score > model["threshold"]: reason |= MODEL_REASON
    if ph < 6.5 or ph > 8.5: reason |= PH_REASON
    if turbidity > 5.0: reason |= TURB_REASON
    if tds > 2000.0: reason |= TDS_REASON
    if level < 10.0: reason |= LEVEL_REASON
    history = ((history << 1) | bool(reason)) & 0xFF
    hard_water = reason & (PH_REASON | TURB_REASON | TDS_REASON)
    recent = history & 0x1F
    state = 2 if hard_water or (reason & (MODEL_REASON | LEVEL_REASON) and recent.bit_count() >= 3) else 1 if reason else 0
    return state, reason, score, history


def post_packet(url: str, packet: dict, timeout: float) -> bool:
    request = Request(url.rstrip("/") + "/packets", data=json.dumps(packet).encode(),
                      headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:
            return 200 <= response.status < 300
    except (URLError, TimeoutError):
        return False


def gateway_summary(url: str, timeout: float) -> dict | None:
    try:
        with urlopen(url.rstrip("/") + "/summary", timeout=timeout) as response:
            return json.loads(response.read())
    except (URLError, TimeoutError, json.JSONDecodeError):
        return None


def reason_labels(reason: int) -> list[str]:
    return [name for bit, name in REASON_NAMES.items() if reason & bit]


def main() -> None:
    parser = argparse.ArgumentParser(description="Jal Sakhi network simulation")
    parser.add_argument("--data", type=Path, default=Path("dataset.csv"))
    parser.add_argument("--model", type=Path, default=Path("model_params.h"))
    parser.add_argument("--gateway-url", default="http://127.0.0.1:5000")
    parser.add_argument("--loss", type=float, default=0.10, help="per-attempt packet loss, 0..1")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--outage-every", type=int, default=0, help="drop every Nth packet (0 disables)")
    parser.add_argument("--limit", type=int, default=0, help="replay only this many rows (0 means all)")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--report", type=Path, default=Path("simulation_report.json"))
    parser.add_argument("--evidence-limit", type=int, default=5,
                        help="maximum unsafe-packet examples saved in the report")
    args = parser.parse_args()
    if not 0 <= args.loss <= 1 or args.retries < 0 or args.outage_every < 0:
        parser.error("loss must be 0..1; retries and outage-every must be non-negative")

    model, rng, history = model_from_header(args.model), random.Random(args.seed), 0
    stats = {"readings": 0, "attempts": 0, "first_attempt_deliveries": 0,
             "retry_recoveries": 0, "delivered": 0, "failed_deliveries": 0,
             "network_dropped": 0, "forced_outage_attempts": 0,
             "gateway_unavailable": 0, "unsafe_samples": 0, "watch_samples": 0}
    confusion = {"true_positive": 0, "false_negative": 0,
                 "false_positive": 0, "true_negative": 0}
    evidence: list[dict] = []
    with args.data.open(newline="", encoding="utf-8") as source:
        for seq, row in enumerate(csv.DictReader(source), start=1):
            if args.limit and seq > args.limit: break
            state, reason, score, history = verdict(row, model, history)
            packet = {"node_id": model["node_id"], "seq": seq, "timestamp": row["ts"],
                      "tds": float(row["tds"]), "turbidity": float(row["turbidity"]),
                      "ph": float(row["ph"]), "temp_c": float(row["temp_c"]),
                      "level_pct": float(row["level_pct"]), "score": round(score, 4),
                      "verdict": state, "reason": reason}
            stats["readings"] += 1
            stats["unsafe_samples"] += state == 2
            stats["watch_samples"] += state == 1
            actual_event = int(row.get("label", "0")) == 1
            predicted_unsafe = state == 2
            if actual_event and predicted_unsafe:
                confusion["true_positive"] += 1
            elif actual_event:
                confusion["false_negative"] += 1
            elif predicted_unsafe:
                confusion["false_positive"] += 1
            else:
                confusion["true_negative"] += 1
            forced_outage = args.outage_every and seq % args.outage_every == 0
            sent = False
            attempts_for_packet = 0
            for attempt_index in range(args.retries + 1):
                attempts_for_packet += 1
                stats["attempts"] += 1
                if forced_outage or rng.random() < args.loss:
                    stats["network_dropped"] += 1
                    stats["forced_outage_attempts"] += bool(forced_outage)
                    continue
                if post_packet(args.gateway_url, packet, timeout=2.0):
                    stats["delivered"] += 1
                    if attempt_index == 0:
                        stats["first_attempt_deliveries"] += 1
                    else:
                        stats["retry_recoveries"] += 1
                    sent = True
                    break
                stats["gateway_unavailable"] += 1
            if not sent:
                packet["delivery"] = "failed"
                stats["failed_deliveries"] += 1
            if state == 2 and len(evidence) < args.evidence_limit:
                evidence.append({"seq": seq, "timestamp": packet["timestamp"], "score": packet["score"],
                                 "reasons": reason_labels(reason), "delivery": "delivered" if sent else "failed",
                                 "attempts_used": attempts_for_packet})
    stats["delivery_rate"] = round(stats["delivered"] / stats["readings"], 4) if stats["readings"] else 0
    actual_events = confusion["true_positive"] + confusion["false_negative"]
    normal_samples = confusion["true_negative"] + confusion["false_positive"]
    detection = confusion | {"unsafe_recall": round(confusion["true_positive"] / actual_events, 4) if actual_events else None,
                             "false_positive_rate": round(confusion["false_positive"] / normal_samples, 4) if normal_samples else None}
    reliability = {"delivery_rate": stats["delivery_rate"],
                   "first_attempt_success_rate": round(stats["first_attempt_deliveries"] / stats["readings"], 4) if stats["readings"] else 0,
                   "retry_recovery_rate": round(stats["retry_recoveries"] / stats["readings"], 4) if stats["readings"] else 0,
                   "mean_attempts_per_reading": round(stats["attempts"] / stats["readings"], 3) if stats["readings"] else 0}
    report = {"experiment": "Jal Sakhi CSV telemetry replay over a configurable lossy link",
              "configuration": vars(args) | {"data": str(args.data), "model": str(args.model), "report": str(args.report)},
              "network_model": {"loss": "per-attempt simulated link loss", "retries": "maximum additional sends after the first attempt", "outage_every": "all attempts for every Nth packet are dropped"},
              "statistics": stats, "reliability": reliability, "detection_against_dataset_labels": detection,
              "unsafe_packet_examples": evidence, "gateway_summary_after_run": gateway_summary(args.gateway_url, timeout=2.0)}
    args.report.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
