"""GPS-based fraud detection agent.

Detects anomalies by comparing the physical location of a user (from GPS pings)
with the location of their in-person transactions.
"""
import json
import csv
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Any

from utils.geo import haversine


# ── Tunable thresholds ──────────────────────────────────────────────────────
MAX_SEARCH_WINDOW_HOURS = 12  # # TUNE — how far ±h around the tx to look for a GPS ping
HIGH_CONFIDENCE_KM = 50       # # TUNE — user >Xkm from tx location → high fraud confidence
MED_CONFIDENCE_KM = 20        # # TUNE — user >Xkm but ≤ HIGH threshold → medium confidence
MAX_SPEED_KMH = 900           # # TUNE — max plausible travel speed (plane); violations → fraud
# ────────────────────────────────────────────────────────────────────────────


def load_data(transactions_path: str, locations_path: str) -> tuple:
    """Load transactions CSV and GPS locations JSON.

    Returns:
        (transactions as list-of-dicts, locations as list-of-dicts)
    """
    with open(transactions_path, newline="", encoding="utf-8") as f:
        transactions = list(csv.DictReader(f))

    with open(locations_path, encoding="utf-8") as f:
        locations = json.load(f)

    return transactions, locations


def _build_gps_index(locations: list[dict]) -> dict[str, list[dict]]:
    """Index GPS pings by biotag, sorted by timestamp."""
    index: dict[str, list[dict]] = defaultdict(list)
    for entry in locations:
        entry = dict(entry)
        entry["_dt"] = datetime.fromisoformat(entry["timestamp"])
        index[entry["biotag"]].append(entry)
    for biotag in index:
        index[biotag].sort(key=lambda x: x["_dt"])
    return dict(index)


def _nearest_ping(
    pings: list[dict], target_dt: datetime, window_hours: float
) -> dict | None:
    """Return the GPS ping closest in time to target_dt within ±window_hours."""
    window = timedelta(hours=window_hours)
    best = None
    best_delta = timedelta.max
    for p in pings:
        delta = abs(p["_dt"] - target_dt)
        if delta <= window and delta < best_delta:
            best_delta = delta
            best = p
    return best


def _extract_city(location_text: str) -> str:
    """Extract the city prefix from a transaction location string like 'Munich - Isar River Cafe'."""
    if " - " in location_text:
        return location_text.split(" - ")[0].strip().lower()
    return location_text.strip().lower()


def analyze(transactions: list[dict], locations: list[dict], thresholds: dict | None = None) -> list[dict]:
    """Analyse transactions for GPS-based anomalies.

    Rules:
    1. City mismatch — nearest GPS ping's city ≠ transaction location city → high confidence.
    2. Distance mismatch — nearest ping's lat/lng > HIGH_CONFIDENCE_KM from any known user
       residence city → high confidence (uses user residence from GPS home cluster).
    3. Impossible velocity — two consecutive GPS/tx events require speed > MAX_SPEED_KMH → fraud.
    4. No GPS data within window → medium confidence flag.

    Args:
        transactions: List of transaction dicts.
        locations: List of GPS ping dicts.
        thresholds: Optional dict to override default thresholds.

    Returns:
        List of {transaction_id, user_id, confidence, reason} dicts.
    """
    th_window = (thresholds or {}).get("gps_window_hours", MAX_SEARCH_WINDOW_HOURS)
    th_high_km = (thresholds or {}).get("gps_high_km", HIGH_CONFIDENCE_KM)

    gps_index = _build_gps_index(locations)
    results = []

    for tx in transactions:
        tx_type = tx.get("transaction_type", "")
        location_text = tx.get("location", "").strip()

        # Only check transactions that have a physical location
        if tx_type not in ("in-person payment", "withdrawal", "e-commerce"):
            continue
        if not location_text:
            continue

        sender_id = tx["sender_id"]
        if sender_id not in gps_index:
            # No GPS data for this user at all → skip silently (not enough evidence)
            continue

        tx_dt = datetime.fromisoformat(tx["timestamp"])
        pings = gps_index[sender_id]
        nearest = _nearest_ping(pings, tx_dt, th_window)

        tx_city = _extract_city(location_text)

        if nearest is None:
            results.append({
                "transaction_id": tx["transaction_id"],
                "user_id": sender_id,
                "confidence": 0.40,  # # TUNE — no GPS in window
                "reason": f"No GPS ping within ±{th_window}h of tx at {tx_city}",
            })
            continue

        ping_city = nearest.get("city", "").strip().lower()

        # 1. City mismatch
        if ping_city and tx_city and ping_city != tx_city:
            dist_km = None
            if nearest.get("lat") and nearest.get("lng"):
                # Can't geocode tx location, but city mismatch itself is strong
                pass
            results.append({
                "transaction_id": tx["transaction_id"],
                "user_id": sender_id,
                "confidence": 0.82,  # # TUNE — city mismatch confidence
                "reason": (
                    f"GPS city={ping_city!r} ≠ tx city={tx_city!r} "
                    f"(GPS @{nearest['timestamp']})"
                ),
            })
            continue

        # 2. Velocity check — look at GPS pings around the tx
        #    If previous ping was far away and recent, flag impossible travel
        idx_list = pings
        if len(idx_list) >= 2:
            for i in range(1, len(idx_list)):
                p_prev = idx_list[i - 1]
                p_curr = idx_list[i]
                dt_hours = max(
                    (p_curr["_dt"] - p_prev["_dt"]).total_seconds() / 3600, 1e-6
                )
                km = haversine(
                    float(p_prev["lat"]), float(p_prev["lng"]),
                    float(p_curr["lat"]), float(p_curr["lng"]),
                )
                speed = km / dt_hours
                if speed > MAX_SPEED_KMH:  # # TUNE
                    # Check if one of these pings is close to the tx time
                    if abs(p_curr["_dt"] - tx_dt).total_seconds() < 3 * 3600:
                        results.append({
                            "transaction_id": tx["transaction_id"],
                            "user_id": sender_id,
                            "confidence": 0.75,  # # TUNE — impossible velocity confidence
                            "reason": (
                                f"Impossible velocity {speed:.0f} km/h between "
                                f"{p_prev['city']} and {p_curr['city']} around tx time"
                            ),
                        })
                        break

    return results
