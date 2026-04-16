"""Adaptive memory module for cross-level fraud pattern learning.

Saves fraud patterns after each world run and uses them to adjust
detection thresholds for the next level.
"""
import json
from collections import Counter
from datetime import datetime


def save_patterns(
    fraud_ids: list[str],
    transactions: list[dict],
    output_path: str,
) -> dict:
    """Compute and persist fraud pattern statistics.

    Statistics saved:
    - avg_fraud_amount: mean transaction amount among flagged txs.
    - fraud_hours: Counter of transaction hours (for night pattern tuning).
    - fraud_types: Counter of transaction types.
    - fraud_recipients: list of commonly targeted recipient IDs.
    - fraud_count / total_count: ratio for calibration.

    Args:
        fraud_ids: List of transaction IDs flagged as fraud.
        transactions: All transactions in this world.
        output_path: JSON file path to write patterns to.

    Returns:
        The patterns dict (also written to output_path).
    """
    fraud_set = set(fraud_ids)
    fraud_txs = [tx for tx in transactions if tx["transaction_id"] in fraud_set]

    if not fraud_txs:
        patterns = {
            "fraud_count": 0,
            "total_count": len(transactions),
            "avg_fraud_amount": 0.0,
            "fraud_hours": {},
            "fraud_types": {},
            "fraud_recipients": [],
        }
    else:
        amounts = [float(tx["amount"]) for tx in fraud_txs]
        hours = [datetime.fromisoformat(tx["timestamp"]).hour for tx in fraud_txs]
        types = [tx["transaction_type"] for tx in fraud_txs]
        recipients = [tx["recipient_id"] for tx in fraud_txs if tx.get("recipient_id")]

        patterns = {
            "fraud_count": len(fraud_txs),
            "total_count": len(transactions),
            "avg_fraud_amount": sum(amounts) / len(amounts),
            "fraud_hours": dict(Counter(hours)),
            "fraud_types": dict(Counter(types)),
            "fraud_recipients": [r for r, _ in Counter(recipients).most_common(20)],
        }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(patterns, f, indent=2)

    return patterns


def load_patterns(input_path: str) -> dict:
    """Load previously saved fraud patterns from a JSON file.

    Returns an empty dict if the file doesn't exist or is malformed.
    """
    try:
        with open(input_path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def adjust_thresholds(base_thresholds: dict, patterns: dict) -> dict:
    """Derive adjusted detection thresholds from prior-level fraud patterns.

    Adjustments:
    - If many night-hour frauds in previous level → lower night-hour confidence
      threshold (make it easier to flag night txs).
    - If avg fraud amount was high → lower the salary multiplier threshold.
    - If burst pattern was common → lower burst window threshold.

    Args:
        base_thresholds: Default threshold dict (can be empty).
        patterns: Previously saved patterns dict.

    Returns:
        Updated threshold dict. All keys are marked # TUNE in defaults.
    """
    if not patterns:
        return base_thresholds

    thresholds = dict(base_thresholds)

    fraud_count = patterns.get("fraud_count", 0)
    total_count = patterns.get("total_count", 1)
    fraud_ratio = fraud_count / total_count if total_count else 0

    # ── Night fraud adjustment ──────────────────────────────────────────────
    fraud_hours = patterns.get("fraud_hours", {})
    night_hours = {0, 1, 2, 3, 4}
    night_fraud = sum(fraud_hours.get(str(h), fraud_hours.get(h, 0)) for h in night_hours)
    if fraud_count > 0 and night_fraud / fraud_count > 0.3:  # # TUNE — >30% night fraud
        # Lower night confidence threshold to catch more night txs
        thresholds.setdefault("night_confidence", 0.35)
        thresholds["night_confidence"] = min(thresholds["night_confidence"] * 1.3, 0.6)  # # TUNE

    # ── High-amount adjustment ──────────────────────────────────────────────
    avg_amount = patterns.get("avg_fraud_amount", 0)
    if avg_amount > 0:
        # If previous frauds were high-amount, lower salary multiplier
        # (flag at a lower multiple of monthly salary)
        current_mult = thresholds.get("salary_multiplier", 0.5)
        # Reduce by 10% if average was very high  # # TUNE
        if avg_amount > 5000:
            thresholds["salary_multiplier"] = max(current_mult * 0.85, 0.2)  # # TUNE

    # ── Frequency adjustment ──────────────────────────────────────────────
    # If fraud_ratio is high, lower decision threshold slightly
    if fraud_ratio > 0.15:  # # TUNE
        current_thr = thresholds.get("decision_threshold", 0.45)
        thresholds["decision_threshold"] = max(current_thr * 0.90, 0.30)  # # TUNE

    return thresholds
