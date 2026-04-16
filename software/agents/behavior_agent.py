"""Behavioral fraud detection agent.

Detects anomalies based on transaction patterns vs. user financial profile.
All thresholds are marked # TUNE.
"""
import csv
import json
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any


# ── Tunable thresholds ──────────────────────────────────────────────────────
SALARY_MULTIPLIER = 0.5          # # TUNE — flag if single tx > X * monthly_salary
NIGHT_HOUR_START = 0             # # TUNE — suspicious night window start (inclusive)
NIGHT_HOUR_END = 5               # # TUNE — suspicious night window end (exclusive)
NIGHT_CONFIDENCE = 0.35          # # TUNE — confidence boost for night transactions
BURST_WINDOW_MINUTES = 5         # # TUNE — window to count rapid-fire transactions
BURST_COUNT_THRESHOLD = 3        # # TUNE — 3+ tx in window → suspicious
BURST_CONFIDENCE = 0.55          # # TUNE
LOW_BALANCE_RATIO = 0.05         # # TUNE — balance < X * monthly_salary → suspicious
LOW_BALANCE_CONFIDENCE = 0.45    # # TUNE
FREQ_SPIKE_MULTIPLIER = 3.0      # # TUNE — daily tx count > X * user baseline → suspicious
FREQ_SPIKE_CONFIDENCE = 0.50     # # TUNE
UNKNOWN_RECIPIENT_CONFIDENCE = 0.30  # # TUNE — recipient never seen before
HIGH_AMOUNT_CONFIDENCE = 0.70    # # TUNE — amount > salary multiplier threshold
# ────────────────────────────────────────────────────────────────────────────


def load_data(transactions_path: str, users_path: str) -> tuple:
    """Load transactions CSV and users JSON.

    Returns:
        (transactions as list-of-dicts, users as list-of-dicts)
    """
    with open(transactions_path, newline="", encoding="utf-8") as f:
        transactions = list(csv.DictReader(f))

    with open(users_path, encoding="utf-8") as f:
        users = json.load(f)

    return transactions, users


def _build_user_map(transactions: list[dict], users: list[dict]) -> dict:
    """Map sender_id (biotag) to user profile using IBAN matching."""
    iban_to_user = {u["iban"]: u for u in users}
    user_map: dict[str, dict] = {}
    for tx in transactions:
        sid = tx["sender_id"]
        if sid in user_map:
            continue
        iban = tx.get("sender_iban", "")
        if iban in iban_to_user:
            user_map[sid] = iban_to_user[iban]
    return user_map


def _monthly_salary(user: dict) -> float:
    """Return monthly salary (annual / 12)."""
    return float(user.get("salary", 0)) / 12.0


def _build_known_recipients(transactions: list[dict]) -> dict[str, set]:
    """Build per-user set of known recipients from ALL transactions (history)."""
    known: dict[str, set] = defaultdict(set)
    for tx in transactions:
        known[tx["sender_id"]].add(tx["recipient_id"])
    return known


def _compute_daily_baseline(transactions: list[dict]) -> dict[str, float]:
    """Compute average daily transaction count per user over the full dataset."""
    daily: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for tx in transactions:
        day = tx["timestamp"][:10]
        daily[tx["sender_id"]][day] += 1
    baseline: dict[str, float] = {}
    for uid, days in daily.items():
        counts = list(days.values())
        baseline[uid] = sum(counts) / len(counts) if counts else 1.0
    return baseline


def analyze(
    transactions: list[dict],
    users: list[dict],
    thresholds: dict | None = None,
) -> list[dict]:
    """Analyse transactions for behavioral anomalies.

    Rules applied per transaction:
    1. High amount — amount > SALARY_MULTIPLIER * monthly_salary.
    2. Night transaction — between NIGHT_HOUR_START and NIGHT_HOUR_END.
    3. Unknown recipient — first time sender sends to this recipient.
    4. Burst — 3+ transactions within BURST_WINDOW_MINUTES.
    5. Low post-tx balance — balance_after < LOW_BALANCE_RATIO * monthly_salary.
    6. Frequency spike — daily tx count > FREQ_SPIKE_MULTIPLIER * user baseline.

    Multiple signals are combined into a single confidence score.

    Args:
        transactions: All transactions for the world.
        users: All user profiles.
        thresholds: Optional dict to override defaults.

    Returns:
        List of {transaction_id, user_id, confidence, reason} dicts.
    """
    th = thresholds or {}
    sal_mult = th.get("salary_multiplier", SALARY_MULTIPLIER)
    night_start = th.get("night_hour_start", NIGHT_HOUR_START)
    night_end = th.get("night_hour_end", NIGHT_HOUR_END)
    burst_win = th.get("burst_window_minutes", BURST_WINDOW_MINUTES)
    burst_thr = th.get("burst_count_threshold", BURST_COUNT_THRESHOLD)
    freq_mult = th.get("freq_spike_multiplier", FREQ_SPIKE_MULTIPLIER)
    low_bal = th.get("low_balance_ratio", LOW_BALANCE_RATIO)

    user_map = _build_user_map(transactions, users)
    known_recipients = _build_known_recipients(transactions)
    baseline = _compute_daily_baseline(transactions)

    # Build sorted per-user timeline for burst detection
    by_user: dict[str, list[dict]] = defaultdict(list)
    for tx in transactions:
        by_user[tx["sender_id"]].append(tx)
    for uid in by_user:
        by_user[uid].sort(key=lambda x: x["timestamp"])

    # Daily counts for frequency spike
    daily_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for tx in transactions:
        day = tx["timestamp"][:10]
        daily_counts[tx["sender_id"]][day] += 1

    results = []
    seen_tx: set[str] = set()

    for tx in transactions:
        tid = tx["transaction_id"]
        if tid in seen_tx:
            continue

        sender_id = tx["sender_id"]
        # Only analyse known users (skip employer EMP* IDs)
        if sender_id not in user_map:
            continue

        user = user_map[sender_id]
        monthly = _monthly_salary(user)
        tx_dt = datetime.fromisoformat(tx["timestamp"])
        amount = float(tx.get("amount", 0))
        balance = float(tx.get("balance_after", 0))

        signals: list[str] = []
        confidence = 0.0

        # 1. High amount
        if monthly > 0 and amount > sal_mult * monthly:  # # TUNE
            signals.append(f"amount {amount:.0f} > {sal_mult}x monthly salary {monthly:.0f}")
            confidence = max(confidence, HIGH_AMOUNT_CONFIDENCE)

        # 2. Night transaction
        hour = tx_dt.hour
        if night_start <= hour < night_end:  # # TUNE
            signals.append(f"night transaction at {hour:02d}:00")
            confidence = max(confidence, NIGHT_CONFIDENCE)

        # 3. Unknown recipient (for person-to-person transfers)
        if tx.get("transaction_type") == "transfer":
            recipient = tx.get("recipient_id", "")
            if recipient and recipient not in known_recipients.get(sender_id, set()) - {recipient}:
                # Actually the set includes this tx too, so check if seen before this tx
                prev_recipients = set()
                for prev_tx in by_user[sender_id]:
                    if prev_tx["timestamp"] < tx["timestamp"]:
                        prev_recipients.add(prev_tx["recipient_id"])
                if recipient not in prev_recipients and not recipient.startswith("EMP"):
                    signals.append(f"first-time recipient {recipient}")
                    confidence = max(confidence, UNKNOWN_RECIPIENT_CONFIDENCE)

        # 4. Burst detection
        user_txs = by_user[sender_id]
        window = timedelta(minutes=burst_win)
        window_txs = [
            t for t in user_txs
            if abs(datetime.fromisoformat(t["timestamp"]) - tx_dt) <= window
            and t["transaction_id"] != tid
        ]
        if len(window_txs) >= burst_thr - 1:  # -1 because we count the current tx too
            signals.append(
                f"burst: {len(window_txs)+1} txs in {burst_win}-min window"
            )
            confidence = max(confidence, BURST_CONFIDENCE)

        # 5. Low balance after transaction
        if monthly > 0 and balance < low_bal * monthly and balance >= 0:  # # TUNE
            signals.append(f"balance after tx {balance:.0f} < {low_bal}x monthly {monthly:.0f}")
            confidence = max(confidence, LOW_BALANCE_CONFIDENCE)

        # 6. Frequency spike
        day = tx["timestamp"][:10]
        day_count = daily_counts[sender_id].get(day, 0)
        user_baseline = baseline.get(sender_id, 1.0)
        if day_count > freq_mult * user_baseline:  # # TUNE
            signals.append(
                f"freq spike: {day_count} txs on {day} vs baseline {user_baseline:.1f}"
            )
            confidence = max(confidence, FREQ_SPIKE_CONFIDENCE)

        if signals:
            seen_tx.add(tid)
            results.append({
                "transaction_id": tid,
                "user_id": sender_id,
                "confidence": round(confidence, 3),
                "reason": "; ".join(signals),
            })

    return results
