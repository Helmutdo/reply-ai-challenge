"""Decision engine: combines GPS, behavior, and comms agent scores.

Produces a final list of fraudulent transaction IDs with reasoning.
"""
from collections import defaultdict


# ── Tunable weights and thresholds ─────────────────────────────────────────
GPS_WEIGHT = 0.45           # # TUNE — GPS anomaly is very strong evidence
BEHAVIOR_WEIGHT = 0.35      # # TUNE
COMMS_WEIGHT = 0.20         # # TUNE
DECISION_THRESHOLD = 0.45   # # TUNE — final score ≥ this → fraud
GPS_BOOST = 0.20            # # TUNE — extra boost when GPS anomaly exists
# ────────────────────────────────────────────────────────────────────────────


def combine_scores(
    gps_results: list[dict],
    behavior_results: list[dict],
    comms_results: list[dict],
    transactions: list[dict],
    thresholds: dict | None = None,
) -> tuple[list[str], dict]:
    """Combine agent scores into a final fraud decision.

    Scoring logic:
    - Each transaction starts at 0.
    - GPS, behavior, and comms scores are weighted and summed.
    - If a GPS anomaly exists for the transaction, GPS_BOOST is added.
    - Transactions above DECISION_THRESHOLD are flagged as fraud.

    Args:
        gps_results: List of {transaction_id, user_id, confidence, reason}.
        behavior_results: List of {transaction_id, user_id, confidence, reason}.
        comms_results: List of {user_id, phishing_detected, confidence, reason}.
        transactions: Full transaction list (for user_id → tx lookup).
        thresholds: Optional dict to override default weights/threshold.

    Returns:
        (fraud_transaction_ids list, reasoning_dict per transaction_id)
    """
    th = thresholds or {}
    gps_w = th.get("gps_weight", GPS_WEIGHT)
    beh_w = th.get("behavior_weight", BEHAVIOR_WEIGHT)
    com_w = th.get("comms_weight", COMMS_WEIGHT)
    dec_thr = th.get("decision_threshold", DECISION_THRESHOLD)
    gps_boost = th.get("gps_boost", GPS_BOOST)

    # Index by transaction_id
    gps_map: dict[str, dict] = {r["transaction_id"]: r for r in gps_results}
    beh_map: dict[str, dict] = {r["transaction_id"]: r for r in behavior_results}

    # Comms results are per-user; map user_id → confidence
    comms_user_map: dict[str, float] = {}
    for r in comms_results:
        if r.get("phishing_detected"):
            comms_user_map[r["user_id"]] = float(r.get("confidence", 0))

    # Build user_id → transactions map (for comms propagation)
    user_to_txids: dict[str, list[str]] = defaultdict(list)
    for tx in transactions:
        user_to_txids[tx["sender_id"]].append(tx["transaction_id"])

    # Collect all transaction IDs that appear in any agent output
    candidate_txids: set[str] = (
        set(gps_map.keys()) | set(beh_map.keys())
    )
    # Also add transactions for users with phishing detection
    for uid, conf in comms_user_map.items():
        for txid in user_to_txids.get(uid, []):
            candidate_txids.add(txid)

    fraud_ids: list[str] = []
    reasoning: dict[str, dict] = {}

    for txid in candidate_txids:
        gps_entry = gps_map.get(txid)
        beh_entry = beh_map.get(txid)

        gps_score = gps_entry["confidence"] if gps_entry else 0.0
        beh_score = beh_entry["confidence"] if beh_entry else 0.0

        # Determine user_id for comms lookup
        user_id = None
        if gps_entry:
            user_id = gps_entry.get("user_id")
        elif beh_entry:
            user_id = beh_entry.get("user_id")

        com_score = comms_user_map.get(user_id, 0.0) if user_id else 0.0

        # Weighted sum
        combined = gps_w * gps_score + beh_w * beh_score + com_w * com_score

        # GPS boost: if GPS signal is present and strong
        if gps_entry and gps_score >= 0.70:  # # TUNE — boost threshold
            combined += gps_boost

        combined = min(combined, 1.0)

        reasoning[txid] = {
            "gps_score": round(gps_score, 3),
            "behavior_score": round(beh_score, 3),
            "comms_score": round(com_score, 3),
            "combined": round(combined, 3),
            "gps_reason": gps_entry["reason"] if gps_entry else None,
            "behavior_reason": beh_entry["reason"] if beh_entry else None,
            "fraud": combined >= dec_thr,
        }

        if combined >= dec_thr:
            fraud_ids.append(txid)

    return fraud_ids, reasoning
