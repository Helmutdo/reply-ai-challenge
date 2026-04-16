"""Multi-agent fraud detection system — Reply AI Hackathon.

Usage:
    python main.py --world "../../Brave+New+World+-+train/Brave New World - train"
    python main.py --world /path/to/world --memory /path/to/patterns.json
"""
import os
import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import ulid
from langfuse import Langfuse, observe
from langfuse.langchain import CallbackHandler

load_dotenv()

# ── LangChain / OpenRouter model ────────────────────────────────────────────
model = ChatOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="gpt-4o-mini",
    temperature=0.7,
    max_tokens=50,
)

# ── Langfuse client ──────────────────────────────────────────────────────────
langfuse_client = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "https://challenges.reply.com/langfuse"),
)


def generate_session_id() -> str:
    """Generate a unique session ID prefixed with the team name."""
    team = os.getenv("TEAM_NAME", "tutorial").replace(" ", "-")
    return f"{team}-{ulid.new().str}"


def invoke_langchain(model, prompt, langfuse_handler, session_id):
    """Invoke LangChain model with Langfuse tracing."""
    messages = [HumanMessage(content=prompt)]
    response = model.invoke(
        messages,
        config={
            "callbacks": [langfuse_handler],
            "metadata": {"langfuse_session_id": session_id},
        },
    )
    return response.content


@observe()
def run_llm_call(session_id, model, prompt):
    """Single traced LLM call (kept from original boilerplate)."""
    langfuse_handler = CallbackHandler(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_HOST", "https://challenges.reply.com/langfuse"),
        session_id=session_id,
    )
    return invoke_langchain(model, prompt, langfuse_handler, session_id)


def build_iban_to_biotag(transactions: list[dict]) -> dict[str, str]:
    """Build IBAN → biotag mapping from transactions (skips employer IDs)."""
    mapping: dict[str, str] = {}
    for tx in transactions:
        sid = tx["sender_id"]
        iban = tx.get("sender_iban", "")
        if iban and not sid.startswith("EMP") and iban not in mapping:
            mapping[iban] = sid
    return mapping


@observe()
def run_fraud_detection(
    world_path: str,
    memory_path: str | None,
    session_id: str,
) -> tuple[list[str], dict, int]:
    """Orchestrate all agents and return fraud IDs, reasoning, LLM call count.

    Args:
        world_path: Path to the world folder with all data files.
        memory_path: Optional path to prior-level patterns JSON.
        session_id: Langfuse session ID.

    Returns:
        (fraud_ids, reasoning_dict, llm_call_count)
    """
    from agents import gps_agent, behavior_agent, comms_agent
    from core import decision, memory

    wp = Path(world_path)
    tx_path = str(wp / "transactions.csv")
    loc_path = str(wp / "locations.json")
    usr_path = str(wp / "users.json")
    sms_path = str(wp / "sms.json")
    mail_path = str(wp / "mails.json")

    # ── Load prior patterns and adjust thresholds ───────────────────────────
    patterns = memory.load_patterns(memory_path) if memory_path else {}
    thresholds = memory.adjust_thresholds({}, patterns)

    print(f"  [memory] loaded {len(patterns)} pattern keys, "
          f"threshold overrides: {thresholds}")

    # ── Load data ────────────────────────────────────────────────────────────
    transactions, locations = gps_agent.load_data(tx_path, loc_path)
    _, users = behavior_agent.load_data(tx_path, usr_path)
    sms_list, mails_list = comms_agent.load_data(sms_path, mail_path)

    iban_to_biotag = build_iban_to_biotag(transactions)

    print(f"  [data] {len(transactions)} transactions, "
          f"{len(locations)} GPS pings, "
          f"{len(users)} users")

    # ── Agent 1: GPS ─────────────────────────────────────────────────────────
    gps_results = gps_agent.analyze(transactions, locations, thresholds)
    print(f"  [gps_agent] {len(gps_results)} anomalies detected")

    # ── Agent 2: Behavior ────────────────────────────────────────────────────
    behavior_results = behavior_agent.analyze(transactions, users, thresholds)
    print(f"  [behavior_agent] {len(behavior_results)} anomalies detected")

    # ── Agent 3: Comms (only for flagged users) ──────────────────────────────
    flagged_users: set[str] = set()
    for r in gps_results:
        flagged_users.add(r["user_id"])
    for r in behavior_results:
        flagged_users.add(r["user_id"])

    comms_results, llm_calls = comms_agent.analyze(
        flagged_users,
        users,
        iban_to_biotag,
        sms_list,
        mails_list,
        model,
        session_id,
    )
    print(f"  [comms_agent] {len(comms_results)} user comms analysed, "
          f"{llm_calls} LLM calls, "
          f"{sum(1 for r in comms_results if r.get('phishing_detected'))} phishing detected")

    # ── Decision combiner ────────────────────────────────────────────────────
    fraud_ids, reasoning = decision.combine_scores(
        gps_results, behavior_results, comms_results, transactions, thresholds
    )
    print(f"  [decision] {len(fraud_ids)} transactions flagged as fraud "
          f"(threshold={thresholds.get('decision_threshold', decision.DECISION_THRESHOLD)})")

    return fraud_ids, reasoning, llm_calls


def main():
    parser = argparse.ArgumentParser(description="Reply AI Hackathon — Fraud Detection")
    parser.add_argument(
        "--world",
        required=True,
        help='Path to world folder, e.g. "../../Brave+New+World+-+train/Brave New World - train"',
    )
    parser.add_argument(
        "--memory",
        default=None,
        help="Path to prior-level patterns JSON (optional)",
    )
    args = parser.parse_args()

    world_path = args.world
    world_name = Path(world_path).name.replace(" ", "_").replace("-", "_")

    session_id = generate_session_id()
    print(f"\n=== Fraud Detection Run ===")
    print(f"World:   {world_path}")
    print(f"Memory:  {args.memory or '(none)'}")
    print(f"Session: {session_id}\n")

    fraud_ids, reasoning, llm_calls = run_fraud_detection(
        world_path, args.memory, session_id
    )

    # ── Output ────────────────────────────────────────────────────────────────
    output_dir = Path(__file__).resolve().parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    # 1. Submission file: ASCII, IDs only, newline separated
    output_file = output_dir / f"output_{world_name}.txt"
    with open(output_file, "w", encoding="ascii") as f:
        for tid in fraud_ids:
            f.write(tid + "\n")
    print(f"\n  [output] Written transaction IDs (ASCII) → {output_file}")

    # 2. Detailed report for review (UTF-8)
    report_file = output_dir / f"report_{world_name}.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(f"FRAUD DETECTION REPORT\n")
        f.write(f"World: {world_name}\n")
        f.write(f"Session: {session_id}\n")
        f.write("="*60 + "\n\n")
        f.write(f"SUMMARY\n")
        f.write(f"Total Transactions Flagged: {len(fraud_ids)}\n")
        f.write("-" * 60 + "\n\n")
        
        for tid in fraud_ids:
            r = reasoning.get(tid, {})
            f.write(f"TRANSACTION ID: {tid}\n")
            f.write(f"  Confidence: {r.get('combined', 0)*100:.1f}%\n")
            if r.get('gps_reason'):
                f.write(f"  - GPS Signal: {r['gps_reason']}\n")
            if r.get('behavior_reason'):
                f.write(f"  - Behavioral Signal: {r['behavior_reason']}\n")
            if r.get('comms_score', 0) > 0:
                f.write(f"  - Communications Signal: Social engineering/phishing detected for this user.\n")
            f.write("\n")

    # ── Save patterns for next level ─────────────────────────────────────────
    from agents import gps_agent as _gps  # already imported but re-import for clarity
    from core import memory as _mem

    # Reload transactions for pattern saving
    from pathlib import Path as _Path
    import csv as _csv
    with open(str(_Path(world_path) / "transactions.csv"), newline="", encoding="utf-8") as f:
        transactions = list(_csv.DictReader(f))

    patterns_file = output_dir / f"patterns_{world_name}.json"
    saved = _mem.save_patterns(fraud_ids, transactions, str(patterns_file))
    print(f"  [memory] Patterns saved → {patterns_file} "
          f"(fraud_count={saved['fraud_count']})")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"Total transactions : {len(transactions)}")
    print(f"Flagged as fraud   : {len(fraud_ids)}")
    print(f"Fraud ratio        : {len(fraud_ids)/len(transactions)*100:.1f}%")
    print(f"LLM calls made     : {llm_calls}")
    print(f"Session ID         : {session_id}")
    print(f"{'='*50}\n")

    langfuse_client.flush()
    print("Langfuse traces flushed.")


if __name__ == "__main__":
    main()
