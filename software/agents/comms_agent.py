"""Communications fraud detection agent.

Uses an LLM to detect phishing/social-engineering signals in SMS and email
threads. Only invoked for users already flagged by GPS or behavior agents
(cost saving).
"""
import json
import re
import os
from collections import defaultdict
from langfuse import observe
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langfuse.langchain import CallbackHandler
from dotenv import load_dotenv

load_dotenv()


def load_data(sms_path: str, mails_path: str) -> tuple:
    """Load SMS and mail JSON files.

    Returns:
        (sms_list, mails_list) — raw lists of dicts with 'sms' and 'mail' keys.
    """
    with open(sms_path, encoding="utf-8") as f:
        sms_list = json.load(f)
    with open(mails_path, encoding="utf-8") as f:
        mails_list = json.load(f)
    return sms_list, mails_list


def _extract_phone(sms_text: str) -> str | None:
    """Extract the 'To:' phone number from an SMS text block."""
    m = re.search(r"^To:\s*(.+)", sms_text, re.MULTILINE)
    return m.group(1).strip() if m else None


def _extract_email(mail_text: str) -> str | None:
    """Extract the recipient email from a mail's 'To:' header."""
    m = re.search(r"^To:.*?<([^>]+)>", mail_text, re.MULTILINE)
    if m:
        return m.group(1).strip().lower()
    m = re.search(r"^To:\s*(\S+@\S+)", mail_text, re.MULTILINE)
    return m.group(1).strip().lower() if m else None


def _build_user_comms_index(
    users: list[dict],
    iban_to_biotag: dict[str, str],
    sms_list: list[dict],
    mails_list: list[dict],
) -> dict[str, dict]:
    """Map biotag → {sms_texts: [], mail_texts: []}.

    Builds the mapping by:
    - SMS: matching phone → user first name (Hi <name>) → biotag via IBAN lookup.
    - Mail: matching email (firstname.lastname@example.com) → biotag.
    """
    # Build name→biotag map (first name, lower)
    name_to_biotag: dict[str, str] = {}
    for user in users:
        fn = user.get("first_name", "").lower()
        iban = user.get("iban", "")
        btag = iban_to_biotag.get(iban)
        if btag:
            name_to_biotag[fn] = btag

    # Build email→biotag map: "jim.ortiz@example.com" → biotag
    email_to_biotag: dict[str, str] = {}
    for user in users:
        fn = user.get("first_name", "").lower()
        ln = user.get("last_name", "").lower()
        iban = user.get("iban", "")
        btag = iban_to_biotag.get(iban)
        if btag:
            email = f"{fn}.{ln}@example.com"
            email_to_biotag[email] = btag

    # Group SMS by phone, then resolve phone→biotag via name matching
    phone_sms: dict[str, list[str]] = defaultdict(list)
    for entry in sms_list:
        text = entry.get("sms", "")
        phone = _extract_phone(text)
        if phone:
            phone_sms[phone].append(text)

    phone_to_btag: dict[str, str] = {}
    for phone, texts in phone_sms.items():
        for text in texts:
            names = re.findall(r"Hi ([A-Z][a-z]+)", text)
            for name in names:
                btag = name_to_biotag.get(name.lower())
                if btag:
                    phone_to_btag[phone] = btag
                    break
            if phone in phone_to_btag:
                break

    # Build comms index
    index: dict[str, dict] = defaultdict(lambda: {"sms_texts": [], "mail_texts": []})

    for phone, texts in phone_sms.items():
        btag = phone_to_btag.get(phone)
        if btag:
            index[btag]["sms_texts"].extend(texts)

    for entry in mails_list:
        text = entry.get("mail", "")
        email = _extract_email(text)
        if email:
            btag = email_to_biotag.get(email)
            if btag:
                index[btag]["mail_texts"].append(text)

    return dict(index)


_PHISHING_PROMPT = """You are a financial fraud analyst. Analyse the following communications for signs of phishing or social engineering.

Look for:
- Urgency language ("act now", "account suspended", "verify immediately")
- Fake bank/institution alerts requesting credentials or OTP
- Suspicious links or unexpected login requests
- Impersonation of banks, government agencies, or known contacts
- Requests to transfer funds urgently

Communications:
{comms_text}

Reply with ONLY a JSON object:
{{"phishing_detected": true/false, "confidence": 0.0-1.0, "reason": "brief explanation"}}"""


@observe()
def _call_llm_phishing(
    user_id: str,
    comms_text: str,
    model: ChatOpenAI,
    langfuse_handler: CallbackHandler,
    session_id: str,
) -> dict:
    """Call LLM to detect phishing in the given communications text."""
    prompt = _PHISHING_PROMPT.format(comms_text=comms_text[:4000])  # truncate to save tokens
    messages = [HumanMessage(content=prompt)]
    response = model.invoke(
        messages,
        config={
            "callbacks": [langfuse_handler],
            "metadata": {"langfuse_session_id": session_id, "user_id": user_id},
        },
    )
    raw = response.content.strip()
    # Strip markdown fences if present
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback if LLM doesn't follow format
        result = {"phishing_detected": False, "confidence": 0.0, "reason": "parse error"}
    result["user_id"] = user_id
    return result


def analyze(
    suspicious_user_ids: set[str],
    users: list[dict],
    iban_to_biotag: dict[str, str],
    sms_list: list[dict],
    mails_list: list[dict],
    model: ChatOpenAI,
    session_id: str,
) -> tuple[list[dict], int]:
    """Detect phishing/social-engineering for already-flagged users only.

    Skips users with no communications data to avoid unnecessary LLM calls.

    Args:
        suspicious_user_ids: Set of biotag IDs flagged by GPS or behavior agents.
        users: Full user profile list.
        iban_to_biotag: IBAN → biotag mapping built from transactions.
        sms_list: Raw SMS records.
        mails_list: Raw mail records.
        model: LangChain ChatOpenAI instance.
        session_id: Langfuse session ID.

    Returns:
        (results list of {user_id, phishing_detected, confidence, reason}, llm_call_count)
    """
    # Use temperature=0, max_tokens=100 for this agent  # # TUNE
    phishing_model = ChatOpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        model="gpt-4o-mini",
        temperature=0,      # TUNE
        max_tokens=100,     # TUNE
    )

    comms_index = _build_user_comms_index(users, iban_to_biotag, sms_list, mails_list)

    results = []
    llm_calls = 0

    for user_id in suspicious_user_ids:
        comms = comms_index.get(user_id)
        if not comms:
            continue

        # Build a condensed comms text — only keep suspicious-looking snippets
        sms_texts = comms.get("sms_texts", [])
        mail_texts = comms.get("mail_texts", [])

        if not sms_texts and not mail_texts:
            continue

        # Extract short snippets (first 200 chars) to save tokens
        sms_snippet = "\n---\n".join(t[:300] for t in sms_texts[:10])
        mail_snippet = "\n---\n".join(t[:300] for t in mail_texts[:5])

        combined = ""
        if sms_snippet:
            combined += f"=== SMS ===\n{sms_snippet}\n"
        if mail_snippet:
            combined += f"=== EMAILS ===\n{mail_snippet}\n"

        if not combined.strip():
            continue
        langfuse_handler = CallbackHandler()

        # langfuse_handler = CallbackHandler(
        #     public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        #     secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        #     host=os.getenv("LANGFUSE_HOST", "https://challenges.reply.com/langfuse"),
        #     session_id=session_id,
        # )
        result = _call_llm_phishing(
            user_id, combined, phishing_model, langfuse_handler, session_id
        )
        llm_calls += 1
        results.append(result)

    return results, llm_calls
