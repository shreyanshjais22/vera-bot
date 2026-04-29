"""
reply_handler.py — Multi-turn conversation handler for /v1/reply endpoint.
Handles: engaged replies, auto-replies, hard NO, curveballs, intent transitions.
"""

import os
import json
import re
import logging
from typing import Optional

from google import genai
from google.genai import types as genai_types
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("vera.reply")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

AUTO_REPLY_PHRASES = [
    "thank you for contacting",
    "our team will respond",
    "automated reply",
    "i am an automated",
    "this is an automated",
    "aapki jaankari ke liye",
    "main ek automated assistant",
    "bahut-bahut shukriya",
    "hamari team tak pahuncha",
    "aapki madad ke liye shukriya, lekin main ek automated",
]

OPT_OUT_PHRASES = [
    "stop messaging", "not interested", "don't message", "stop",
    "band karo", "mat bhejo", "nahi chahiye", "unsubscribe",
    "abusive", "useless", "bothering me", "spam",
]

INTENT_ACTION_PHRASES = [
    "let's do it", "lets do it", "ok go ahead", "go ahead", "yes do it",
    "karo", "kar do", "please do", "send it", "haan karo", "yes please send",
    "confirm", "chalega", "theek hai karo", "proceed",
]

OUT_OF_SCOPE = [
    "gst", "income tax", "legal", "court", "police", "loan", "insurance",
    "government", "visa", "passport", "emi",
]

REPLY_SYSTEM = """You are Vera, magicpin's merchant AI assistant. You're handling a reply from the merchant.

RULES:
1. Respond in ≤ 320 chars.
2. No URLs in body.
3. One CTA only.
4. Match merchant's language (Hindi-English mix if needed).
5. If merchant said YES/let's do it → execute the promised action immediately, don't re-qualify.
6. If out-of-scope question → decline politely in 1 line, redirect to the original topic.
7. If hostile/opt-out → action=end, no more messages.
8. If auto-reply detected → action=wait (4h first time, 24h second time).

Return ONLY valid JSON:
{
  "action": "send|wait|end",
  "body": "<message, only if action=send>",
  "cta": "<open_ended|binary_yes_no|binary_confirm_cancel|none>",
  "wait_seconds": <integer, only if action=wait>,
  "rationale": "<brief reasoning>"
}"""


def _is_auto_reply(message: str) -> bool:
    msg_lower = message.lower()
    return any(phrase in msg_lower for phrase in AUTO_REPLY_PHRASES)


def _is_opt_out(message: str) -> bool:
    msg_lower = message.lower()
    return any(phrase in msg_lower for phrase in OPT_OUT_PHRASES)


def _is_intent_action(message: str) -> bool:
    msg_lower = message.lower()
    return any(phrase in msg_lower for phrase in INTENT_ACTION_PHRASES)


def _is_out_of_scope(message: str) -> bool:
    msg_lower = message.lower()
    return any(phrase in msg_lower for phrase in OUT_OF_SCOPE)


def _count_auto_replies_in_history(history: list) -> int:
    """Count consecutive auto-replies at the end of conversation history."""
    count = 0
    for turn in reversed(history):
        if turn.get("from") in ("merchant", "customer") and _is_auto_reply(turn.get("msg", "")):
            count += 1
        else:
            break
    return count


def handle_reply(
    conversation_id: str,
    merchant_id: str,
    customer_id: Optional[str],
    from_role: str,
    message: str,
    turn_number: int,
    conversation_history: list,
    merchant_ctx: Optional[dict],
    category_ctx: Optional[dict],
) -> dict:
    """
    Handle incoming reply and produce next action.
    Returns: {action, body?, cta?, wait_seconds?, rationale}
    """
    # --- Rule-based fast paths ---

    # 1. Opt-out / hostile
    if _is_opt_out(message):
        return {
            "action": "end",
            "rationale": "Merchant explicitly opted out or expressed frustration. Closing conversation and suppressing future sends.",
        }

    # 2. Auto-reply detection
    if _is_auto_reply(message):
        prior_auto = _count_auto_replies_in_history(conversation_history)
        if prior_auto == 0:
            # First auto-reply — try once more
            return {
                "action": "send",
                "body": "Looks like an auto-reply 😊 When you're free, just reply 'Yes' to continue — I'll keep this warm.",
                "cta": "binary_yes_no",
                "rationale": "Detected auto-reply (canned phrasing). One prompt to flag it for the owner.",
            }
        elif prior_auto == 1:
            return {
                "action": "wait",
                "wait_seconds": 86400,
                "rationale": "Auto-reply twice in a row. Owner likely not at phone — backing off 24h.",
            }
        else:
            return {
                "action": "end",
                "rationale": "Auto-reply 3+ times. No real engagement signal. Closing conversation.",
            }

    # 3. Intent action — execute immediately, don't re-qualify
    if _is_intent_action(message):
        if GEMINI_API_KEY:
            return _llm_reply(message, conversation_history, merchant_ctx, category_ctx, mode="execute_intent")
        # Fallback: extract what we were working on from history
        last_vera = next((t["msg"] for t in reversed(conversation_history) if t.get("from") == "vera"), "")
        offer = ""
        if merchant_ctx:
            offers = [o["title"] for o in merchant_ctx.get("offers", []) if o.get("status") == "active"]
            offer = offers[0] if offers else ""
        name = merchant_ctx.get("identity", {}).get("owner_first_name", "") if merchant_ctx else ""
        body = f"Great, {name}! Drafting your campaign now — 90 seconds. "
        if offer:
            body += f"Pushing '{offer}' to your customer list. Reply CONFIRM to send."
        else:
            body += "I'll send you the draft shortly. Reply CONFIRM to proceed."
        body = body[:320]
        return {
            "action": "send",
            "body": body,
            "cta": "binary_confirm_cancel",
            "rationale": "Merchant committed to action; executing immediately without re-qualification.",
        }

    # 4. Out-of-scope redirect
    if _is_out_of_scope(message):
        last_topic = _get_last_vera_topic(conversation_history)
        body = f"I'll leave that to a specialist — outside my scope. Back to {last_topic} — want me to proceed?"
        if len(body) > 320:
            body = body[:317] + "..."
        return {
            "action": "send",
            "body": body,
            "cta": "binary_yes_no",
            "rationale": "Out-of-scope request politely declined. Redirected to original topic.",
        }

    # 5. LLM-powered response for everything else
    if GEMINI_API_KEY:
        return _llm_reply(message, conversation_history, merchant_ctx, category_ctx, mode="continue")

    # 6. Context-aware fallback
    name = merchant_ctx.get("identity", {}).get("owner_first_name", "") if merchant_ctx else ""
    offers = [o["title"] for o in merchant_ctx.get("offers", []) if o.get("status") == "active"] if merchant_ctx else []
    offer_hook = f" I'll use your '{offers[0]}' as the hook." if offers else ""
    body = f"Got it{', ' + name if name else ''}!{offer_hook} Drafting your response — reply CONFIRM to proceed."
    return {
        "action": "send",
        "body": body[:320],
        "cta": "binary_confirm_cancel",
        "rationale": "Acknowledged merchant reply; advancing conversation.",
    }


def _get_last_vera_topic(history: list) -> str:
    for turn in reversed(history):
        if turn.get("from") == "vera":
            msg = turn.get("msg", "")
            # Extract first meaningful phrase
            if msg:
                return msg[:60] + "..." if len(msg) > 60 else msg
    return "our previous topic"


def _llm_reply(
    message: str,
    history: list,
    merchant_ctx: Optional[dict],
    category_ctx: Optional[dict],
    mode: str = "continue",
) -> dict:
    """Use Gemini to generate the next reply in conversation."""
    try:
        merchant_summary = ""
        if merchant_ctx:
            name = merchant_ctx.get("identity", {}).get("owner_first_name", "")
            offers = [o["title"] for o in merchant_ctx.get("offers", []) if o.get("status") == "active"]
            signals = merchant_ctx.get("signals", [])
            merchant_summary = f"Merchant: {name}, Offers: {offers}, Signals: {signals}"

        history_str = "\n".join(
            f"[{t.get('from', 'unknown').upper()}]: {t.get('msg', '')}"
            for t in history[-6:]
        )
        mode_hint = ""
        if mode == "execute_intent":
            mode_hint = "\nMERCHANT SAID YES/LET'S DO IT — execute immediately, do NOT re-qualify."

        prompt = f"""{merchant_summary}
{mode_hint}

CONVERSATION SO FAR:
{history_str}

LATEST MESSAGE FROM MERCHANT/CUSTOMER: "{message}"

Generate the next Vera response."""

        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                system_instruction=REPLY_SYSTEM,
                temperature=0.0,
                max_output_tokens=400,
                response_mime_type="application/json",
            ),
        )
        text = response.text.strip()
        if "```" in text:
            text = re.sub(r"```[a-z]*\n?", "", text).replace("```", "").strip()

        result = json.loads(text)

        # Validate
        if result.get("action") == "send":
            body = result.get("body", "")
            body = re.sub(r'https?://\S+', '', body).strip()
            if len(body) > 320:
                body = body[:317] + "..."
            result["body"] = body
            if result.get("cta") not in {"open_ended", "binary_yes_no", "binary_confirm_cancel", "none", "multi_choice_slot"}:
                result["cta"] = "open_ended"
        elif result.get("action") == "wait":
            result.setdefault("wait_seconds", 3600)
        elif result.get("action") == "end":
            pass
        else:
            result["action"] = "send"
            result["body"] = "Got it — working on it now. I'll send you the draft shortly."
            result["cta"] = "open_ended"

        return result

    except Exception as e:
        logger.error(f"LLM reply failed: {e}")
        # For intent-execution mode, still return binary_confirm_cancel
        if mode == "execute_intent":
            name = (merchant_ctx or {}).get("identity", {}).get("owner_first_name", "")
            offers = [(merchant_ctx or {}).get("offers", [])]
            flat_offers = [o["title"] for o in merchant_ctx.get("offers", []) if o.get("status") == "active"] if merchant_ctx else []
            offer_hook = f" Using your '{flat_offers[0]}' as the hook." if flat_offers else ""
            body = f"Great{', ' + name if name else ''}!{offer_hook} Drafting now — reply CONFIRM to send."
            return {
                "action": "send",
                "body": body[:320],
                "cta": "binary_confirm_cancel",
                "rationale": f"Intent execution; LLM fallback (error={str(e)[:40]})",
            }
        return {
            "action": "send",
            "body": "Got it! Working on your request now — will send the draft shortly.",
            "cta": "open_ended",
            "rationale": f"LLM fallback; error={str(e)[:50]}",
        }
