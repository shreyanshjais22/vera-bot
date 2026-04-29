"""
main.py — Vera Bot HTTP server.
Exposes: GET /v1/healthz, GET /v1/metadata, POST /v1/context, POST /v1/tick, POST /v1/reply
"""

import os
import time
import uuid
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from composer import compose
from reply_handler import handle_reply

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("vera.main")

app = FastAPI(title="Vera Bot", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

START_TIME = time.time()

# ── In-memory state ──────────────────────────────────────────────────────────
# contexts[(scope, context_id)] = {"version": int, "payload": dict}
contexts: dict[tuple[str, str], dict] = {}

# conversations[conv_id] = [{"from": "vera|merchant|customer", "msg": str, "ts": str}]
conversations: dict[str, list] = {}

# suppressed suppression keys (ended conversations)
suppressed_keys: set[str] = set()

# conv_id → merchant_id mapping
conv_merchant_map: dict[str, str] = {}
conv_customer_map: dict[str, Optional[str]] = {}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _get_context(scope: str, ctx_id: str) -> Optional[dict]:
    entry = contexts.get((scope, ctx_id))
    return entry["payload"] if entry else None


def _count_contexts() -> dict:
    counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for (scope, _) in contexts:
        if scope in counts:
            counts[scope] += 1
    return counts


# ── Models ───────────────────────────────────────────────────────────────────

class CtxBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: dict[str, Any]
    delivered_at: str


class TickBody(BaseModel):
    now: str
    available_triggers: list[str] = []


class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: Optional[str] = None
    turn_number: int = 1


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/v1/healthz")
async def healthz():
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START_TIME),
        "contexts_loaded": _count_contexts(),
    }


@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": os.getenv("TEAM_NAME", "VeraBot"),
        "team_members": [os.getenv("TEAM_MEMBER", "Shreyansh")],
        "model": "gemini-2.0-flash",
        "approach": (
            "Trigger-routed LLM composer using Gemini 2.0 Flash. "
            "4-context prompt with trigger-kind hints, voice matching, and post-LLM validation. "
            "Rule-based fast paths for auto-reply detection, opt-out, intent transitions, and out-of-scope."
        ),
        "contact_email": os.getenv("CONTACT_EMAIL", ""),
        "version": "1.0.0",
        "submitted_at": "2026-04-29T14:00:00Z",
    }


@app.post("/v1/context")
async def push_context(body: CtxBody):
    if body.scope not in ("category", "merchant", "customer", "trigger"):
        return JSONResponse(
            status_code=400,
            content={"accepted": False, "reason": "invalid_scope", "details": f"Unknown scope: {body.scope}"},
        )

    key = (body.scope, body.context_id)
    current = contexts.get(key)

    if current:
        if current["version"] > body.version:
            return JSONResponse(
                status_code=409,
                content={"accepted": False, "reason": "stale_version", "current_version": current["version"]},
            )
        if current["version"] == body.version:
            # Idempotent — same version is a no-op
            return {
                "accepted": True,
                "ack_id": f"ack_{body.context_id}_v{body.version}_noop",
                "stored_at": _now_iso(),
            }

    # Store (new or version bump)
    contexts[key] = {"version": body.version, "payload": body.payload}
    logger.info(f"Stored context: scope={body.scope} id={body.context_id} v={body.version}")

    return {
        "accepted": True,
        "ack_id": f"ack_{body.context_id}_v{body.version}",
        "stored_at": _now_iso(),
    }


@app.post("/v1/tick")
async def tick(body: TickBody):
    actions = []

    for trg_id in body.available_triggers:
        trg = _get_context("trigger", trg_id)
        if not trg:
            logger.warning(f"Trigger not found: {trg_id}")
            continue

        # Check suppression
        supp_key = trg.get("suppression_key", "")
        if supp_key in suppressed_keys:
            logger.info(f"Trigger suppressed: {trg_id} (key={supp_key})")
            continue

        # Check expiry
        expires_at = trg.get("expires_at")
        if expires_at:
            try:
                exp_dt = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
                now_dt = datetime.now(timezone.utc)
                if now_dt > exp_dt:
                    logger.info(f"Trigger expired: {trg_id}")
                    continue
            except Exception:
                pass

        merchant_id = trg.get("merchant_id")
        customer_id = trg.get("customer_id")

        merchant = _get_context("merchant", merchant_id) if merchant_id else None
        if not merchant:
            logger.warning(f"Merchant context missing for trigger {trg_id}: merchant_id={merchant_id}")
            continue

        category_slug = merchant.get("category_slug", "")
        category = _get_context("category", category_slug)
        if not category:
            logger.warning(f"Category context missing: {category_slug}")
            continue

        customer = None
        if customer_id:
            customer = _get_context("customer", customer_id)

        # Generate conv_id — reuse if merchant already has open conversation for this trigger kind
        trigger_kind = trg.get("kind", "generic")
        if customer_id:
            conv_id = f"conv_{customer_id}_{trigger_kind}"
        else:
            conv_id = f"conv_{merchant_id}_{trigger_kind}_{trg.get('suppression_key', trg_id)[:20]}"

        # Don't re-initiate a suppressed conversation
        if conv_id in conversations:
            history = conversations[conv_id]
            if history and any(_is_ended(h) for h in history):
                continue

        try:
            result = compose(category, merchant, trg, customer)
        except Exception as e:
            logger.error(f"Compose failed for trigger {trg_id}: {e}")
            continue

        # Record in conversation
        conversations.setdefault(conv_id, []).append({
            "from": "vera",
            "msg": result["body"],
            "ts": _now_iso(),
        })
        conv_merchant_map[conv_id] = merchant_id
        conv_customer_map[conv_id] = customer_id

        # Suppress this key going forward
        suppressed_keys.add(supp_key)

        # Determine template params
        owner = merchant.get("identity", {}).get("owner_first_name", "")
        body_text = result["body"]
        # Split body into up to 3 template params
        parts = body_text[:100], body_text[100:200], body_text[200:320]
        template_params = [p for p in parts if p.strip()]

        action = {
            "conversation_id": conv_id,
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "send_as": result.get("send_as", "vera"),
            "trigger_id": trg_id,
            "template_name": f"vera_{trigger_kind}_v1",
            "template_params": [owner] + template_params,
            "body": result["body"],
            "cta": result.get("cta", "open_ended"),
            "suppression_key": result.get("suppression_key", supp_key),
            "rationale": result.get("rationale", ""),
        }
        actions.append(action)

        if len(actions) >= 20:
            break

    return {"actions": actions}


@app.post("/v1/reply")
async def reply(body: ReplyBody):
    conv_id = body.conversation_id
    merchant_id = body.merchant_id or conv_merchant_map.get(conv_id)
    customer_id = body.customer_id or conv_customer_map.get(conv_id)

    # Record incoming message
    conversations.setdefault(conv_id, []).append({
        "from": body.from_role,
        "msg": body.message,
        "ts": body.received_at or _now_iso(),
    })

    # Load contexts for reply handler
    merchant_ctx = _get_context("merchant", merchant_id) if merchant_id else None
    category_ctx = None
    if merchant_ctx:
        category_ctx = _get_context("category", merchant_ctx.get("category_slug", ""))

    history = conversations.get(conv_id, [])

    result = handle_reply(
        conversation_id=conv_id,
        merchant_id=merchant_id,
        customer_id=customer_id,
        from_role=body.from_role,
        message=body.message,
        turn_number=body.turn_number,
        conversation_history=history,
        merchant_ctx=merchant_ctx,
        category_ctx=category_ctx,
    )

    # Record Vera's reply if it's a send
    if result.get("action") == "send" and result.get("body"):
        conversations[conv_id].append({
            "from": "vera",
            "msg": result["body"],
            "ts": _now_iso(),
        })

    return result


@app.post("/v1/teardown")
async def teardown():
    """Optional endpoint — wipe state at end of test."""
    contexts.clear()
    conversations.clear()
    suppressed_keys.clear()
    conv_merchant_map.clear()
    conv_customer_map.clear()
    logger.info("State wiped via /v1/teardown")
    return {"status": "wiped"}


def _is_ended(turn: dict) -> bool:
    return turn.get("from") == "vera" and "closed" in turn.get("msg", "").lower()


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
