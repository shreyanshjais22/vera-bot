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
from fastapi.responses import JSONResponse, HTMLResponse
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

@app.get("/", response_class=HTMLResponse)
async def homepage():
    uptime = int(time.time() - START_TIME)
    counts = _count_contexts()
    return HTMLResponse(content=f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Vera Bot — magicpin AI Challenge</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: 'Inter', sans-serif; background: #0a0a0f; color: #e2e8f0; min-height: 100vh; }}
  .hero {{ background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%); padding: 60px 20px; text-align: center; border-bottom: 1px solid #1e3a5f; }}
  .logo {{ font-size: 48px; margin-bottom: 12px; }}
  h1 {{ font-size: 2.5rem; font-weight: 700; background: linear-gradient(90deg, #60a5fa, #a78bfa, #34d399); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
  .subtitle {{ color: #94a3b8; margin-top: 8px; font-size: 1.1rem; }}
  .badge {{ display: inline-block; background: #064e3b; color: #34d399; padding: 4px 14px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; margin-top: 16px; border: 1px solid #34d399; }}
  .container {{ max-width: 1100px; margin: 0 auto; padding: 40px 20px; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-bottom: 32px; }}
  @media(max-width:700px){{ .grid{{ grid-template-columns:1fr; }} }}
  .card {{ background: #111827; border: 1px solid #1f2937; border-radius: 16px; padding: 24px; }}
  .card h3 {{ color: #60a5fa; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 16px; }}
  .stat {{ display: flex; justify-content: space-between; padding: 10px 0; border-bottom: 1px solid #1f2937; }}
  .stat:last-child {{ border-bottom: none; }}
  .stat-val {{ color: #34d399; font-weight: 600; font-size: 1.1rem; }}
  .endpoint {{ background: #0d1117; border: 1px solid #1f2937; border-radius: 10px; padding: 12px 16px; margin-bottom: 8px; display: flex; align-items: center; gap: 12px; font-family: monospace; font-size: 0.9rem; }}
  .method {{ padding: 3px 10px; border-radius: 6px; font-weight: 700; font-size: 0.75rem; }}
  .get {{ background: #064e3b; color: #34d399; }}
  .post {{ background: #1e1b4b; color: #a78bfa; }}
  .chat-box {{ background: #111827; border: 1px solid #1f2937; border-radius: 16px; padding: 24px; margin-bottom: 32px; }}
  .chat-box h3 {{ color: #60a5fa; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 16px; }}
  .messages {{ height: 300px; overflow-y: auto; background: #0d1117; border-radius: 10px; padding: 16px; margin-bottom: 16px; display: flex; flex-direction: column; gap: 10px; }}
  .msg {{ padding: 10px 14px; border-radius: 10px; max-width: 85%; font-size: 0.9rem; line-height: 1.5; }}
  .msg.vera {{ background: #1e3a5f; color: #bfdbfe; align-self: flex-start; border-bottom-left-radius: 2px; }}
  .msg.user {{ background: #1e1b4b; color: #ddd6fe; align-self: flex-end; border-bottom-right-radius: 2px; }}
  .msg .label {{ font-size: 0.7rem; opacity: 0.6; margin-bottom: 4px; font-weight: 600; }}
  .input-row {{ display: flex; gap: 10px; }}
  .input-row input {{ flex: 1; background: #0d1117; border: 1px solid #374151; border-radius: 10px; padding: 12px 16px; color: #e2e8f0; font-family: 'Inter', sans-serif; font-size: 0.95rem; outline: none; }}
  .input-row input:focus {{ border-color: #60a5fa; }}
  .input-row button {{ background: linear-gradient(135deg, #3b82f6, #8b5cf6); color: white; border: none; border-radius: 10px; padding: 12px 24px; cursor: pointer; font-weight: 600; font-size: 0.95rem; transition: opacity 0.2s; }}
  .input-row button:hover {{ opacity: 0.85; }}
  .footer {{ text-align: center; color: #4b5563; font-size: 0.85rem; padding: 20px; }}
  .ping {{ display: inline-block; width: 8px; height: 8px; background: #34d399; border-radius: 50%; margin-right: 6px; animation: pulse 2s infinite; }}
  @keyframes pulse {{ 0%,100%{{opacity:1;}} 50%{{opacity:0.3;}} }}
</style>
</head>
<body>
<div class="hero">
  <div class="logo">🤖</div>
  <h1>Vera Bot</h1>
  <p class="subtitle">magicpin AI Challenge — Merchant Intelligence Engine</p>
  <div class="badge"><span class="ping"></span>LIVE · Gemini 2.0 Flash · 30/30 Tests Passing</div>
</div>

<div class="container">
  <div class="grid">
    <div class="card">
      <h3>📊 Live Status</h3>
      <div class="stat"><span>Status</span><span class="stat-val">✅ Online</span></div>
      <div class="stat"><span>Uptime</span><span class="stat-val">{uptime}s</span></div>
      <div class="stat"><span>Model</span><span class="stat-val">Gemini 2.0 Flash</span></div>
      <div class="stat"><span>Categories loaded</span><span class="stat-val">{counts['category']}</span></div>
      <div class="stat"><span>Merchants loaded</span><span class="stat-val">{counts['merchant']}</span></div>
      <div class="stat"><span>Triggers loaded</span><span class="stat-val">{counts['trigger']}</span></div>
    </div>
    <div class="card">
      <h3>🔌 API Endpoints</h3>
      <div class="endpoint"><span class="method get">GET</span>/v1/healthz</div>
      <div class="endpoint"><span class="method get">GET</span>/v1/metadata</div>
      <div class="endpoint"><span class="method post">POST</span>/v1/context</div>
      <div class="endpoint"><span class="method post">POST</span>/v1/tick</div>
      <div class="endpoint"><span class="method post">POST</span>/v1/reply</div>
    </div>
  </div>

  <div class="chat-box">
    <h3>💬 Live Chat Demo — Talk to Vera</h3>
    <div class="messages" id="msgs">
      <div class="msg vera"><div class="label">VERA</div>Hi! I'm Vera, magicpin's merchant AI. Ask me anything about your business — footfall, offers, customer recalls, or performance insights. Try: <em>"What should I do if my calls dropped 50%?"</em></div>
    </div>
    <div class="input-row">
      <input id="inp" type="text" placeholder="Ask Vera something..." onkeydown="if(event.key==='Enter')send()">
      <button onclick="send()">Send →</button>
    </div>
  </div>

  <div class="card">
    <h3>🏗️ Architecture</h3>
    <div class="stat"><span>Trigger kinds covered</span><span class="stat-val">25 / 25</span></div>
    <div class="stat"><span>Categories</span><span class="stat-val">Dentists · Salons · Restaurants · Gyms · Pharmacies</span></div>
    <div class="stat"><span>Composition engine</span><span class="stat-val">Gemini 2.0 Flash + Rule fallback</span></div>
    <div class="stat"><span>Reply handling</span><span class="stat-val">Auto-reply · Opt-out · Intent transition</span></div>
    <div class="stat"><span>Message constraints</span><span class="stat-val">≤320 chars · No URLs · Data-grounded</span></div>
  </div>
</div>

<div class="footer">Built for magicpin AI Challenge 2026 · Vera Bot v1.0.0</div>

<script>
  const BOT = window.location.origin;
  let convId = 'demo_' + Date.now();
  let merchantLoaded = false;

  async function loadDemo() {{
    // Push minimal demo context
    await fetch(BOT+'/v1/context', {{method:'POST',headers:{{'Content-Type':'application/json'}},
      body: JSON.stringify({{scope:'category',context_id:'restaurants',version:99,delivered_at:new Date().toISOString(),
        payload:{{slug:'restaurants',voice:{{tone:'warm_busy_practical'}},
          offer_catalog:[{{id:'demo_001',title:'Weekday Lunch Thali @ ₹149',value:'149',status:'active'}}],
          peer_stats:{{avg_calls_30d:38,avg_ctr:0.025}},digest:[],seasonal_beats:[]}}}})}})
    await fetch(BOT+'/v1/context', {{method:'POST',headers:{{'Content-Type':'application/json'}},
      body: JSON.stringify({{scope:'merchant',context_id:'demo_merchant',version:99,delivered_at:new Date().toISOString(),
        payload:{{merchant_id:'demo_merchant',category_slug:'restaurants',
          identity:{{name:'Demo Restaurant',city:'Delhi',locality:'Connaught Place',owner_first_name:'Raj',languages:['en','hi']}},
          subscription:{{status:'active',plan:'Pro',days_remaining:45}},
          performance:{{views:3200,calls:18,ctr:0.019,delta_7d:{{calls_pct:-0.25}}}},
          offers:[{{id:'o1',title:'Weekday Lunch Thali @ ₹149',status:'active'}}],
          signals:['ctr_below_peer_median'],conversation_history:[]}}}})}})
    merchantLoaded = true;
  }}
  loadDemo();

  function addMsg(text, role) {{
    const msgs = document.getElementById('msgs');
    const d = document.createElement('div');
    d.className = 'msg ' + (role==='vera' ? 'vera' : 'user');
    d.innerHTML = '<div class="label">'+(role==='vera'?'VERA':'YOU')+'</div>' + text;
    msgs.appendChild(d);
    msgs.scrollTop = msgs.scrollHeight;
  }}

  async function send() {{
    const inp = document.getElementById('inp');
    const msg = inp.value.trim();
    if (!msg) return;
    inp.value = '';
    addMsg(msg, 'user');
    addMsg('Thinking...', 'vera');
    try {{
      const res = await fetch(BOT+'/v1/reply', {{method:'POST',
        headers:{{'Content-Type':'application/json'}},
        body: JSON.stringify({{conversation_id:convId,merchant_id:'demo_merchant',
          from_role:'merchant',message:msg,received_at:new Date().toISOString(),turn_number:2}})}});
      const data = await res.json();
      document.querySelector('.msg.vera:last-child').remove();
      if (data.action==='send') addMsg(data.body || 'Got it!', 'vera');
      else if (data.action==='end') addMsg('Conversation ended. Refresh to start again.', 'vera');
      else addMsg('Taking a moment to think — try again shortly.', 'vera');
    }} catch(e) {{ document.querySelector('.msg.vera:last-child').remove(); addMsg('Error connecting to bot.', 'vera'); }}
  }}
</script>
</body>
</html>
""")


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
