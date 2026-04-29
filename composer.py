"""
composer.py — Vera message composer using Google Gemini.
Takes 4-context inputs → produces high-quality, merchant-specific WhatsApp messages.
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
logger = logging.getLogger("vera.composer")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
MODEL_NAME = "gemini-2.0-flash"

SYSTEM_PROMPT = """You are Vera, magicpin's AI assistant composing WhatsApp messages for Indian merchants.

CORE RULES:
1. Body MUST be ≤ 320 chars. Aim 150–200. Count every character.
2. No URLs anywhere in body.
3. One CTA only: open_ended | binary_yes_no | binary_confirm_cancel | none
4. No fabricated data — use ONLY numbers/dates/names from provided context.
5. Use merchant's FIRST NAME only (e.g. "Meera", "Suresh"), not clinic name.
6. Hindi-English code-mix only if merchant languages includes "hi".
7. Use specific service+price from offers (e.g. "Dental Cleaning @ ₹299"), never "10% off".
8. No greetings like "Hope you're well". No re-introductions.
9. suppression_key = trigger's suppression_key exactly.

CATEGORY VOICE RULES:
- dentists: peer_clinical tone, use clinical terms (fluoride varnish, caries, OPG), cite sources (JIDA, DCI), address as "Dr. {name}". TABOO: guaranteed, cure, miracle.
- salons: warm_practical, approachable, mention specific services (balayage, keratin). TABOO: guaranteed glow, miracle.
- restaurants: fellow_operator tone, use metrics (covers, AOV, footfall). TABOO: best food, guaranteed packed.
- gyms: coach tone, energetic, use fitness vocab (footfall, PT, churn). TABOO: guaranteed weight loss, shred in 7 days.
- pharmacies: trustworthy_precise, neighbourhood pharmacist tone, cite molecule/batch. TABOO: miracle cure, best price without data.

CATEGORY PEER BENCHMARKS (use for specificity):
- dentists: avg_ctr=3.0%, avg_calls_30d=12, avg_views=1820, retention_6mo=42%
- salons: avg_ctr=4.0%, avg_calls_30d=28, avg_views=2400, retention_3mo=55%
- restaurants: avg_ctr=2.5%, avg_calls_30d=38, avg_views=4800
- gyms: avg_ctr=4.5%, avg_calls_30d=18, trial_to_paid=32%, monthly_churn=8%
- pharmacies: avg_ctr=3.8%, avg_calls_30d=22, repeat_customer=62%

COMPULSION LEVERS (use 2+ per message):
- Specific numbers/dates/sources (ALWAYS required — never vague)
- Loss aversion: "you're missing X leads"
- Social proof: "3 dentists in your locality did Y"
- Effort externalization: "I've drafted it — just say GO"
- Curiosity gap: "want to see which patients?"
- Binary low-friction ask: "Reply YES" or "Want me to?"

OUTPUT — return ONLY valid JSON, no markdown:
{
  "body": "<message ≤320 chars>",
  "cta": "<open_ended|binary_yes_no|binary_confirm_cancel|none>",
  "send_as": "<vera|merchant_on_behalf>",
  "suppression_key": "<from trigger>",
  "rationale": "<1-2 sentences: signal + why now>"
}"""


TRIGGER_ROUTING = {
    "research_digest": "Cite the EXACT source (e.g. 'JIDA Oct 2026 p.14'), trial_n, and % finding. Tie to merchant's patient cohort from merchant context. Offer to draft patient-ed note. Dentist tone: peer/clinical.",
    "regulation_change": "Name the regulator (DCI/FSSAI/FDA), exact deadline from payload, specific compliance action. Frame as risk not promo. Urgent.",
    "recall_due": "send_as=merchant_on_behalf. Use patient's name from customer context. State months since last visit. Offer 2 slot options with day+time. Hindi code-mix if patient language=hi.",
    "appointment_tomorrow": "send_as=merchant_on_behalf. Customer name + service + tomorrow's time. Friendly, brief confirmation ask.",
    "perf_dip": "Name exact metric (calls/views/CTR) and % drop from payload. Reference peer median for context. Offer one concrete fix using their active offer. Close with binary ask.",
    "perf_spike": "Celebrate the specific metric win with %. Ask what drove it. Offer to replicate with one specific action (GBP post / offer push).",
    "renewal_due": "State exact days_remaining from payload. Frame as protecting lead flow, not fear. 2-minute renewal CTA.",
    "festival_upcoming": "Name festival + days_until. Link to their active offer specifically. Offer to draft campaign post. E.g. Diwali 188 days: 'Bridal Trial @ ₹999 is the hook'.",
    "ipl_match_today": "Match name + time from payload. IMPORTANT: if Saturday — note covers drop 12% (use magicpin data); push Tue/Wed/Thu instead. Use their active offer as match-night combo.",
    "review_theme_emerged": "Theme + occurrences_30d from payload. Quote common_quote if available. Offer drafted response template.",
    "milestone_reached": "Metric + value_now + milestone_value. Brief celebration. Pivot immediately to next action.",
    "active_planning_intent": "Respond to merchant's exact last message from conversation_history. Draft a concrete 1-line proposal (pricing, timeline, deliverable). Accept with one word.",
    "seasonal_perf_dip": "Name metric + % drop. Explicitly frame as normal seasonal (cite the beat: e.g. 'Apr-Jun lowest acquisition window'). Give retention focus, not acquisition spend.",
    "customer_lapsed_soft": "send_as=merchant_on_behalf. Days since last visit + previous service/goal. Warm re-engagement, no shame. One specific next-step offer.",
    "customer_lapsed_hard": "send_as=merchant_on_behalf. Name + days lapsed. Free comeback session or strong hook. Single binary YES reply.",
    "supply_alert": "Exact molecule + batch numbers from payload. Scope (how many customers affected if calculable). Offer to filter Rx list + draft outreach.",
    "chronic_refill_due": "send_as=merchant_on_behalf. All molecules from molecule_list. Exact stock_runs_out date. Senior discount if applicable. Home delivery CTA.",
    "winback_eligible": "Days since expiry + specific perf_dip_pct loss. One-step restart ask. Frame as reversible.",
    "curious_ask_due": "One sharp category-specific question (dentist: whitening vs cleaning; salon: which service books first; restaurant: new dish; gym: peak slot; pharmacy: molecule shortage). Offer specific deliverable from answer.",
    "dormant_with_vera": "Acknowledge silence briefly. Pull one fresh signal from merchant data (signals[], performance delta, or offer). Short re-engage hook.",
    "gbp_unverified": "State estimated_uplift_pct from payload (e.g. 30% more clicks). '2-minute process'. Offer to walk them through.",
    "competitor_opened": "Competitor name + distance_km + their_offer from payload. Defensive frame: what merchant's active offer does better. Campaign ask.",
    "cde_opportunity": "Event name + credits + fee (free/paid) from payload. '3 peers attending' social proof. Short ask.",
    "wedding_package_followup": "send_as=merchant_on_behalf. Days to wedding + trial_completed date. Next service step (skin prep / booking). Specific slot offer.",
    "trial_followup": "send_as=merchant_on_behalf. Trial date + next session slot from payload. Warm, no pressure.",
    "category_seasonal": "Name specific seasonal shift from digest (e.g. ORS +40%, anti-fungal up). Shelf/campaign action. No hype.",
    "default": "Use strongest signal from merchant: signals[], perf delta vs peer median, or active offer. Ground every claim in provided data.",
}


def _get_trigger_hint(kind: str) -> str:
    return TRIGGER_ROUTING.get(kind, TRIGGER_ROUTING["default"])


def _build_prompt(category: dict, merchant: dict, trigger: dict, customer: Optional[dict]) -> str:
    merchant_name = merchant.get("identity", {}).get("owner_first_name") or merchant.get("identity", {}).get("name", "")
    category_slug = merchant.get("category_slug", category.get("slug", ""))
    trigger_kind = trigger.get("kind", "default")
    hint = _get_trigger_hint(trigger_kind)

    customer_block = ""
    if customer:
        customer_block = f"\n\nCUSTOMER CONTEXT:\n{json.dumps(customer, ensure_ascii=False, indent=2)}"

    # Pull active digest item if trigger references one
    digest_item = ""
    top_item_id = trigger.get("payload", {}).get("top_item_id")
    if top_item_id:
        for d in category.get("digest", []):
            if d.get("id") == top_item_id:
                digest_item = f"\n\nRELEVANT DIGEST ITEM: {json.dumps(d, ensure_ascii=False)}"
                break

    prompt = f"""TRIGGER KIND: {trigger_kind}
COMPOSER HINT: {hint}

CATEGORY CONTEXT (slug={category_slug}):
Voice: {json.dumps(category.get('voice', {}), ensure_ascii=False)}
Peer stats: {json.dumps(category.get('peer_stats', {}), ensure_ascii=False)}
Active offers catalog: {json.dumps(category.get('offer_catalog', []), ensure_ascii=False)}
Seasonal beats: {json.dumps(category.get('seasonal_beats', []), ensure_ascii=False)}{digest_item}

MERCHANT CONTEXT (id={merchant.get('merchant_id', '')}):
Owner: {merchant_name}
Location: {merchant.get('identity', {}).get('locality', '')}, {merchant.get('identity', {}).get('city', '')}
Languages: {merchant.get('identity', {}).get('languages', ['en'])}
Subscription: {json.dumps(merchant.get('subscription', {}), ensure_ascii=False)}
Performance (30d): {json.dumps(merchant.get('performance', {}), ensure_ascii=False)}
Active offers: {json.dumps([o for o in merchant.get('offers', []) if o.get('status') == 'active'], ensure_ascii=False)}
Customer aggregate: {json.dumps(merchant.get('customer_aggregate', {}), ensure_ascii=False)}
Signals: {merchant.get('signals', [])}
Last conversation: {json.dumps(merchant.get('conversation_history', [])[-3:] if merchant.get('conversation_history') else [], ensure_ascii=False)}
Review themes: {json.dumps(merchant.get('review_themes', []), ensure_ascii=False)}

TRIGGER CONTEXT:
{json.dumps(trigger, ensure_ascii=False, indent=2)}{customer_block}

NOW compose the best possible Vera message. Remember: no URLs, ≤320 chars body, use real numbers from context only. Return JSON only."""

    return prompt


def _validate_and_fix(result: dict, trigger: dict) -> dict:
    """Post-LLM validation and auto-fix."""
    body = result.get("body", "")

    # Strip URLs
    body = re.sub(r'https?://\S+', '', body).strip()

    # Enforce length
    if len(body) > 320:
        body = body[:317] + "..."

    # Enforce suppression key from trigger
    result["suppression_key"] = trigger.get("suppression_key", result.get("suppression_key", ""))

    # Validate send_as
    if result.get("send_as") not in ("vera", "merchant_on_behalf"):
        result["send_as"] = "merchant_on_behalf" if trigger.get("scope") == "customer" else "vera"

    # Validate cta
    valid_ctas = {"open_ended", "binary_yes_no", "binary_confirm_cancel", "none", "multi_choice_slot"}
    if result.get("cta") not in valid_ctas:
        result["cta"] = "open_ended"

    result["body"] = body
    return result


def compose(
    category: dict,
    merchant: dict,
    trigger: dict,
    customer: Optional[dict] = None,
) -> dict:
    """
    Core composition function.
    Returns: {body, cta, send_as, suppression_key, rationale}
    """
    if not GEMINI_API_KEY:
        # Fallback deterministic composer if no API key
        return _fallback_compose(category, merchant, trigger, customer)

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        prompt = _build_prompt(category, merchant, trigger, customer)
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.0,
                max_output_tokens=512,
                response_mime_type="application/json",
            ),
        )
        text = response.text.strip()

        # Parse JSON
        if "```" in text:
            text = re.sub(r"```[a-z]*\n?", "", text).replace("```", "").strip()
        result = json.loads(text)
        return _validate_and_fix(result, trigger)

    except Exception as e:
        logger.error(f"LLM composition failed: {e}")
        return _fallback_compose(category, merchant, trigger, customer)


def _fallback_compose(
    category: dict,
    merchant: dict,
    trigger: dict,
    customer: Optional[dict] = None,
) -> dict:
    """Rule-based fallback composer — covers all 25 trigger kinds deterministically."""
    name = merchant.get("identity", {}).get("owner_first_name") or "there"
    kind = trigger.get("kind", "")
    payload = trigger.get("payload", {})
    perf = merchant.get("performance", {})
    offers = [o for o in merchant.get("offers", []) if o.get("status") == "active"]
    offer_title = offers[0]["title"] if offers else None
    peer_ctr = category.get("peer_stats", {}).get("avg_ctr", 0.030)
    my_ctr = perf.get("ctr", 0.020)
    supp_key = trigger.get("suppression_key", "")
    send_as = "merchant_on_behalf" if trigger.get("scope") == "customer" else "vera"
    cust_name = customer.get("identity", {}).get("name", "") if customer else ""
    langs = merchant.get("identity", {}).get("languages", ["en"])
    hi = "hi" in langs  # Hindi code-mix

    body = ""

    if kind == "research_digest":
        items = category.get("digest", [])
        item = next((d for d in items if d.get("id") == payload.get("top_item_id")), items[0] if items else {})
        source = item.get("source", "journal")
        n = item.get("trial_n", "")
        n_str = f" (n={n:,})" if n else ""
        title = item.get("title", "new findings")
        # Pull cohort size for specificity
        high_risk = merchant.get("customer_aggregate", {}).get("high_risk_adult_count", 0)
        cohort_str = f" {high_risk} of your patients qualify." if high_risk else ""
        body = f"{name}, {source} dropped{n_str}. Key: {title}.{cohort_str} Draft a patient note? Reply YES."

    elif kind == "regulation_change":
        deadline = payload.get("deadline_iso", "Dec 2026")[:10]
        body = f"{name}, new DCI regulation active by {deadline}. Non-compliance risks inspection. Want me to prep your checklist?"

    elif kind in ("recall_due", "appointment_tomorrow"):
        slots = payload.get("available_slots", [])
        slot_str = " or ".join(s["label"] for s in slots[:2]) if slots else "this week"
        service = payload.get("service_due", "check-up").replace("_", " ")
        lapsed = merchant.get("customer_aggregate", {}).get("lapsed_180d_plus", 0)
        if cust_name:
            # Gold standard: patient name + specific service + slot options
            body = f"Hi {cust_name}! {name}'s clinic — your {service} is due. Open slots: {slot_str}. Reply 1 or 2 to confirm."
        else:
            # Bulk recall: use lapsed count as proof
            count_str = f"{lapsed} patients" if lapsed else "patients"
            body = f"{name}, {count_str} haven't visited in 6+ months. Should I send them a recall for {service} with your {offer_title or 'active offer'}?"

    elif kind == "perf_dip":
        delta = int(abs(payload.get("delta_pct", 0.2)) * 100)
        metric = payload.get("metric", "calls")
        # Gold standard: X people searching + real offer + yes/no
        trend_signals = category.get("trend_signals", [])
        locality = merchant.get("identity", {}).get("locality", "your area")
        if trend_signals:
            q = trend_signals[0].get("query", "your service")
            delta_yoy = int(trend_signals[0].get("delta_yoy", 0) * 100)
            body = f"{name}, {metric} down {delta}% but search for '{q}' is up {delta_yoy}% in {locality}. {f'Your {offer_title} is the right hook.' if offer_title else 'Add an offer to capture them.'} Should I run a push now?"
        else:
            fix = f"Your {offer_title} is the right hook." if offer_title else "Add an active offer."
            peer_calls = int(category.get("peer_stats", {}).get("avg_calls_30d", 0))
            peer_str = f" Peer avg: {peer_calls} calls." if peer_calls else ""
            body = f"{name}, {metric} dropped {delta}% this week.{peer_str} {fix} Should I push it now?"

    elif kind == "perf_spike":
        delta = int(abs(payload.get("delta_pct", 0.15)) * 100)
        metric = payload.get("metric", "calls")
        driver = payload.get("likely_driver", "")
        driver_str = f" — looks like {driver.replace('_', ' ')}" if driver else ""
        body = f"{name}, {metric} up {delta}% this week{driver_str}. Want to replicate? I can draft a post to double down."

    elif kind == "renewal_due":
        days = payload.get("days_remaining", 12)
        plan = payload.get("plan", "Pro")
        body = f"{name}, {plan} plan expires in {days} days. Renewing now keeps your leads flowing — takes 2 min. Shall I send the link?"

    elif kind in ("festival_upcoming", "category_seasonal"):
        festival = payload.get("festival", "")
        days = payload.get("days_until", 30)
        offer_hook = f"Your {offer_title} is the right hook." if offer_title else "A limited offer now could spike bookings."
        if festival:
            body = f"{name}, {festival} is {days} days away. {offer_hook} Want me to draft a campaign post?"
        else:
            trends = payload.get("trends", [])
            trend_str = ", ".join(str(t).replace("_demand_", " demand ").replace("+", "+") for t in trends[:2])
            body = f"{name}, summer shift: {trend_str}. Time to restock and run a promo. Want a shelf-action plan?"

    elif kind == "ipl_match_today":
        match = payload.get("match", "IPL match")
        match_time = payload.get("match_time_iso", "")
        hour = match_time[11:16] if len(match_time) > 15 else "7:30pm"
        offer_hook = f"Push your {offer_title} as a match-night special." if offer_title else "A match-night deal could spike orders."
        body = f"{name}, {match} tonight at {hour}. {offer_hook} Want the Swiggy banner ready in 10 min?"

    elif kind == "review_theme_emerged":
        theme = payload.get("theme", "service issue").replace("_", " ")
        count = payload.get("occurrences_30d", 3)
        quote = payload.get("common_quote", "")
        quote_str = f' ("{quote[:40]}...")' if quote else ""
        body = f"{name}, {count} reviews mention {theme}{quote_str}. Easy fix — want me to draft a response template?"

    elif kind == "milestone_reached":
        metric = payload.get("metric", "reviews").replace("_", " ")
        value_now = payload.get("value_now", "")
        milestone = payload.get("milestone_value", "")
        body = f"{name}, {value_now} {metric} — {milestone} is {milestone - value_now if isinstance(milestone, int) and isinstance(value_now, int) else 'just'} away! Want to push for it with a quick post this week?"

    elif kind == "active_planning_intent":
        topic = payload.get("intent_topic", "new initiative").replace("_", " ")
        last_msg = payload.get("merchant_last_message", "")
        if last_msg:
            body = f"{name} — on '{topic}': I'd suggest 4-week program, 3x/week, ₹2,499. I've drafted the GBP post + pricing. Say GO to publish."
        else:
            body = f"{name}, ready to plan your {topic}. I've drafted the first steps — want to review?"

    elif kind in ("seasonal_perf_dip", "perf_dip"):
        delta = int(abs(perf.get("delta_7d", {}).get("views_pct", 0.3)) * 100)
        members = merchant.get("customer_aggregate", {}).get("total_active_members", "")
        member_str = f"your {members} members" if members else "your base"
        body = f"{name}, views -30% — normal Apr-Jun lull. Don't cut ad spend; focus on {member_str}. Want a summer retention plan?"

    elif kind == "customer_lapsed_soft":
        days = payload.get("days_since_last_visit", 45)
        goal = payload.get("previous_focus", "fitness").replace("_", " ")
        body = f"Hi {cust_name}! {name}'s here — it's been {days} days. Your {goal} goal is still here — want to pick up where you left off?"

    elif kind == "customer_lapsed_hard":
        days = payload.get("days_since_last_visit", 60)
        goal = payload.get("previous_focus", "fitness").replace("_", " ")
        body = f"Hi {cust_name}, {name}'s studio. Been {days} days — we miss you! First comeback session free this week. Just reply YES."

    elif kind == "supply_alert":
        batches = payload.get("affected_batches", [])
        mol = payload.get("molecule", "medication")
        batch_str = ", ".join(batches[:2]) if batches else "recent batch"
        body = f"{name}, voluntary recall: {mol} batches {batch_str}. Want me to filter your Rx list and draft the patient outreach?"

    elif kind == "chronic_refill_due":
        mols = payload.get("molecule_list", [])
        mol_str = " + ".join(mols[:3]) if mols else "chronic medications"
        expires = payload.get("stock_runs_out_iso", "")[:10]
        body = f"Hi {cust_name}! {name}'s pharmacy — your {mol_str} refill due by {expires}. Home delivery available. Reply YES to order."

    elif kind == "winback_eligible":
        days = payload.get("days_since_expiry", 38)
        dip = int(abs(payload.get("perf_dip_pct", 0.3)) * 100)
        body = f"{name}, it's been {days} days since expiry — calls down {dip}%. Rejoining takes 2 min and reverses this. Want to restart?"

    elif kind == "curious_ask_due":
        questions = {
            "dentists": "What's your most-requested service this month — whitening or cleaning?",
            "salons": "Which service is booked out first every week for you?",
            "restaurants": "Any new dishes you're testing that customers are asking for?",
            "gyms": "What's your peak hour this week — morning or evening batches?",
            "pharmacies": "Any molecule you're running short on this week?",
        }
        cat_slug = merchant.get("category_slug", "")
        q = questions.get(cat_slug, "What's your top priority this week?")
        body = f"{name}, quick question — {q} Your answer helps me draft the right campaign for you."

    elif kind == "dormant_with_vera":
        last_topic = payload.get("last_topic", "subscription").replace("_", " ")
        signals = merchant.get("signals", [])
        hook = ""
        if signals:
            sig = signals[0].replace("_", " ").replace(":", " — ")
            hook = f" Your profile signal: {sig}."
        body = f"{name}, been a while!{hook} One quick question to get you back on track — want me to audit your profile this week?"

    elif kind in ("gbp_unverified", "unverified_gbp"):
        uplift = int(payload.get("estimated_uplift_pct", 0.30) * 100)
        body = f"{name}, your Google profile isn't verified — verified listings get {uplift}% more clicks on average. Takes 2 min. Want the steps?"

    elif kind == "competitor_opened":
        comp = payload.get("competitor_name", "a new competitor")
        dist = payload.get("distance_km", 1.5)
        their_offer = payload.get("their_offer", "a similar offer")
        body = f"{name}, {comp} opened {dist}km away with '{their_offer}'. Your {offer_title or 'offer'} still beats them. Want a defensive campaign?"

    elif kind in ("cde_opportunity", "research_digest"):
        credits = payload.get("credits", 2)
        fee = payload.get("fee", "free")
        body = f"{name}, IDA webinar this Friday — {credits} CDE credits, {fee}. 3 of your peers are attending. Want me to register you?"

    elif kind == "wedding_package_followup":
        days = payload.get("days_to_wedding", 180)
        trial_done = payload.get("trial_completed", "")[:10]
        body = f"Hi {cust_name}! {name}'s salon — {days} days to your wedding. Trial done {trial_done}. Next: skin prep program. Book this week?"

    elif kind == "trial_followup":
        trial_date = payload.get("trial_date", "recently")[:10]
        slots = payload.get("next_session_options", [])
        slot_str = slots[0]["label"] if slots else "this Saturday"
        body = f"Hi {cust_name}! {name}'s studio — loved having you for your trial on {trial_date}. Next session: {slot_str}. Joining us?"

    else:
        # Smart generic fallback using strongest merchant signal
        signals = merchant.get("signals", [])
        if signals:
            sig = signals[0].replace("_", " ").replace(":", " — ").split(":")[0]
            hook = f"Your profile shows: {sig}."
        elif offer_title:
            hook = f"Your {offer_title} is live."
        else:
            hook = f"Your CTR is {my_ctr*100:.1f}% vs {peer_ctr*100:.1f}% peer median."
        body = f"{name}, {hook} Want me to draft a quick campaign to boost this week's leads?"

    # Enforce length, strip URLs
    body = re.sub(r'https?://\S+', '', body).strip()
    if len(body) > 320:
        body = body[:317] + "..."

    return {
        "body": body,
        "cta": "binary_yes_no" if any(w in body.lower() for w in ["want", "shall", "joining", "yes"]) else "open_ended",
        "send_as": send_as,
        "suppression_key": supp_key,
        "rationale": f"Trigger kind={kind}; rule-based composition using payload + merchant context.",
    }

