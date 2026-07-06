"""Multi-turn agentic service supporting Gemini and OpenAI providers."""

import asyncio
import json
import time
import uuid
import httpx
from cachetools import TTLCache
from app.core.config import settings
from app.core.logging_config import get_logger
from app.core.tenant_context import TenantContext
from app.core.tools import (
    search_products,
    get_company_policy,
    prepare_order,
    confirm_order,
    check_order_status,
    send_product_image,
)
from app.core.dependencies import get_supabase
from app.services.memory_service import memory_service
from app.services.messaging_service import messaging_service
from app.services.persistence_service import persistence_service
from app.services.tenant_config import get_ai_config

logger = get_logger(__name__)


# Platform rules — non-negotiable behavior appended to every tenant persona.
# The tenant-specific persona (who the store is, what it sells, its voice)
# comes from ai_configurations.system_prompt via tenant_config.
PLATFORM_RULES = (
    "\n\n## LANGUAGE (STRICT)\n"
    "Match the user's language exactly:\n"
    "- English → reply in English\n"
    "- Banglish (Bengali in English letters, e.g. 'kemon acho', 'dekhan', 'eta koto') → reply in Banglish\n"
    "- Bangla (বাংলা script) → reply in Bangla\n"
    "NEVER switch languages. If they write Banglish, you write Banglish. No exceptions.\n\n"

    "## MESSENGER FORMATTING\n"
    "- This is Messenger chat. Messenger does NOT render markdown.\n"
    "- NEVER use: **bold**, *italic*, bullet points (- or *), numbered lists (1. 2. 3.), ![image](url), or [text](url).\n"
    "- Plain text ONLY. Use line breaks to separate info.\n"
    "- Keep every reply 2-4 lines max. Think texting, not email.\n\n"

    "## HOW TO SHOW PRODUCTS\n"
    "When search results come back with multiple products:\n"
    "- Show only the BEST match first (1 product). Send its image using 'send_product_image', mention name and price.\n"
    "- Then ask: 'Want to see more options?' or 'Ar dekhben?' (match their language).\n"
    "- Only show the next product when they ask for it.\n"
    "- NEVER dump a list of 3-4 products at once. One at a time, conversationally.\n"
    "- If only 1 result exists, just show that one.\n"
    "- NEVER paste any URL or link in your text. There are no product page links — the sale happens right here in the chat.\n\n"

    "## SIZES, VARIANTS & STOCK\n"
    "- Each product in search results has a 'variants' list — every size is its OWN product_id with its own stock.\n"
    "- Before preparing an order, ALWAYS know the size. If the user hasn't said one, ask, and mention which sizes are actually available (stock > 0).\n"
    "- In prepare_order, use the product_id of the EXACT variant matching the user's size — never the product_id of a different size.\n"
    "- If the requested size is missing or out of stock, say so honestly and offer the sizes that are in stock.\n"
    "- Don't recite stock numbers unless asked; just treat stock 0 as unavailable.\n\n"

    "## IMAGE RULES\n"
    "- NEVER paste image URLs in your text. The user can't click image links on Messenger.\n"
    "- Use the 'send_product_image' tool with the image_url from search results.\n"
    "- Send the image BEFORE or alongside your text about that product.\n"
    "- Max 1 image per reply.\n\n"

    "## WHEN TO USE TOOLS\n"
    "- 'search_products': When user asks about any product, color, size, price, or says something like 'show me', 'ache?', 'dekhan'.\n"
    "- 'send_product_image': Right after getting search results, send the best match's image. Use the image_url field from results.\n"
    "- 'get_company_policy': When user asks about shipping, return policy, operating hours, delivery time.\n"
    "- 'prepare_order': Once the user has given items, quantities, delivery address, and contact number.\n"
    "- 'confirm_order': ONLY after the user explicitly confirms the prepared order (see order rules below).\n"
    "- 'check_order_status': When user asks about an existing order status, tracking, or gives an order number.\n\n"

    "## ORDER FLOW (STRICT TWO-STEP)\n"
    "When a user wants to buy:\n"
    "1. Ask for missing details in ONE message: which item(s) (from search results), quantity, size/variant if relevant, delivery address, contact number.\n"
    "2. Call 'prepare_order' with the product_id values from search results. It returns the validated summary and exact total.\n"
    "3. Relay that summary and ask 'Confirm korben?' / 'Shall I place this order?'\n"
    "4. ONLY call 'confirm_order' after the user explicitly says yes ('yes', 'haan', 'confirm', 'go ahead').\n"
    "5. NEVER call confirm_order in the same turn as prepare_order. The user must confirm in between.\n\n"

    "## TONE\n"
    "- Be casual and warm, like a friend helping them shop.\n"
    "- Short replies. No essays.\n"
    "- One question at a time.\n"
    "- If you don't know something, say so honestly.\n"
    "- Don't over-apologize or sound robotic."
)

# Order drafts awaiting explicit user confirmation.
# Keyed by "{shop_id}:{sender_id}". 15-minute TTL: an unconfirmed draft dies quietly.
_order_drafts: TTLCache = TTLCache(maxsize=2000, ttl=900)

# Image URLs the model is allowed to send — only ones returned by
# search_products for this conversation. Blocks prompt-injected/hallucinated URLs.
_allowed_images: TTLCache = TTLCache(maxsize=2000, ttl=3600)

# Customer profile snippets, keyed by "{shop_id}:{sender_id}".
_profile_cache: TTLCache = TTLCache(maxsize=2000, ttl=120)


def _conversation_key(tenant: TenantContext) -> str:
    """Memory/draft key. PSIDs are page-scoped, so namespace by shop."""
    return f"{tenant.shop_id}:{tenant.sender_id}"


class AgentService:
    """Agent orchestrator for handling user messages and tool execution."""

    def __init__(self):
        self.provider = None       # "gemini" or "openai"
        self.gemini_client = None
        self.openai_client = None
        # Provide the actual Python functions. The SDK parses their signatures and docstrings.
        self.tools = [
            search_products,
            get_company_policy,
            prepare_order,
            confirm_order,
            check_order_status,
            send_product_image,
        ]
        # Senders with an in-flight history summarization (prevents lost updates)
        self._summarizing: set[str] = set()

    def initialize(self):
        """Configure the LLM client based on the selected provider."""
        self.provider = settings.llm_provider.lower().strip()
        logger.info(f"Initializing agent with provider: {self.provider}")

        if self.provider == "openai":
            from openai import AsyncOpenAI
            if not settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
            self.openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
            logger.info("Agent service initialized — OpenAI (gpt-5-mini)")
        else:
            from app.core.dependencies import genai_client
            self.gemini_client = genai_client
            self.provider = "gemini"
            logger.info("Agent service initialized — Gemini (gemini-3-flash-preview)")

    # ─────────────────────────────────────────────────────────────────────
    #  Tool execution bridge
    # ─────────────────────────────────────────────────────────────────────

    async def _execute_tool(self, call_name: str, call_args: dict, tenant: TenantContext) -> dict:
        """Execute a tool with tenant context injected server-side.

        The LLM only provides call_args from its tool schema (no shop_id, no sender_id).
        Tenant isolation is enforced here by using tenant.shop_id and tenant.sender_id.
        """
        logger.info(f"[{tenant.sender_id}] 🔧 TOOL CALL: {call_name}({json.dumps(call_args, ensure_ascii=False)})")

        # Send typing to show we're doing stuff
        await messaging_service.send_typing_on(tenant.sender_id, access_token=tenant.page_access_token)

        tool_start = time.perf_counter()

        try:
            if call_name == "search_products":
                result = await self._tool_search_products(call_args, tenant)
            elif call_name == "get_company_policy":
                result = await self._tool_get_company_policy(tenant)
            elif call_name == "prepare_order":
                result = await self._tool_prepare_order(call_args, tenant)
            elif call_name == "confirm_order":
                result = await self._tool_confirm_order(tenant)
            elif call_name == "check_order_status":
                result = await self._tool_check_order_status(call_args, tenant)
            elif call_name == "send_product_image":
                result = await self._tool_send_product_image(call_args, tenant)
            else:
                result = {"error": f"Unknown tool: {call_name}"}
        except Exception as e:
            logger.error(f"[{tenant.sender_id}] Tool {call_name} raised exception: {e}", exc_info=True)
            result = {"error": str(e)}

        # Ensure result is always a dict
        if not isinstance(result, dict):
            result = {"result": result}

        elapsed_ms = (time.perf_counter() - tool_start) * 1000
        # Truncate result for log readability
        result_preview = json.dumps(result, ensure_ascii=False)
        if len(result_preview) > 300:
            result_preview = result_preview[:300] + "…"
        logger.info(f"[{tenant.sender_id}] 🔧 TOOL RESULT ({call_name}, {elapsed_ms:.0f}ms): {result_preview}")

        return result

    async def _tool_search_products(self, call_args: dict, tenant: TenantContext) -> dict:
        from app.services.rag_service import rag_service
        products = await rag_service.search_catalog(
            query=call_args.get("query", ""),
            shop_id=tenant.shop_id,  # shop_id injected, not from LLM
        )

        if not products:
            return {"message": "No relevant products found in the catalog."}

        # Whitelist the returned image URLs for send_product_image
        key = _conversation_key(tenant)
        allowed: set = _allowed_images.get(key) or set()

        for p in products:
            # Whitelist every variant's image, but only expose the primary one
            for url in p.pop("all_image_urls", []):
                allowed.add(url)
            if p.get("image_url"):
                allowed.add(p["image_url"])
            if isinstance(p.get("description"), str):
                p["description"] = p["description"][:100] + ("..." if len(p["description"]) > 100 else "")

        _allowed_images[key] = allowed
        return {"products_found": products}

    async def _tool_get_company_policy(self, tenant: TenantContext) -> dict:
        try:
            supabase = await get_supabase()
            db_result = await supabase.table("bot_settings") \
                .select("store_policies") \
                .eq("shop_id", tenant.shop_id) \
                .maybe_single() \
                .execute()
            policies = db_result.data.get("store_policies", "") if db_result and db_result.data else ""
        except Exception as db_err:
            logger.error(f"[{tenant.sender_id}] Failed to fetch store_policies: {db_err}")
            policies = ""

        if policies:
            return {"policies": policies}
        return {"message": "No store policies are configured yet. Please tell the customer to contact support directly."}

    async def _tool_prepare_order(self, call_args: dict, tenant: TenantContext) -> dict:
        """Validate items against the catalog, compute the total, store a draft."""
        product_ids = call_args.get("product_ids") or []
        quantities = call_args.get("quantities") or []
        delivery_address = (call_args.get("delivery_address") or "").strip()
        contact_number = (call_args.get("contact_number") or "").strip()
        notes = (call_args.get("notes") or "").strip()

        if not product_ids:
            return {"error": "No product_ids given. Use the product_id values from search_products results."}
        if len(quantities) != len(product_ids):
            return {"error": "product_ids and quantities must have the same length."}
        if not delivery_address:
            return {"error": "Missing delivery_address. Ask the user for their full delivery address."}
        if not contact_number:
            return {"error": "Missing contact_number. Ask the user for their phone number."}

        try:
            quantities = [int(q) for q in quantities]
        except (TypeError, ValueError):
            return {"error": "quantities must be whole numbers."}
        if any(q < 1 or q > 50 for q in quantities):
            return {"error": "Each quantity must be between 1 and 50."}

        # Validate against the catalog — the LLM can only order real products of THIS shop
        supabase = await get_supabase()
        db_result = await supabase.table("products") \
            .select("id, name, price, attributes") \
            .eq("shop_id", tenant.shop_id) \
            .in_("id", product_ids) \
            .execute()

        found = {row["id"]: row for row in (db_result.data or [])}
        missing = [pid for pid in product_ids if pid not in found]
        if missing:
            return {
                "error": f"These product_ids don't exist in this store's catalog: {missing}. "
                         "Use exact product_id values from search_products results."
            }

        items = []
        total = 0.0
        for pid, qty in zip(product_ids, quantities):
            row = found[pid]
            attrs = row.get("attributes") if isinstance(row.get("attributes"), dict) else {}
            size = attrs.get("size")
            display_name = f"{row['name']} (size {size})" if size else row["name"]

            # Stock check — attributes.stock is per size variant
            stock = attrs.get("stock")
            try:
                stock = int(stock) if stock is not None else None
            except (TypeError, ValueError):
                stock = None
            if stock is not None and stock <= 0:
                return {
                    "error": f"'{display_name}' is OUT OF STOCK. Tell the user and offer "
                             "another size or product from the search results."
                }
            if stock is not None and qty > stock:
                return {
                    "error": f"Only {stock} left in stock for '{display_name}' but {qty} requested. "
                             "Ask the user if the available quantity works."
                }

            unit_price = float(row["price"])
            items.append({
                "product_id": pid,
                "name": display_name,
                "unit_price": unit_price,
                "quantity": qty,
                "line_total": round(unit_price * qty, 2),
            })
            total += unit_price * qty

        draft = {
            "items": items,
            "total_amount": round(total, 2),
            "delivery_address": delivery_address,
            "contact_number": contact_number,
            "notes": notes,
        }
        _order_drafts[_conversation_key(tenant)] = draft

        logger.info(
            f"[{tenant.sender_id}] 📝 Order draft prepared — {len(items)} item(s), "
            f"total={draft['total_amount']} (shop={tenant.shop_id})"
        )
        return {
            "status": "draft_ready",
            "summary": draft,
            "instruction": (
                "Relay this summary (items, quantities, total, address) to the user in their "
                "language and ask them to confirm. Call confirm_order ONLY after they say yes."
            ),
        }

    async def _tool_confirm_order(self, tenant: TenantContext) -> dict:
        """Write the confirmed draft to customers → orders → order_items."""
        key = _conversation_key(tenant)
        draft = _order_drafts.get(key)
        if not draft:
            return {
                "error": "There is no prepared order to confirm. Use prepare_order first "
                         "(or the draft expired after 15 minutes — prepare it again)."
            }

        try:
            supabase = await get_supabase()

            # 1. Resolve (or create) the customer and refresh their profile
            customer_id = await persistence_service.get_or_create_customer(
                tenant.shop_id, tenant.sender_id
            )
            profile_update = {
                "contact_number": draft["contact_number"],
                "last_delivery_address": draft["delivery_address"],
            }
            if draft["notes"]:
                profile_update["preferred_sizes"] = draft["notes"]
            await supabase.table("customers").update(profile_update) \
                .eq("id", customer_id).execute()

            # 2. Insert the order
            order_number = f"ORD-{uuid.uuid4().hex[:8].upper()}"
            order_insert = await supabase.table("orders").insert({
                "order_number": order_number,
                "shop_id": tenant.shop_id,
                "customer_id": customer_id,
                # DB check constraint allows: processing | shipped | delivered | cancelled
                "status": "processing",
                "total_amount": draft["total_amount"],
                "delivery_address": draft["delivery_address"],
                "contact_number": draft["contact_number"],
                "notes": draft["notes"] or None,
            }).execute()
            order_id = order_insert.data[0]["id"]

            # 3. Insert line items
            await supabase.table("order_items").insert([
                {
                    "order_id": order_id,
                    "product_id": item["product_id"],
                    "quantity": item["quantity"],
                    "unit_price_at_time": item["unit_price"],
                }
                for item in draft["items"]
            ]).execute()

            # Draft consumed — a second confirm_order call can't double-book
            _order_drafts.pop(key, None)
            _profile_cache.pop(key, None)

            logger.info(
                f"[{tenant.sender_id}] 📋 Order created: {order_number} "
                f"total={draft['total_amount']} (shop={tenant.shop_id})"
            )
            return {
                "status": "success",
                "order_number": order_number,
                "total_amount": draft["total_amount"],
                "message": f"Order {order_number} placed successfully.",
            }

        except Exception as db_err:
            logger.error(f"[{tenant.sender_id}] Order creation failed: {db_err}", exc_info=True)
            return {"status": "failed", "error": "Failed to place the order. Please try again."}

    async def _tool_check_order_status(self, call_args: dict, tenant: TenantContext) -> dict:
        """Look up an order — scoped to THIS customer so PSIDs can't read each other's orders."""
        order_num = (call_args.get("order_number") or "").strip()
        if not order_num:
            return {"error": "No order number provided."}

        try:
            supabase = await get_supabase()

            cust = await supabase.table("customers") \
                .select("id") \
                .eq("shop_id", tenant.shop_id) \
                .eq("messenger_psid", tenant.sender_id) \
                .limit(1) \
                .execute()
            if not cust.data:
                return {"message": "No orders found for this customer yet."}

            db_result = await supabase.table("orders") \
                .select(
                    "order_number, status, total_amount, tracking_link, "
                    "delivery_address, created_at, "
                    "order_items(quantity, unit_price_at_time, products(name))"
                ) \
                .eq("order_number", order_num) \
                .eq("shop_id", tenant.shop_id) \
                .eq("customer_id", cust.data[0]["id"]) \
                .limit(1) \
                .execute()

            if db_result.data:
                return {"order": db_result.data[0]}
            return {"message": f"No order found with number '{order_num}' for this customer."}
        except Exception as db_err:
            logger.error(f"[{tenant.sender_id}] Order lookup failed: {db_err}")
            return {"error": "Failed to look up the order. Please try again."}

    async def _tool_send_product_image(self, call_args: dict, tenant: TenantContext) -> dict:
        url = (call_args.get("image_url") or "").strip()
        if not url or url.lower() in ["none", "null", "undefined"]:
            logger.warning(f"[{tenant.sender_id}] send_product_image called with invalid URL: '{url}'")
            return {"status": "Failed: You must provide a valid image_url string."}

        # Only URLs that came back from search_products may be sent — blocks
        # hallucinated or prompt-injected URLs going out under the shop's name.
        allowed = _allowed_images.get(_conversation_key(tenant)) or set()
        if url not in allowed:
            logger.warning(f"[{tenant.sender_id}] send_product_image blocked non-catalog URL: {url[:100]}")
            return {
                "status": "Failed: That URL is not from this store's search results. "
                          "Use the exact image_url field from search_products output."
            }

        success = await messaging_service.send_image(
            tenant.sender_id, url,
            access_token=tenant.page_access_token,
        )
        if success:
            # Log to the dashboard transcript
            persistence_service.log_message_bg(tenant, "bot", f"[image] {url}")
            return {"status": "Image successfully dispatched to the user interface."}
        return {"status": "Failed to dispatch image to Facebook. Invalid URL format or Facebook API error."}

    # ─────────────────────────────────────────────────────────────────────
    #  Customer profile
    # ─────────────────────────────────────────────────────────────────────

    async def _get_customer_profile(self, tenant: TenantContext) -> str:
        """Fetch customer profile from Supabase and return a context string."""
        key = _conversation_key(tenant)
        cached = _profile_cache.get(key)
        if cached is not None:
            return cached

        profile_context = ""
        try:
            supabase = await get_supabase()
            result = await supabase.table("customers") \
                .select("name, preferred_sizes, last_delivery_address, contact_number") \
                .eq("messenger_psid", tenant.sender_id) \
                .eq("shop_id", tenant.shop_id) \
                .maybe_single() \
                .execute()

            if result and result.data:
                profile = result.data
                parts = []
                if profile.get("name"):
                    parts.append(f"Name: {profile['name']}")
                if profile.get("preferred_sizes"):
                    parts.append(f"Known Sizes: {profile['preferred_sizes']}")
                if profile.get("last_delivery_address"):
                    parts.append(f"Last Delivery Address: {profile['last_delivery_address']}")
                if profile.get("contact_number"):
                    parts.append(f"Contact: {profile['contact_number']}")

                if parts:
                    profile_context = (
                        f"\n\n[Customer Profile: {', '.join(parts)}. "
                        "Use this data to streamline confirmations if they re-order.]"
                    )
        except Exception as e:
            logger.warning(f"[{tenant.sender_id}] Customer profile lookup failed: {e}")

        _profile_cache[key] = profile_context
        return profile_context

    # ─────────────────────────────────────────────────────────────────────
    #  Main entry point
    # ─────────────────────────────────────────────────────────────────────

    async def process(self, sender_id: str, message_text: str = "", image_urls: list[str] = None, tenant: TenantContext = None) -> str:
        """Process a message through the ReAct agent loop."""
        # Truncate message for log readability
        msg_preview = (message_text[:120] + "…") if len(message_text) > 120 else message_text
        img_count = len(image_urls) if image_urls else 0
        logger.info(
            f"[{sender_id}] ━━━ INCOMING ━━━ provider={self.provider} | "
            f"text=\"{msg_preview}\" | images={img_count}"
        )

        request_start = time.perf_counter()
        mem_key = _conversation_key(tenant)

        # Rehydrate memory from the DB after a restart or TTL eviction, so the
        # bot doesn't lose the thread mid-conversation.
        if not memory_service.get_history(mem_key):
            transcript = await persistence_service.fetch_recent_transcript(
                tenant.shop_id, tenant.sender_id
            )
            if transcript:
                memory_service.seed_history(mem_key, transcript)
                logger.info(f"[{sender_id}] 💧 Rehydrated {len(transcript)} messages from DB")

        # Compose the per-request system instruction: tenant persona + platform
        # rules + customer profile. Passed as a parameter (never stored on self)
        # so concurrent conversations can't leak profiles into each other.
        ai_config = await get_ai_config(tenant.shop_id)
        profile_context = await self._get_customer_profile(tenant)
        greeting_hint = (
            f"\n\n[If this is the start of the conversation, open with: \"{ai_config['greeting_message']}\"]"
            if ai_config["greeting_message"] else ""
        )
        system_instruction = (
            ai_config["system_prompt"] + PLATFORM_RULES + greeting_hint + profile_context
        )
        fallback_reply = ai_config["fallback_message"]

        if self.provider == "openai":
            reply, tokens = await self._process_openai(
                mem_key, sender_id, message_text, image_urls, tenant, system_instruction, fallback_reply
            )
        else:
            reply, tokens = await self._process_gemini(
                mem_key, sender_id, message_text, image_urls, tenant, system_instruction, fallback_reply
            )

        total_ms = (time.perf_counter() - request_start) * 1000

        # Truncate reply for logging
        reply_preview = (reply[:200] + "…") if len(reply) > 200 else reply
        logger.info(
            f"[{sender_id}] ━━━ REPLY ({total_ms:.0f}ms) ━━━ "
            f"tokens={{in={tokens['prompt']}, out={tokens['completion']}, total={tokens['total']}, turns={tokens['turns']}}} | "
            f"\"{reply_preview}\""
        )

        if len(memory_service.get_history(mem_key)) > 15 and mem_key not in self._summarizing:
            self._summarizing.add(mem_key)
            task = asyncio.create_task(self._summarize_history_task(mem_key, sender_id))
            task.add_done_callback(lambda t, k=mem_key: self._summarizing.discard(k))

        return reply

    async def _summarize_history_task(self, mem_key: str, sender_id: str):
        """Background task to summarize older history to save tokens."""
        history = memory_service.get_history(mem_key)
        if len(history) <= 15:
            return

        logger.info(f"[{sender_id}] Triggering background history summarization...")

        # We'll take everything except the last 8 messages to summarize
        keep_last_n = 8
        snapshot_len = len(history)
        to_summarize = history[:-keep_last_n]

        # Create a tiny prompt to summarize
        prompt = "Summarize this conversation concisely. Focus on user intent, missing information, and products discussed. Maximum 3 sentences."

        # Build text representation of older conversation
        text_convo = ""
        for msg in to_summarize:
            role = msg.get("role", "unknown")
            parts = msg.get("parts", [])
            for p in parts:
                if p.get("type") == "text" and p.get("text"):
                    text_convo += f"{role}: {p['text']}\n"

        if not text_convo.strip():
            return

        try:
            summary = ""
            if self.provider == "openai" and self.openai_client:
                messages = [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": text_convo}
                ]
                resp = await self.openai_client.chat.completions.create(
                    model="gpt-4.1-nano",  # Use cheapest model for summarization
                    messages=messages,
                )
                if resp.choices and resp.choices[0].message.content:
                    summary = resp.choices[0].message.content
            elif self.provider == "gemini" and self.gemini_client:
                from google.genai import types
                resp = await self.gemini_client.aio.models.generate_content(
                    model="gemini-2.5-flash",  # Cheaper model for summarization
                    contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt + "\n\n" + text_convo)])],
                )
                if resp.candidates and resp.candidates[0].content.parts:
                    summary = "\n".join([p.text for p in resp.candidates[0].content.parts if p.text])

            if summary:
                logger.info(f"[{sender_id}] Replaced history with summary: {summary}")
                # Keep everything appended since the snapshot, not just the last 8
                grown_by = len(memory_service.get_history(mem_key)) - snapshot_len
                memory_service.replace_with_summary(mem_key, keep_last_n + max(0, grown_by), summary)
        except Exception as e:
            logger.error(f"[{sender_id}] Summarization failed: {e}", exc_info=True)

    # ─────────────────────────────────────────────────────────────────────
    #  Gemini path
    # ─────────────────────────────────────────────────────────────────────

    async def _process_gemini(
        self,
        mem_key: str,
        sender_id: str,
        message_text: str,
        image_urls: list[str] | None,
        tenant: TenantContext,
        system_instruction: str,
        fallback_reply: str,
    ) -> tuple[str, dict]:
        from google import genai
        from google.genai import types

        _zero_tokens = {"prompt": 0, "completion": 0, "total": 0, "turns": 0}

        if not self.gemini_client:
            logger.error(f"[{sender_id}] Gemini client not initialized!")
            return fallback_reply, _zero_tokens

        # 1. Load History (as Gemini Content objects)
        history = memory_service.get_gemini_history(mem_key)
        logger.debug(f"[{sender_id}] Loaded {len(history)} history entries")

        # 2. Append User Message
        parts = []
        if message_text:
            parts.append(types.Part.from_text(text=message_text))

        if image_urls:
            for url in image_urls:
                try:
                    import tempfile
                    import os

                    async with httpx.AsyncClient(timeout=10.0) as http_client:
                        response = await http_client.get(url)
                        response.raise_for_status()
                        image_bytes = response.content

                    # Write to temp file because the SDK upload function requires a valid file path or supported file-like object
                    temp_file_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}.jpg")
                    with open(temp_file_path, "wb") as f:
                        f.write(image_bytes)

                    uploaded_file = await self.gemini_client.aio.files.upload(
                        file=temp_file_path,
                        config={'mime_type': 'image/jpeg'}
                    )

                    # Cleanup temp
                    os.remove(temp_file_path)

                    # Use the Cloud URI so your server's context memory RAM stays tiny!
                    parts.append(types.Part.from_uri(uri=uploaded_file.uri, mime_type="image/jpeg"))
                    logger.info(f"[{sender_id}] 📷 Uploaded image to Gemini Cloud: {uploaded_file.uri}")

                except Exception as e:
                    logger.error(f"[{sender_id}] Failed to upload image: {e}")
                    parts.append(types.Part.from_text(text="[User attached an image but the system failed to download it]"))

        if not parts:
            parts.append(types.Part.from_text(text="[Empty message]"))

        user_content = types.Content(role="user", parts=parts)
        memory_service.append_content(mem_key, user_content)
        history.append(user_content)

        # 3. Setup Agent Prompt
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=self.tools,
            temperature=0.7,
            # We explicitly handle the loop ourselves
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
        )
        # Final-turn config: tools stay declared (history contains calls) but the
        # model is forbidden from calling them, forcing a text answer.
        final_turn_config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=self.tools,
            temperature=0.7,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="NONE")
            ),
        )

        # 4. Agent Execution Loop
        MAX_TURNS = 5
        total_prompt = 0
        total_completion = 0
        turns_used = 0

        for turn in range(MAX_TURNS):
            is_final_turn = turn == MAX_TURNS - 1
            logger.debug(f"[{sender_id}] Gemini agent loop — turn {turn + 1}/{MAX_TURNS}")
            try:
                response = await self.gemini_client.aio.models.generate_content(
                    model="gemini-3-flash-preview",
                    contents=history,
                    config=final_turn_config if is_final_turn else config
                )
            except Exception as e:
                logger.error(f"[{sender_id}] Gemini generation failed: {e}", exc_info=True)
                memory_service.append_content(mem_key, types.Content(role="model", parts=[types.Part.from_text(text=fallback_reply)]))
                return fallback_reply, {"prompt": total_prompt, "completion": total_completion, "total": total_prompt + total_completion, "turns": turns_used}

            turns_used += 1

            if hasattr(response, "usage_metadata") and response.usage_metadata:
                usage = response.usage_metadata
                p = usage.prompt_token_count or 0
                c = usage.candidates_token_count or 0
                total_prompt += p
                total_completion += c

            if not response.candidates:
                memory_service.append_content(mem_key, types.Content(role="model", parts=[types.Part.from_text(text=fallback_reply)]))
                return fallback_reply, {"prompt": total_prompt, "completion": total_completion, "total": total_prompt + total_completion, "turns": turns_used}

            candidate = response.candidates[0]
            if not candidate.content or not candidate.content.parts:
                memory_service.append_content(mem_key, types.Content(role="model", parts=[types.Part.from_text(text=fallback_reply)]))
                return fallback_reply, {"prompt": total_prompt, "completion": total_completion, "total": total_prompt + total_completion, "turns": turns_used}

            # Copy content to avoid modifying references and append to history & memory cache
            memory_service.append_content(mem_key, candidate.content)
            history.append(candidate.content)

            # Check for tool calls
            tool_calls = [p.function_call for p in candidate.content.parts if p.function_call]

            if not tool_calls:
                # Text response generated, break loop and return the text
                text_parts = [p.text for p in candidate.content.parts if p.text]
                final_text = "\n".join(text_parts)
                return final_text, {"prompt": total_prompt, "completion": total_completion, "total": total_prompt + total_completion, "turns": turns_used}

            logger.info(f"[{sender_id}] 🤖 Agent decided to call {len(tool_calls)} tool(s): {[c.name for c in tool_calls]}")

            # Execute tools in parallel — tenant context injected, not from LLM args
            tasks = [
                self._execute_tool(call.name, dict(call.args) if call.args else {}, tenant)
                for call in tool_calls
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Match results back to their tool calls by index order
            tool_responses = []
            for call, result in zip(tool_calls, results):
                if isinstance(result, Exception):
                    logger.error(f"[{sender_id}] Tool {call.name} raised in parallel: {result}")
                    result = {"error": str(result)}
                tool_responses.append(types.Part.from_function_response(
                    name=call.name,
                    response=result
                ))

            # Send tool responses back to model
            tool_content = types.Content(role="user", parts=tool_responses)
            memory_service.append_content(mem_key, tool_content)
            history.append(tool_content)

        # Unreachable in practice — the final turn forbids tool calls, so it
        # always returns text above. Kept as a safety net.
        logger.warning(f"[{sender_id}] Agent exceeded max turns ({MAX_TURNS})")
        memory_service.append_content(mem_key, types.Content(role="model", parts=[types.Part.from_text(text=fallback_reply)]))
        return fallback_reply, {"prompt": total_prompt, "completion": total_completion, "total": total_prompt + total_completion, "turns": turns_used}

    # ─────────────────────────────────────────────────────────────────────
    #  OpenAI path (same logic, different SDK)
    # ─────────────────────────────────────────────────────────────────────

    async def _process_openai(
        self,
        mem_key: str,
        sender_id: str,
        message_text: str,
        image_urls: list[str] | None,
        tenant: TenantContext,
        system_instruction: str,
        fallback_reply: str,
    ) -> tuple[str, dict]:
        from app.core.openai_tools import OPENAI_TOOLS

        _zero_tokens = {"prompt": 0, "completion": 0, "total": 0, "turns": 0}

        if not self.openai_client:
            logger.error(f"[{sender_id}] OpenAI client not initialized!")
            return fallback_reply, _zero_tokens

        # 1. Load History (as OpenAI message dicts)
        messages = [{"role": "system", "content": system_instruction}]
        messages.extend(memory_service.get_openai_history(mem_key))
        logger.debug(f"[{sender_id}] Loaded {len(messages) - 1} history entries (excl. system)")

        # 2. Build user message
        if image_urls:
            # Multimodal message with images — download and base64-encode because
            # Facebook CDN URLs are temporary/restricted and OpenAI can't fetch them.
            import base64
            content_parts = []
            if message_text:
                content_parts.append({"type": "text", "text": message_text})
            for url in image_urls:
                try:
                    async with httpx.AsyncClient(timeout=10.0) as http_client:
                        img_resp = await http_client.get(url)
                        img_resp.raise_for_status()
                        image_bytes = img_resp.content
                    b64 = base64.b64encode(image_bytes).decode("utf-8")
                    data_uri = f"data:image/jpeg;base64,{b64}"
                    content_parts.append({"type": "image_url", "image_url": {"url": data_uri}})
                    logger.info(f"[{sender_id}] 📷 Downloaded image for OpenAI ({len(image_bytes)} bytes)")
                except Exception as e:
                    logger.error(f"[{sender_id}] Failed to download image for OpenAI: {e}")
                    content_parts.append({"type": "text", "text": "[User sent an image but the system failed to download it]"})
            user_msg = {"role": "user", "content": content_parts}
        elif message_text:
            user_msg = {"role": "user", "content": message_text}
        else:
            user_msg = {"role": "user", "content": "[Empty message]"}

        # Store in provider-agnostic memory
        memory_service.append_content(mem_key, {
            "role": "user",
            "parts": self._openai_msg_to_parts(user_msg),
        })
        messages.append(user_msg)

        # 3. Agent Execution Loop
        MAX_TURNS = 5
        total_prompt = 0
        total_completion = 0
        turns_used = 0

        for turn in range(MAX_TURNS):
            is_final_turn = turn == MAX_TURNS - 1
            logger.debug(f"[{sender_id}] OpenAI agent loop — turn {turn + 1}/{MAX_TURNS}")
            try:
                response = await self.openai_client.chat.completions.create(
                    model="gpt-5-mini",
                    messages=messages,
                    tools=OPENAI_TOOLS,
                    # The final turn forbids tools, forcing a text answer instead
                    # of dying with "I had too much to process".
                    tool_choice="none" if is_final_turn else "auto",
                )
            except Exception as e:
                logger.error(f"[{sender_id}] OpenAI generation failed: {e}", exc_info=True)
                memory_service.append_content(mem_key, {
                    "role": "model",
                    "parts": [{"type": "text", "text": fallback_reply}],
                })
                return fallback_reply, {"prompt": total_prompt, "completion": total_completion, "total": total_prompt + total_completion, "turns": turns_used}

            turns_used += 1

            if hasattr(response, "usage") and response.usage:
                usage = response.usage
                total_prompt += usage.prompt_tokens or 0
                total_completion += usage.completion_tokens or 0

            choice = response.choices[0]
            assistant_msg = choice.message

            if not assistant_msg:
                memory_service.append_content(mem_key, {
                    "role": "model",
                    "parts": [{"type": "text", "text": fallback_reply}],
                })
                return fallback_reply, {"prompt": total_prompt, "completion": total_completion, "total": total_prompt + total_completion, "turns": turns_used}

            # Save assistant message to history
            assistant_dict = assistant_msg.model_dump(exclude_none=True)
            messages.append(assistant_dict)

            # Save to memory in our internal format
            internal_parts = []
            if assistant_msg.content:
                internal_parts.append({"type": "text", "text": assistant_msg.content})
            if assistant_msg.tool_calls:
                for tc in assistant_msg.tool_calls:
                    internal_parts.append({
                        "type": "function_call",
                        "name": tc.function.name,
                        "args": json.loads(tc.function.arguments) if tc.function.arguments else {},
                        "tool_call_id": tc.id,
                    })
            if not internal_parts:
                internal_parts.append({"type": "text", "text": ""})

            memory_service.append_content(mem_key, {
                "role": "model",
                "parts": internal_parts,
            })

            # Check for tool calls
            if not assistant_msg.tool_calls:
                final = assistant_msg.content or fallback_reply
                return final, {"prompt": total_prompt, "completion": total_completion, "total": total_prompt + total_completion, "turns": turns_used}

            tool_names = [tc.function.name for tc in assistant_msg.tool_calls]
            logger.info(f"[{sender_id}] 🤖 Agent decided to call {len(tool_names)} tool(s): {tool_names}")

            # Execute tools in parallel — tenant context injected, not from LLM args
            tasks = [
                self._execute_tool(
                    tc.function.name,
                    json.loads(tc.function.arguments) if tc.function.arguments else {},
                    tenant
                )
                for tc in assistant_msg.tool_calls
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Match results back to their tool calls by index order
            for tc, result in zip(assistant_msg.tool_calls, results):
                if isinstance(result, Exception):
                    logger.error(f"[{sender_id}] Tool {tc.function.name} raised in parallel: {result}")
                    result = {"error": str(result)}

                tool_msg = {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result),
                }
                messages.append(tool_msg)

                # Save to internal memory
                memory_service.append_content(mem_key, {
                    "role": "user",
                    "parts": [{
                        "type": "function_response",
                        "name": tc.function.name,
                        "response": result,
                        "tool_call_id": tc.id,
                    }],
                })

        # Unreachable in practice — the final turn forbids tool calls. Safety net.
        logger.warning(f"[{sender_id}] Agent exceeded max turns ({MAX_TURNS})")
        memory_service.append_content(mem_key, {
            "role": "model",
            "parts": [{"type": "text", "text": fallback_reply}],
        })
        return fallback_reply, {"prompt": total_prompt, "completion": total_completion, "total": total_prompt + total_completion, "turns": turns_used}

    @staticmethod
    def _openai_msg_to_parts(msg: dict) -> list[dict]:
        """Convert an OpenAI-style user message to our internal parts format."""
        content = msg.get("content", "")
        if isinstance(content, str):
            return [{"type": "text", "text": content}]
        elif isinstance(content, list):
            parts = []
            for item in content:
                if item.get("type") == "text":
                    parts.append({"type": "text", "text": item["text"]})
                elif item.get("type") == "image_url":
                    parts.append({"type": "file_data", "uri": item["image_url"]["url"], "mime_type": "image/jpeg"})
            return parts
        return [{"type": "text", "text": str(content)}]


agent_service = AgentService()
