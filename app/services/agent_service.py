"""Multi-turn agentic service supporting Gemini and OpenAI providers."""

import asyncio
import json
import time
import uuid
import httpx
from app.core.config import settings
from app.core.logging_config import get_logger
from app.core.tenant_context import TenantContext
from app.core.tools import search_products, get_company_policy, execute_order, send_product_image, check_order_status
from app.core.dependencies import get_supabase
from app.services.memory_service import memory_service
from app.services.messaging_service import messaging_service

logger = get_logger(__name__)


# System instruction shared by both providers
SYSTEM_INSTRUCTION = (
    "You are a friendly sales assistant for an online clothing store, chatting with customers on Facebook Messenger.\n\n"

    "## LANGUAGE (STRICT)\n"
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
    "- Show only the BEST match first (1 product). Send its image using 'send_product_image', mention name, price, and paste the product_url on a new line.\n"
    "- Then ask: 'Want to see more options?' or 'Ar dekhben?' (match their language).\n"
    "- Only show the next product when they ask for it.\n"
    "- NEVER dump a list of 3-4 products at once. One at a time, conversationally.\n"
    "- If only 1 result exists, just show that one.\n\n"

    "## IMAGE RULES\n"
    "- NEVER paste image URLs in your text. The user can't click image links on Messenger.\n"
    "- Use the 'send_product_image' tool with the image_url from search results.\n"
    "- Send the image BEFORE or alongside your text about that product.\n"
    "- Max 1 image per reply.\n\n"

    "## PRODUCT LINKS\n"
    "- When mentioning a product, include its product_url as a raw link on its own line (not inside markdown brackets).\n"
    "- Example: 'Here is the link\\nhttps://store.example.com/products/...' — Messenger will auto-preview it.\n\n"

    "## WHEN TO USE TOOLS\n"
    "- 'search_products': When user asks about any product, color, size, price, or says something like 'show me', 'ache?', 'dekhan'.\n"
    "- 'send_product_image': Right after getting search results, send the best match's image. Use the image_url field from results.\n"
    "- 'get_company_policy': When user asks about shipping, return policy, operating hours, delivery time.\n"
    "- 'execute_order': ONLY after user explicitly confirms the order (see order rules below).\n"
    "- 'check_order_status': When user asks about an existing order status, tracking, or gives an order number.\n\n"

    "## ORDER FLOW\n"
    "When a user wants to buy:\n"
    "1. Ask for missing details in ONE message: item name, size, delivery address, contact number.\n"
    "2. Once they provide everything, summarize back to them and ask 'Confirm korben?' / 'Shall I place this order?'\n"
    "3. ONLY call 'execute_order' after explicit confirmation ('yes', 'haan', 'confirm', 'go ahead').\n"
    "4. Never call execute_order without confirmation.\n\n"

    "## TONE\n"
    "- Be casual and warm, like a friend helping them shop.\n"
    "- Short replies. No essays.\n"
    "- One question at a time.\n"
    "- If you don't know something, say so honestly.\n"
    "- Don't over-apologize or sound robotic."
)


class AgentService:
    """Agent orchestrator for handling user messages and tool execution."""

    def __init__(self):
        self.provider = None       # "gemini" or "openai"
        self.gemini_client = None
        self.openai_client = None
        # Provide the actual Python functions. The SDK parses their signatures and docstrings.
        self.tools = [search_products, get_company_policy, execute_order, check_order_status, send_product_image]

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
                from app.services.rag_service import rag_service
                products = await rag_service.search_catalog(
                    query=call_args.get("query", ""),
                    page_id=tenant.shop_id  # shop_id injected, not from LLM
                )
                
                # Truncate tool response to save prompt tokens
                if products:
                    for p in products:
                        p.pop("score", None)  # The agent doesn't need vector similarity scores
                        if "description" in p and isinstance(p["description"], str):
                            p["description"] = p["description"][:100] + ("..." if len(p["description"]) > 100 else "")
                            
                result = {"products_found": products} if products else {"message": "No relevant products found in the catalog."}

            elif call_name == "get_company_policy":
                # Query bot_settings.store_policies for this tenant
                try:
                    db_result = get_supabase().table("bot_settings") \
                        .select("store_policies") \
                        .eq("shop_id", tenant.shop_id) \
                        .maybe_single() \
                        .execute()
                    policies = db_result.data.get("store_policies", "") if db_result.data else ""
                except Exception as db_err:
                    logger.error(f"[{tenant.sender_id}] Failed to fetch store_policies: {db_err}")
                    policies = ""

                if policies:
                    result = {"policies": policies}
                else:
                    result = {"message": "No store policies are configured yet. Please tell the customer to contact support directly."}

            elif call_name == "check_order_status":
                # Query orders table strictly by order_number AND shop_id
                order_num = call_args.get("order_number", "").strip()
                if not order_num:
                    result = {"error": "No order number provided."}
                else:
                    try:
                        db_result = get_supabase().table("orders") \
                            .select("order_number, status, items, sizes, total_amount, delivery_address, created_at") \
                            .eq("order_number", order_num) \
                            .eq("shop_id", tenant.shop_id) \
                            .maybe_single() \
                            .execute()

                        if db_result.data:
                            result = {"order": db_result.data}
                        else:
                            result = {"message": f"No order found with number '{order_num}' for this store."}
                    except Exception as db_err:
                        logger.error(f"[{tenant.sender_id}] Order lookup failed: {db_err}")
                        result = {"error": "Failed to look up the order. Please try again."}

            elif call_name == "execute_order":
                try:
                    # 1. Upsert customer by messenger_psid + shop_id (including profile fields)
                    customer_data = {
                        "messenger_psid": tenant.sender_id,
                        "shop_id": tenant.shop_id,
                        "phone": call_args.get("contact_number", ""),
                        "address": call_args.get("delivery_address", ""),
                        "contact_number": call_args.get("contact_number", ""),
                        "last_delivery_address": call_args.get("delivery_address", ""),
                    }
                    sizes = call_args.get("sizes", "")
                    if sizes:
                        customer_data["preferred_sizes"] = sizes
                    get_supabase().table("customers").upsert(
                        customer_data, on_conflict="messenger_psid,shop_id"
                    ).execute()

                    # 2. Generate order number & insert order
                    order_number = f"ORD-{uuid.uuid4().hex[:8].upper()}"
                    get_supabase().table("orders").insert({
                        "order_number": order_number,
                        "shop_id": tenant.shop_id,
                        "customer_psid": tenant.sender_id,
                        "items": call_args.get("item_names", ""),
                        "sizes": call_args.get("sizes", ""),
                        "delivery_address": call_args.get("delivery_address", ""),
                        "contact_number": call_args.get("contact_number", ""),
                        "status": "processing",
                    }).execute()

                    result = {
                        "status": "success",
                        "order_number": order_number,
                        "message": f"Order {order_number} placed successfully for {call_args.get('item_names', 'items')}."
                    }
                    logger.info(f"[{tenant.sender_id}] 📋 Order created: {order_number} (shop={tenant.shop_id})")

                except Exception as db_err:
                    logger.error(f"[{tenant.sender_id}] Order creation failed: {db_err}", exc_info=True)
                    result = {"status": "failed", "error": "Failed to place the order. Please try again."}

            elif call_name == "send_product_image":
                url = call_args.get("image_url", "")
                if not url or url.lower() in ["none", "null", "undefined"]:
                    logger.warning(f"[{tenant.sender_id}] send_product_image called with invalid URL: '{url}'")
                    result = {"status": "Failed: You must provide a valid image_url string."}
                else:
                    success = await messaging_service.send_image(
                        tenant.sender_id, url,
                        access_token=tenant.page_access_token
                    )
                    if success:
                        result = {"status": "Image successfully dispatched to the user interface."}
                    else:
                        result = {"status": "Failed to dispatch image to Facebook. Invalid URL format or Facebook API error."}
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

    # ─────────────────────────────────────────────────────────────────────
    #  Main entry point
    # ─────────────────────────────────────────────────────────────────────

    async def _get_customer_profile(self, sender_id: str, tenant: TenantContext) -> str:
        """Fetch customer profile from Supabase and return a context string."""
        try:
            result = get_supabase().table("customers") \
                .select("name, preferred_sizes, last_delivery_address, contact_number") \
                .eq("messenger_psid", sender_id) \
                .eq("shop_id", tenant.shop_id) \
                .maybe_single() \
                .execute()

            if not result.data:
                return ""

            profile = result.data
            # Only inject if there's meaningful data
            parts = []
            if profile.get("name"):
                parts.append(f"Name: {profile['name']}")
            if profile.get("preferred_sizes"):
                parts.append(f"Known Sizes: {profile['preferred_sizes']}")
            if profile.get("last_delivery_address"):
                parts.append(f"Last Delivery Address: {profile['last_delivery_address']}")
            if profile.get("contact_number"):
                parts.append(f"Contact: {profile['contact_number']}")

            if not parts:
                return ""

            return f"\n\n[Customer Profile: {', '.join(parts)}. Use this data to streamline confirmations if they re-order.]"

        except Exception as e:
            logger.warning(f"[{sender_id}] Customer profile lookup failed: {e}")
            return ""

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

        # Inject customer profile into system instruction if available
        profile_context = await self._get_customer_profile(sender_id, tenant) if tenant else ""
        self._active_system_instruction = SYSTEM_INSTRUCTION + profile_context

        if self.provider == "openai":
            reply, tokens = await self._process_openai(sender_id, message_text, image_urls, tenant)
        else:
            reply, tokens = await self._process_gemini(sender_id, message_text, image_urls, tenant)

        total_ms = (time.perf_counter() - request_start) * 1000

        # Truncate reply for logging
        reply_preview = (reply[:200] + "…") if len(reply) > 200 else reply
        logger.info(
            f"[{sender_id}] ━━━ REPLY ({total_ms:.0f}ms) ━━━ "
            f"tokens={{in={tokens['prompt']}, out={tokens['completion']}, total={tokens['total']}, turns={tokens['turns']}}} | "
            f"\"{reply_preview}\""
        )

        import asyncio
        if len(memory_service.get_history(sender_id)) > 15:
            asyncio.create_task(self._summarize_history_task(sender_id))

        return reply

    async def _summarize_history_task(self, sender_id: str):
        """Background task to summarize older history to save tokens."""
        history = memory_service.get_history(sender_id)
        if len(history) <= 15:
            return
            
        logger.info(f"[{sender_id}] Triggering background history summarization...")
        
        # We'll take everything except the last 8 messages to summarize
        keep_last_n = 8
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
                memory_service.replace_with_summary(sender_id, keep_last_n, summary)
        except Exception as e:
            logger.error(f"[{sender_id}] Summarization failed: {e}", exc_info=True)

    # ─────────────────────────────────────────────────────────────────────
    #  Gemini path
    # ─────────────────────────────────────────────────────────────────────

    async def _process_gemini(self, sender_id: str, message_text: str, image_urls: list[str] | None, tenant: TenantContext) -> tuple[str, dict]:
        from google import genai
        from google.genai import types

        _zero_tokens = {"prompt": 0, "completion": 0, "total": 0, "turns": 0}

        if not self.gemini_client:
            logger.error(f"[{sender_id}] Gemini client not initialized!")
            return "Server is currently unavailable.", _zero_tokens

        # 1. Load History (as Gemini Content objects)
        history = memory_service.get_gemini_history(sender_id)
        logger.debug(f"[{sender_id}] Loaded {len(history)} history entries")

        # 2. Append User Message
        parts = []
        if message_text:
            parts.append(types.Part.from_text(text=message_text))

        if image_urls:
            for url in image_urls:
                try:
                    import io
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
        memory_service.append_content(sender_id, user_content)
        history.append(user_content)

        # 3. Setup Agent Prompt
        config = types.GenerateContentConfig(
            system_instruction=self._active_system_instruction,
            tools=self.tools,
            temperature=0.7,
            # We explicitly handle the loop ourselves
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
        )

        # 4. Agent Execution Loop
        MAX_TURNS = 5
        total_prompt = 0
        total_completion = 0
        turns_used = 0

        for turn in range(MAX_TURNS):
            logger.debug(f"[{sender_id}] Gemini agent loop — turn {turn + 1}/{MAX_TURNS}")
            try:
                response = await self.gemini_client.aio.models.generate_content(
                    model="gemini-3-flash-preview",
                    contents=history,
                    config=config
                )
            except Exception as e:
                logger.error(f"[{sender_id}] Gemini generation failed: {e}", exc_info=True)
                err_reply = "I ran into a server error processing your request."
                memory_service.append_content(sender_id, types.Content(role="model", parts=[types.Part.from_text(text=err_reply)]))
                return err_reply, {"prompt": total_prompt, "completion": total_completion, "total": total_prompt + total_completion, "turns": turns_used}

            turns_used += 1

            if hasattr(response, "usage_metadata") and response.usage_metadata:
                usage = response.usage_metadata
                p = usage.prompt_token_count or 0
                c = usage.candidates_token_count or 0
                total_prompt += p
                total_completion += c

            if not response.candidates:
                err_reply = "I'm not sure how to respond to that."
                memory_service.append_content(sender_id, types.Content(role="model", parts=[types.Part.from_text(text=err_reply)]))
                return err_reply, {"prompt": total_prompt, "completion": total_completion, "total": total_prompt + total_completion, "turns": turns_used}

            candidate = response.candidates[0]
            if not candidate.content or not candidate.content.parts:
                err_reply = "I'm having trouble thinking."
                memory_service.append_content(sender_id, types.Content(role="model", parts=[types.Part.from_text(text=err_reply)]))
                return err_reply, {"prompt": total_prompt, "completion": total_completion, "total": total_prompt + total_completion, "turns": turns_used}

            # Copy content to avoid modifying references and append to history & memory cache
            memory_service.append_content(sender_id, candidate.content)
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
            memory_service.append_content(sender_id, tool_content)
            history.append(tool_content)

        # If it reached here, it exceeded the loop limit
        logger.warning(f"[{sender_id}] Agent exceeded max turns ({MAX_TURNS})")
        reply = "I had too much to process. Let's start over."
        memory_service.append_content(sender_id, types.Content(role="model", parts=[types.Part.from_text(text=reply)]))
        return reply, {"prompt": total_prompt, "completion": total_completion, "total": total_prompt + total_completion, "turns": turns_used}

    # ─────────────────────────────────────────────────────────────────────
    #  OpenAI path (same logic, different SDK)
    # ─────────────────────────────────────────────────────────────────────

    async def _process_openai(self, sender_id: str, message_text: str, image_urls: list[str] | None, tenant: TenantContext) -> tuple[str, dict]:
        from app.core.openai_tools import OPENAI_TOOLS

        _zero_tokens = {"prompt": 0, "completion": 0, "total": 0, "turns": 0}

        if not self.openai_client:
            logger.error(f"[{sender_id}] OpenAI client not initialized!")
            return "Server is currently unavailable.", _zero_tokens

        # 1. Load History (as OpenAI message dicts)
        messages = [{"role": "system", "content": self._active_system_instruction}]
        messages.extend(memory_service.get_openai_history(sender_id))
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
        memory_service.append_content(sender_id, {
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
            logger.debug(f"[{sender_id}] OpenAI agent loop — turn {turn + 1}/{MAX_TURNS}")
            try:
                response = await self.openai_client.chat.completions.create(
                    model="gpt-5-mini",
                    messages=messages,
                    tools=OPENAI_TOOLS,
                    tool_choice="auto",
                )
            except Exception as e:
                logger.error(f"[{sender_id}] OpenAI generation failed: {e}", exc_info=True)
                err_reply = "I ran into a server error processing your request."
                memory_service.append_content(sender_id, {
                    "role": "model",
                    "parts": [{"type": "text", "text": err_reply}],
                })
                return err_reply, {"prompt": total_prompt, "completion": total_completion, "total": total_prompt + total_completion, "turns": turns_used}

            turns_used += 1

            if hasattr(response, "usage") and response.usage:
                usage = response.usage
                total_prompt += usage.prompt_tokens or 0
                total_completion += usage.completion_tokens or 0

            choice = response.choices[0]
            assistant_msg = choice.message

            if not assistant_msg:
                err_reply = "I'm not sure how to respond to that."
                memory_service.append_content(sender_id, {
                    "role": "model",
                    "parts": [{"type": "text", "text": err_reply}],
                })
                return err_reply, {"prompt": total_prompt, "completion": total_completion, "total": total_prompt + total_completion, "turns": turns_used}

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

            memory_service.append_content(sender_id, {
                "role": "model",
                "parts": internal_parts,
            })

            # Check for tool calls
            if not assistant_msg.tool_calls:
                final = assistant_msg.content or "I'm having trouble thinking."
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
                memory_service.append_content(sender_id, {
                    "role": "user",
                    "parts": [{
                        "type": "function_response",
                        "name": tc.function.name,
                        "response": result,
                        "tool_call_id": tc.id,
                    }],
                })

        # Exceeded loop limit
        logger.warning(f"[{sender_id}] Agent exceeded max turns ({MAX_TURNS})")
        reply = "I had too much to process. Let's start over."
        memory_service.append_content(sender_id, {
            "role": "model",
            "parts": [{"type": "text", "text": reply}],
        })
        return reply, {"prompt": total_prompt, "completion": total_completion, "total": total_prompt + total_completion, "turns": turns_used}

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
