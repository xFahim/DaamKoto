"""Durable conversation persistence — customers, threads, and messages.

Writes every user message and bot reply to Supabase so:
  - the tormoose dashboard can show live chat history (threads/messages tables)
  - conversations survive restarts: on an in-memory cache miss, recent
    history is rehydrated from the DB instead of greeting the customer
    with amnesia mid-order.

All writes are designed to be fired-and-forgotten from the hot path —
failures are logged, never raised into the reply flow.
"""

import asyncio
from cachetools import TTLCache

from app.core.dependencies import get_supabase
from app.core.logging_config import get_logger
from app.core.tenant_context import TenantContext

logger = get_logger(__name__)

# (shop_id, psid) -> customer_id — saves two queries per message
_customer_cache: TTLCache = TTLCache(maxsize=2000, ttl=600)
# (shop_id, customer_id) -> thread_id
_thread_cache: TTLCache = TTLCache(maxsize=2000, ttl=600)
# (shop_id, psid) -> bool: is a human agent handling this thread right now?
# Short TTL so a dashboard "Take Over" click silences the bot within ~20s.
_takeover_cache: TTLCache = TTLCache(maxsize=2000, ttl=20)


class PersistenceService:

    async def get_or_create_customer(
        self, shop_id: str, psid: str, name: str | None = None
    ) -> str:
        """Return the customers.id for this (shop, PSID), creating the row if new."""
        cache_key = (shop_id, psid)
        cached = _customer_cache.get(cache_key)
        if cached:
            return cached

        supabase = await get_supabase()
        result = await supabase.table("customers") \
            .select("id") \
            .eq("shop_id", shop_id) \
            .eq("messenger_psid", psid) \
            .limit(1) \
            .execute()

        if result.data:
            customer_id = result.data[0]["id"]
        else:
            row = {"shop_id": shop_id, "messenger_psid": psid}
            if name:
                row["name"] = name
            try:
                insert = await supabase.table("customers").insert(row).execute()
                customer_id = insert.data[0]["id"]
                logger.info(f"[{psid}] New customer created (shop={shop_id})")
            except Exception:
                # Lost an insert race — another task created the row first
                retry = await supabase.table("customers") \
                    .select("id") \
                    .eq("shop_id", shop_id) \
                    .eq("messenger_psid", psid) \
                    .limit(1) \
                    .execute()
                if not retry.data:
                    raise
                customer_id = retry.data[0]["id"]

        _customer_cache[cache_key] = customer_id
        return customer_id

    async def get_or_create_thread(self, shop_id: str, customer_id: str) -> str:
        """Return the active thread for this customer, creating one if needed.

        Live thread_status enum: 'bot_active' (default) | 'closed' — there is
        no 'open'. Reuse any non-closed thread.
        """
        cache_key = (shop_id, customer_id)
        cached = _thread_cache.get(cache_key)
        if cached:
            return cached

        supabase = await get_supabase()
        result = await supabase.table("threads") \
            .select("id") \
            .eq("shop_id", shop_id) \
            .eq("customer_id", customer_id) \
            .neq("status", "closed") \
            .order("updated_at", desc=True) \
            .limit(1) \
            .execute()

        if result.data:
            thread_id = result.data[0]["id"]
        else:
            insert = await supabase.table("threads").insert({
                "shop_id": shop_id,
                "customer_id": customer_id,
                "status": "bot_active",
            }).execute()
            thread_id = insert.data[0]["id"]
            logger.info(f"Thread created for customer={customer_id} (shop={shop_id})")

        _thread_cache[cache_key] = thread_id
        return thread_id

    async def is_human_active(self, shop_id: str, psid: str) -> bool:
        """True if a human agent has taken over this customer's thread
        (thread_status = 'human_active', set from the tormoose dashboard).

        The bot must then log messages but stay silent. Fails OPEN (bot
        keeps replying) — a DB hiccup shouldn't mute every conversation.
        """
        cache_key = (shop_id, psid)
        cached = _takeover_cache.get(cache_key)
        if cached is not None:
            return cached

        human_active = False
        try:
            supabase = await get_supabase()
            cust = await supabase.table("customers") \
                .select("id") \
                .eq("shop_id", shop_id) \
                .eq("messenger_psid", psid) \
                .limit(1) \
                .execute()
            if cust.data:
                thread = await supabase.table("threads") \
                    .select("status") \
                    .eq("shop_id", shop_id) \
                    .eq("customer_id", cust.data[0]["id"]) \
                    .neq("status", "closed") \
                    .order("updated_at", desc=True) \
                    .limit(1) \
                    .execute()
                human_active = bool(thread.data) and thread.data[0]["status"] == "human_active"
        except Exception as e:
            logger.warning(f"[{psid}] Takeover check failed (bot stays on): {e}")

        _takeover_cache[cache_key] = human_active
        return human_active

    def log_message_bg(
        self, tenant: TenantContext, sender_type: str, content: str
    ) -> None:
        """Fire-and-forget persistence of one message. Never blocks the reply path."""
        if not content:
            return
        task = asyncio.create_task(self._log_message(tenant, sender_type, content))
        task.add_done_callback(self._log_bg_error)

    @staticmethod
    def _log_bg_error(task: asyncio.Task) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc:
            logger.error(f"Message persistence failed: {exc}")

    async def _log_message(
        self, tenant: TenantContext, sender_type: str, content: str
    ) -> None:
        """Write one message row (live sender_type enum: 'customer' | 'bot' | 'human')."""
        customer_id = await self.get_or_create_customer(tenant.shop_id, tenant.sender_id)
        thread_id = await self.get_or_create_thread(tenant.shop_id, customer_id)

        supabase = await get_supabase()
        await supabase.table("messages").insert({
            "thread_id": thread_id,
            "shop_id": tenant.shop_id,
            "sender_type": sender_type,
            "content": content[:8000],
        }).execute()

        # Bump the thread so the dashboard sorts active conversations first
        try:
            await supabase.table("threads").update({"updated_at": "now()"}) \
                .eq("id", thread_id).execute()
        except Exception as e:
            logger.debug(f"Thread timestamp bump failed: {e}")

    async def fetch_recent_transcript(
        self, shop_id: str, psid: str, limit: int = 12
    ) -> list[dict]:
        """Load recent messages from the DB in internal memory format.

        Used to rehydrate in-memory history after a restart or TTL eviction.
        Returns oldest-first entries shaped like MemoryService's internal
        dicts, starting with a user message (Gemini requirement).
        """
        try:
            supabase = await get_supabase()
            cust = await supabase.table("customers") \
                .select("id") \
                .eq("shop_id", shop_id) \
                .eq("messenger_psid", psid) \
                .limit(1) \
                .execute()
            if not cust.data:
                return []
            customer_id = cust.data[0]["id"]

            thread = await supabase.table("threads") \
                .select("id") \
                .eq("shop_id", shop_id) \
                .eq("customer_id", customer_id) \
                .order("updated_at", desc=True) \
                .limit(1) \
                .execute()
            if not thread.data:
                return []

            msgs = await supabase.table("messages") \
                .select("sender_type, content") \
                .eq("thread_id", thread.data[0]["id"]) \
                .order("created_at", desc=True) \
                .limit(limit) \
                .execute()
            if not msgs.data:
                return []

            history = []
            for m in reversed(msgs.data):  # oldest first
                role = "user" if m["sender_type"] == "customer" else "model"
                text = m["content"]
                if m["sender_type"] == "human":
                    text = f"[Human agent replied]: {text}"
                history.append({"role": role, "parts": [{"type": "text", "text": text}]})

            # Must start with a user message
            while history and history[0]["role"] != "user":
                history.pop(0)
            return history

        except Exception as e:
            logger.warning(f"[{psid}] Transcript rehydration failed: {e}")
            return []


persistence_service = PersistenceService()
