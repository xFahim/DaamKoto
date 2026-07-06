"""Per-tenant AI configuration — persona, greeting, and fallback messages.

Reads the ai_configurations table (managed from the dashboard/admin space)
and composes it with the non-negotiable platform rules. Cached per shop so
the hot path costs one dict lookup.
"""

from cachetools import TTLCache

from app.core.dependencies import get_supabase
from app.core.logging_config import get_logger

logger = get_logger(__name__)

# shop_id -> {"system_prompt": ..., "greeting_message": ..., "fallback_message": ...}
# 120s TTL: prompt edits in the dashboard go live within two minutes.
_config_cache: TTLCache = TTLCache(maxsize=500, ttl=120)

DEFAULT_PERSONA = (
    "You are a friendly sales assistant for this online store, chatting with "
    "customers on Facebook Messenger."
)

DEFAULT_FALLBACK = (
    "Sorry, I'm having trouble processing your message right now! "
    "Please try again later!"
)


async def get_ai_config(shop_id: str) -> dict:
    """Return {system_prompt, greeting_message, fallback_message} for a shop.

    Missing rows or DB errors fall back to platform defaults — the bot must
    never go silent because a config row doesn't exist yet.
    """
    cached = _config_cache.get(shop_id)
    if cached is not None:
        return cached

    config = {
        "system_prompt": DEFAULT_PERSONA,
        "greeting_message": "",
        "fallback_message": DEFAULT_FALLBACK,
    }

    try:
        supabase = await get_supabase()
        result = await supabase.table("ai_configurations") \
            .select("system_prompt, greeting_message, fallback_message") \
            .eq("shop_id", shop_id) \
            .maybe_single() \
            .execute()

        if result and result.data:
            row = result.data
            if row.get("system_prompt"):
                config["system_prompt"] = row["system_prompt"].strip()
            if row.get("greeting_message"):
                config["greeting_message"] = row["greeting_message"].strip()
            if row.get("fallback_message"):
                config["fallback_message"] = row["fallback_message"].strip()
            logger.debug(f"AI config loaded for shop={shop_id}")
    except Exception as e:
        logger.warning(f"ai_configurations lookup failed for shop={shop_id}: {e} — using defaults")

    _config_cache[shop_id] = config
    return config


async def get_fallback_message(shop_id: str) -> str:
    """Shortcut for error paths — the shop's fallback line or the default."""
    config = await get_ai_config(shop_id)
    return config["fallback_message"]
