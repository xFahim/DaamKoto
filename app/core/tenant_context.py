"""Tenant context resolution for multi-tenant request handling.

Resolves a Facebook page ID to an internal shop_id + access token
by querying the bot_settings table in Supabase. The result is cached
in-memory for 60 seconds to avoid repeated DB hits on every message.
"""

from dataclasses import dataclass, replace
from cachetools import TTLCache
from app.core.dependencies import get_supabase
from app.core.logging_config import get_logger

logger = get_logger(__name__)

# Cache tenant lookups for 60s — avoids hitting Supabase on every webhook message.
# Inactive tenants are cached too, so a disabled bot doesn't hammer the DB.
# Flipping is_active takes effect within 60s.
_tenant_cache: TTLCache = TTLCache(maxsize=200, ttl=60)


class TenantNotFoundError(Exception):
    """Raised when no bot_settings row matches the given Facebook page ID."""
    pass


class TenantInactiveError(Exception):
    """Raised when the bot for this page exists but is_active is false."""
    pass


@dataclass(frozen=True)
class TenantContext:
    """Per-message tenant state resolved from bot_settings.

    Frozen: one webhook entry can contain events from multiple senders, and
    the batcher holds references to this object across await points — a
    mutable sender_id would let one sender's order be attributed to another.
    Use for_sender() to derive a per-message copy.

    Tools never see shop_id — it's injected server-side in the ReAct
    execution bridge.
    """
    shop_id: str                    # Internal UUID from bot_settings
    page_access_token: str          # Facebook Graph API token for this page
    facebook_page_id: str           # The numeric Facebook page ID (entry.id)
    sender_id: str = ""             # Messenger PSID — stamped per-message via for_sender()
    allow_split_replies: bool = False  # Owner toggle: bot may send multi-bubble replies

    def for_sender(self, sender_id: str) -> "TenantContext":
        """Return a copy of this context stamped with a specific sender PSID."""
        return replace(self, sender_id=sender_id)


async def resolve_tenant(facebook_page_id: str) -> TenantContext:
    """Look up bot_settings by facebook_page_id and return a TenantContext.

    Uses a 60-second TTL cache to avoid repeated Supabase queries.

    Args:
        facebook_page_id: The Facebook page ID from the webhook entry.

    Returns:
        A TenantContext with shop_id and page_access_token populated.

    Raises:
        TenantNotFoundError: If no bot_settings row matches.
        TenantInactiveError: If the bot exists but is switched off (is_active=false).
    """
    # Check cache first
    cached = _tenant_cache.get(facebook_page_id)
    if cached is None:
        # Query Supabase
        try:
            supabase = await get_supabase()
            try:
                result = await supabase.table("bot_settings") \
                    .select("shop_id, page_access_token, is_active, allow_split_replies") \
                    .eq("page_id", facebook_page_id) \
                    .maybe_single() \
                    .execute()
            except Exception as col_err:
                # allow_split_replies migration not applied yet — the bot must
                # keep working, so fall back to the legacy column set.
                logger.warning(
                    f"bot_settings select with allow_split_replies failed ({col_err}) — "
                    "retrying without it. Run the allow_split_replies migration."
                )
                result = await supabase.table("bot_settings") \
                    .select("shop_id, page_access_token, is_active") \
                    .eq("page_id", facebook_page_id) \
                    .maybe_single() \
                    .execute()
        except Exception as e:
            logger.error(f"Supabase query failed for facebook_page_id={facebook_page_id}: {e}")
            raise TenantNotFoundError(f"DB error resolving tenant: {e}") from e

        if not result or not result.data:
            raise TenantNotFoundError(
                f"No bot_settings row for facebook_page_id={facebook_page_id}"
            )

        cached = {
            "shop_id": result.data["shop_id"],
            "page_access_token": result.data["page_access_token"],
            "is_active": bool(result.data.get("is_active")),
            "allow_split_replies": bool(result.data.get("allow_split_replies")),
        }
        _tenant_cache[facebook_page_id] = cached
        logger.info(
            f"Tenant resolved: facebook_page_id={facebook_page_id} → "
            f"shop_id={cached['shop_id']} (active={cached['is_active']})"
        )
    else:
        logger.debug(f"Tenant cache hit for facebook_page_id={facebook_page_id}")

    if not cached["is_active"]:
        raise TenantInactiveError(
            f"Bot for facebook_page_id={facebook_page_id} is inactive (is_active=false)"
        )

    return TenantContext(
        shop_id=cached["shop_id"],
        page_access_token=cached["page_access_token"],
        facebook_page_id=facebook_page_id,
        allow_split_replies=cached.get("allow_split_replies", False),
    )
