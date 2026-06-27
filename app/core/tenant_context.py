"""Tenant context resolution for multi-tenant request handling.

Resolves a Facebook page ID to an internal shop_id + access token
by querying the bot_settings table in Supabase. The result is cached
in-memory for 60 seconds to avoid repeated DB hits on every message.
"""

from dataclasses import dataclass, field
from cachetools import TTLCache
from app.core.dependencies import get_supabase
from app.core.logging_config import get_logger

logger = get_logger(__name__)

# Cache tenant lookups for 60s — avoids hitting Supabase on every webhook message
_tenant_cache: TTLCache = TTLCache(maxsize=200, ttl=60)


class TenantNotFoundError(Exception):
    """Raised when no bot_settings row matches the given Facebook page ID."""
    pass


@dataclass
class TenantContext:
    """Per-request tenant state resolved from bot_settings.

    Resolved once at webhook entry and threaded through the entire
    message processing pipeline. Tools never see shop_id — it's
    injected server-side in the ReAct execution bridge.
    """
    shop_id: str                    # Internal UUID from bot_settings
    page_access_token: str          # Facebook Graph API token for this page
    facebook_page_id: str           # The numeric Facebook page ID (entry.id)
    sender_id: str = ""             # Messenger PSID — stamped per-message


async def resolve_tenant(facebook_page_id: str) -> TenantContext:
    """Look up bot_settings by facebook_page_id and return a TenantContext.

    Uses a 60-second TTL cache to avoid repeated Supabase queries.

    Args:
        facebook_page_id: The Facebook page ID from the webhook entry.

    Returns:
        A TenantContext with shop_id and page_access_token populated.

    Raises:
        TenantNotFoundError: If no bot_settings row matches.
    """
    # Check cache first
    cached = _tenant_cache.get(facebook_page_id)
    if cached:
        logger.debug(f"Tenant cache hit for facebook_page_id={facebook_page_id}")
        return TenantContext(
            shop_id=cached["shop_id"],
            page_access_token=cached["page_access_token"],
            facebook_page_id=facebook_page_id,
        )

    # Query Supabase
    try:
        result = get_supabase().table("bot_settings") \
            .select("shop_id, facebook_page_access_token") \
            .eq("facebook_page_id", facebook_page_id) \
            .maybe_single() \
            .execute()
    except Exception as e:
        logger.error(f"Supabase query failed for facebook_page_id={facebook_page_id}: {e}")
        raise TenantNotFoundError(f"DB error resolving tenant: {e}") from e

    if not result or not result.data:
        raise TenantNotFoundError(
            f"No bot_settings row for facebook_page_id={facebook_page_id}"
        )

    row = result.data
    shop_id = row["shop_id"]
    token = row["facebook_page_access_token"]

    # Cache it
    _tenant_cache[facebook_page_id] = {
        "shop_id": shop_id,
        "page_access_token": token,
    }

    logger.info(f"Tenant resolved: facebook_page_id={facebook_page_id} → shop_id={shop_id}")

    return TenantContext(
        shop_id=shop_id,
        page_access_token=token,
        facebook_page_id=facebook_page_id,
    )
