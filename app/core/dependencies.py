"""Shared external client instances.

Centralizes initialization of Supabase and Google GenAI clients
so they can be imported by any service without circular dependencies.
"""

import asyncio

from supabase import acreate_client, AsyncClient as SupabaseClient
from google import genai
from app.core.config import settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)

# ── Google GenAI ─────────────────────────────────────────────────────────
# Shared client used for both LLM generation (agent) and embedding generation (webhook/RAG)
genai_client: genai.Client = genai.Client(api_key=settings.gemini_api_key)
logger.info("Google GenAI client initialized")

# ── Supabase ─────────────────────────────────────────────────────────────
# Async service-role client for backend operations (embedding writes, token
# lookups, orders, etc.). The sync client blocks the event loop on every
# query — with one process serving all tenants, that stalls every active
# conversation, so all DB access goes through this async client.
# Initialized lazily to avoid crashing on import when placeholder
# credentials are set in .env during local dev.
_supabase_client: SupabaseClient | None = None
_init_lock = asyncio.Lock()


async def get_supabase() -> SupabaseClient:
    """Return the shared async Supabase client, initializing on first call."""
    global _supabase_client
    if _supabase_client is None:
        async with _init_lock:
            if _supabase_client is None:
                _supabase_client = await acreate_client(
                    settings.supabase_url, settings.supabase_service_role_key
                )
                logger.info("Supabase async client initialized")
    return _supabase_client
