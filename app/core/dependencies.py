"""Shared external client instances.

Centralizes initialization of Supabase and Google GenAI clients
so they can be imported by any service without circular dependencies.
"""

from supabase import create_client, Client as SupabaseClient
from google import genai
from app.core.config import settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)

# ── Google GenAI ─────────────────────────────────────────────────────────
# Shared client used for both LLM generation (agent) and embedding generation (webhook/RAG)
genai_client: genai.Client = genai.Client(api_key=settings.gemini_api_key)
logger.info("Google GenAI client initialized")

# ── Supabase ─────────────────────────────────────────────────────────────
# Service-role client for backend operations (embedding writes, token lookups, etc.)
# Initialized lazily via get_supabase() to avoid crashing on import
# when placeholder credentials are set in .env during local dev.
_supabase_client: SupabaseClient | None = None


def get_supabase() -> SupabaseClient:
    """Return the Supabase client, initializing on first call."""
    global _supabase_client
    if _supabase_client is None:
        _supabase_client = create_client(settings.supabase_url, settings.supabase_service_role_key)
        logger.info("Supabase client initialized")
    return _supabase_client
