"""Lightweight mid → text map for resolving Messenger reply-to references.

Stores both incoming user messages and outgoing bot replies so the agent
can see what a user is replying to when they long-press a message.
"""

from cachetools import TTLCache

# Keep mids for 1 hour — more than enough for active conversations.
# maxsize=5000 ≈ a few KB, negligible memory.
_mid_cache: TTLCache = TTLCache(maxsize=5000, ttl=3600)


def store_mid(mid: str, text: str) -> None:
    """Record a message id → text mapping."""
    if mid and text:
        _mid_cache[mid] = text


def resolve_mid(mid: str) -> str | None:
    """Look up the text for a given message id. Returns None if expired/unknown."""
    return _mid_cache.get(mid)
