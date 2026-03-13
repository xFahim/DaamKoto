"""Short-term per-sender conversation memory using TTLCache."""

from cachetools import TTLCache
from app.core.config import settings

# Not thread-safe on its own, but asyncio runs on a single thread so no lock needed.
_cache: TTLCache = TTLCache(maxsize=1000, ttl=settings.conversation_ttl)


class MemoryService:

    def get_history(self, sender_id: str) -> str:
        """Return formatted conversation history, or '' if none exists."""
        turns = _cache.get(sender_id, [])
        if not turns:
            return ""
        return "\n".join(
            f"{'User' if t['role'] == 'user' else 'Bot'}: {t['text']}"
            for t in turns
        )

    def save_turn(self, sender_id: str, user_message: str, bot_reply: str) -> None:
        """Append a user+bot turn and keep only the last N turns."""
        turns = list(_cache.get(sender_id, []))
        turns.append({"role": "user", "text": user_message})
        turns.append({"role": "bot", "text": bot_reply})
        max_entries = settings.conversation_max_turns * 2
        _cache[sender_id] = turns[-max_entries:]


memory_service = MemoryService()
