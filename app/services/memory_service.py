"""Short-term per-sender conversation memory using TTLCache."""

from cachetools import TTLCache
from google.genai import types
from app.core.config import settings

# TTLCache automatically ejects idle users after exactly `conversation_ttl` seconds (e.g. 5 mins)
_cache: TTLCache = TTLCache(maxsize=1000, ttl=settings.conversation_ttl)

class MemoryService:

    def get_history(self, sender_id: str) -> list[types.Content]:
        """Return the structured conversation history for Gemini."""
        return list(_cache.get(sender_id, []))

    def append_content(self, sender_id: str, content: types.Content) -> None:
        """Append a Gemini Content block and ensure max message limits are respected."""
        history = list(_cache.get(sender_id, []))
        history.append(content)
        
        # Increased max messages to prevent aggressive clipping during long order flows
        max_messages = 30 
        
        if len(history) > max_messages:
            history = history[-max_messages:]
            
            # Gemini strictly forbids starting a chat sequence with a model output
            while history and history[0].role != "user":
                history.pop(0)
                
            # Gemini strictly requires a function_response to immediately follow a function_call.
            # If we sliced off the model's function_call, the array now starts with a dangling 
            # user function_response. We must drop it to find a clean user text message.
            while history and any(getattr(p, 'function_response', None) for p in history[0].parts):
                history.pop(0)  # Drop orphan response
                # Drop subsequent model text to reach next clean user text
                while history and history[0].role != "user":
                    history.pop(0)

        _cache[sender_id] = history
        
    def clear_history(self, sender_id: str) -> None:
        """Manually wipe history if needed (e.g., after order completion)."""
        if sender_id in _cache:
            del _cache[sender_id]

memory_service = MemoryService()
