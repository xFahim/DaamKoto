"""Short-term per-sender conversation memory using TTLCache.

Provider-agnostic: stores history as plain dicts internally,
with converters for Gemini and OpenAI formats.
"""

import json
from cachetools import TTLCache
from app.core.config import settings

# TTLCache automatically ejects idle users after exactly `conversation_ttl` seconds (e.g. 5 mins)
_cache: TTLCache = TTLCache(maxsize=1000, ttl=settings.conversation_ttl)


def _content_to_dict(content) -> dict:
    """Convert a Gemini types.Content object OR an already-normalised dict to our internal dict format."""
    # Already a dict — pass through
    if isinstance(content, dict):
        return content

    # Gemini types.Content object → normalise
    parts = []
    for p in content.parts:
        if p.text:
            parts.append({"type": "text", "text": p.text})
        elif p.function_call:
            parts.append({
                "type": "function_call",
                "name": p.function_call.name,
                "args": dict(p.function_call.args) if p.function_call.args else {},
            })
        elif p.function_response:
            parts.append({
                "type": "function_response",
                "name": p.function_response.name,
                "response": dict(p.function_response.response) if p.function_response.response else {},
            })
        elif hasattr(p, 'inline_data') and p.inline_data:
            parts.append({"type": "inline_data", "mime_type": p.inline_data.mime_type})
        elif hasattr(p, 'file_data') and p.file_data:
            parts.append({"type": "file_data", "uri": p.file_data.file_uri, "mime_type": p.file_data.mime_type})
        else:
            # Fallback for uri-based parts  
            try:
                parts.append({"type": "file_data", "uri": p.uri, "mime_type": p.mime_type})
            except Exception:
                parts.append({"type": "text", "text": "[unsupported part]"})

    return {"role": content.role, "parts": parts}


def _dict_to_gemini(d: dict):
    """Convert our internal dict back to a Gemini types.Content object."""
    from google.genai import types

    parts = []
    for p in d["parts"]:
        ptype = p.get("type", "text")
        if ptype == "text":
            parts.append(types.Part.from_text(text=p["text"]))
        elif ptype == "function_call":
            parts.append(types.Part.from_function_call(name=p["name"], args=p["args"]))
        elif ptype == "function_response":
            parts.append(types.Part.from_function_response(name=p["name"], response=p["response"]))
        elif ptype == "file_data":
            parts.append(types.Part.from_uri(uri=p["uri"], mime_type=p.get("mime_type", "image/jpeg")))
        else:
            parts.append(types.Part.from_text(text="[unsupported]"))

    return types.Content(role=d["role"], parts=parts)


def _dict_to_openai(d: dict) -> dict | None:
    """Convert our internal dict to an OpenAI chat message dict.
    
    Returns None if the message should be skipped (e.g. unsupported part types).
    """
    role = d["role"]
    parts = d["parts"]

    # --- user message with function_response(s) → tool messages ---
    if role == "user":
        func_responses = [p for p in parts if p.get("type") == "function_response"]
        if func_responses:
            # Return a list of tool messages
            msgs = []
            for fr in func_responses:
                msgs.append({
                    "role": "tool",
                    "tool_call_id": fr.get("tool_call_id", fr["name"]),
                    "content": json.dumps(fr["response"]),
                })
            return msgs

        # Normal user text message
        text_parts = [p["text"] for p in parts if p.get("type") == "text"]
        content_parts = []
        for p in parts:
            if p.get("type") == "text":
                content_parts.append({"type": "text", "text": p["text"]})
            elif p.get("type") == "file_data":
                content_parts.append({"type": "image_url", "image_url": {"url": p["uri"]}})

        if len(content_parts) == 1 and content_parts[0]["type"] == "text":
            return {"role": "user", "content": content_parts[0]["text"]}
        elif content_parts:
            return {"role": "user", "content": content_parts}
        else:
            return {"role": "user", "content": "[empty]"}

    # --- model message ---
    if role == "model":
        text_parts = [p["text"] for p in parts if p.get("type") == "text"]
        func_calls = [p for p in parts if p.get("type") == "function_call"]

        msg = {"role": "assistant", "content": "\n".join(text_parts) if text_parts else None}

        if func_calls:
            msg["tool_calls"] = []
            for fc in func_calls:
                msg["tool_calls"].append({
                    "id": fc.get("tool_call_id", fc["name"]),
                    "type": "function",
                    "function": {
                        "name": fc["name"],
                        "arguments": json.dumps(fc["args"]),
                    }
                })

        return msg

    return None


class MemoryService:

    def get_history(self, sender_id: str) -> list[dict]:
        """Return the internal dict-based conversation history."""
        return list(_cache.get(sender_id, []))

    def get_gemini_history(self, sender_id: str) -> list:
        """Return history converted to Gemini types.Content objects."""
        return [_dict_to_gemini(d) for d in self.get_history(sender_id)]

    def get_openai_history(self, sender_id: str) -> list[dict]:
        """Return history converted to OpenAI message format."""
        messages = []
        for d in self.get_history(sender_id):
            converted = _dict_to_openai(d)
            if converted is None:
                continue
            if isinstance(converted, list):
                messages.extend(converted)
            else:
                messages.append(converted)
        return messages

    def append_content(self, sender_id: str, content) -> None:
        """Append a Gemini Content block or internal dict and ensure max message limits are respected."""
        history = list(_cache.get(sender_id, []))

        entry = _content_to_dict(content)
        history.append(entry)

        # Increased max messages to prevent aggressive clipping during long order flows
        max_messages = 30 

        if len(history) > max_messages:
            history = history[-max_messages:]

            # Must start with a user message (Gemini requirement, good practice for OpenAI too)
            while history and history[0].get("role") != "user":
                history.pop(0)

            # Must not start with an orphan function_response
            # (mirrors the original Gemini-specific logic)
            while history and any(
                p.get("type") == "function_response" for p in history[0].get("parts", [])
            ):
                history.pop(0)  # Drop orphan response
                # Drop subsequent model text to reach next clean user text
                while history and history[0].get("role") != "user":
                    history.pop(0)

        _cache[sender_id] = history

    def replace_with_summary(self, sender_id: str, keep_last_n: int, summary_text: str) -> None:
        """Replace older messages with a summary string to save tokens."""
        history = list(_cache.get(sender_id, []))
        if len(history) <= keep_last_n:
            return
            
        # Keep the last n messages
        recent = history[-keep_last_n:]
        
        # Ensure the first message in the recent list is a 'user' message
        while recent and recent[0].get("role") != "user":
            recent.pop(0)
            
        while recent and any(
            p.get("type") == "function_response" for p in recent[0].get("parts", [])
        ):
            recent.pop(0)
            while recent and recent[0].get("role") != "user":
                recent.pop(0)

        # Create summary message
        summary_msg = {
            "role": "user",
            "parts": [{"type": "text", "text": f"[System Context Summary: {summary_text}]"}]
        }
        
        # New history is: Summary -> Recent N messages
        _cache[sender_id] = [summary_msg] + recent

    def clear_history(self, sender_id: str) -> None:
        """Manually wipe history if needed (e.g., after order completion)."""
        if sender_id in _cache:
            del _cache[sender_id]

memory_service = MemoryService()
