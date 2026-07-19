"""Fire-and-forget LLM token-usage logging to the Supabase llm_usage table.

One row per LLM run — agent replies (kind='chat') and background history
summarizations (kind='summary'). Never blocks or fails the reply path: if
the table doesn't exist yet (migration not run) the first failure logs a
warning with the hint, later ones only debug.
"""

import asyncio

from app.core.dependencies import get_supabase
from app.core.logging_config import get_logger

logger = get_logger(__name__)

_warned_once = False


class UsageService:

    def log_bg(
        self,
        *,
        shop_id: str,
        sender_psid: str,
        provider: str,
        model: str,
        kind: str,
        prompt_tokens: int,
        completion_tokens: int,
        turns: int = 0,
        tools_used: list[str] | None = None,
        message_chars: int = 0,
        reply_chars: int = 0,
        latency_ms: int = 0,
    ) -> None:
        """Queue one usage row; returns immediately."""
        row = {
            "shop_id": shop_id,
            "sender_psid": sender_psid,
            "kind": kind,
            "provider": provider,
            "model": model,
            "prompt_tokens": int(prompt_tokens or 0),
            "completion_tokens": int(completion_tokens or 0),
            "total_tokens": int(prompt_tokens or 0) + int(completion_tokens or 0),
            "turns": int(turns or 0),
            "tools_used": tools_used or [],
            "message_chars": int(message_chars or 0),
            "reply_chars": int(reply_chars or 0),
            "latency_ms": int(latency_ms or 0),
        }
        task = asyncio.create_task(self._insert(row))
        task.add_done_callback(self._on_done)

    @staticmethod
    async def _insert(row: dict) -> None:
        supabase = await get_supabase()
        await supabase.table("llm_usage").insert(row).execute()

    @staticmethod
    def _on_done(task: asyncio.Task) -> None:
        global _warned_once
        if task.cancelled():
            return
        exc = task.exception()
        if exc:
            if not _warned_once:
                _warned_once = True
                logger.warning(
                    f"llm_usage insert failed ({exc}) — run the 20260719_llm_usage.sql "
                    "migration. Further failures logged at debug."
                )
            else:
                logger.debug(f"llm_usage insert failed: {exc}")


usage_service = UsageService()
