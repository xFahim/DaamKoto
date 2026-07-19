"""Handler for processing text messages from Facebook Messenger."""

import asyncio
import random

from app.core.logging_config import get_logger
from app.core.tenant_context import TenantContext
from app.services.agent_service import agent_service, SPLIT_TOKEN
from app.services.memory_service import memory_service
from app.services.messaging_service import messaging_service
from app.services.persistence_service import persistence_service
from app.services.scope_guard import scope_guard

logger = get_logger(__name__)

# Human-like typing delay: the first TYPING_FREE_CHARS are "free" (short
# replies go out immediately after the debounce wait), then every
# TYPING_CHARS_PER_SEC characters add one second, capped so long replies
# don't stall the conversation.
TYPING_FREE_CHARS = 12
TYPING_CHARS_PER_SEC = 15
TYPING_DELAY_CAP = 10.0
# FB expires typing_on after ~20s — refresh well inside that while waiting.
TYPING_REFRESH_INTERVAL = 6.0

# Minimum pause between consecutive bubbles of a split reply, so a
# double-send never lands as an instant robotic burst.
SPLIT_GAP_MIN = 0.9
SPLIT_GAP_MAX = 1.8

# Never send more than this many bubbles per reply, whatever the model does.
MAX_SPLIT_PARTS = 3


def typing_delay_for(text: str) -> float:
    """Seconds a human would plausibly take to type this message."""
    return min(TYPING_DELAY_CAP, max(0.0, (len(text) - TYPING_FREE_CHARS) / TYPING_CHARS_PER_SEC))


class TextHandler:
    """Handler for processing text-based messages using Agentic orchestration."""

    @staticmethod
    async def _type_for(seconds: float, sender_id: str, access_token: str) -> None:
        """Wait `seconds` while keeping the typing indicator alive."""
        remaining = seconds
        while remaining > 0:
            await messaging_service.send_typing_on(sender_id, access_token=access_token)
            step = min(TYPING_REFRESH_INTERVAL, remaining)
            await asyncio.sleep(step)
            remaining -= step

    async def process(
        self,
        sender_id: str,
        message_text: str,
        tenant: TenantContext,
        image_urls: list[str] = None
    ) -> None:
        """
        Pass the message to the central Agent Service, get the reply, and send it.
        """
        try:
            # Persist the user's message for the dashboard transcript (fire-and-forget)
            transcript_text = message_text
            if image_urls:
                suffix = "\n".join(f"[image] {u}" for u in image_urls)
                transcript_text = f"{transcript_text}\n{suffix}".strip()
            persistence_service.log_message_bg(tenant, "customer", transcript_text)

            # Human takeover: if an agent owns this thread from the dashboard
            # (thread_status = 'human_active'), the bot logs and stays silent.
            # Memory is dropped so the handback rehydrates from the DB and the
            # bot sees what the human agent said while it was muted.
            if await persistence_service.is_human_active(tenant.shop_id, sender_id):
                memory_service.clear_history(f"{tenant.shop_id}:{sender_id}")
                scope_guard.reset(f"{tenant.shop_id}:{sender_id}")
                logger.info(f"[{sender_id}] 🙋 Human agent active — bot staying silent")
                return

            # Show typing indicator
            await messaging_service.send_typing_on(sender_id, access_token=tenant.page_access_token)

            # Let the agent handle the entire multi-turn logic
            reply = await agent_service.process(sender_id, message_text, image_urls=image_urls, tenant=tenant)

            # Empty reply = internal error already logged upstream, OR a
            # deliberate silence decision ([SILENT] scope rule). Errors are
            # NEVER surfaced to the user — stay quiet.
            if not reply or not reply.strip():
                logger.info(f"[{sender_id}] No reply to send (error or deliberate silence)")
                return

            # Split into bubbles: only when the shop enabled the feature.
            # When it's off, a stray marker must never leak to the customer.
            if tenant.allow_split_replies and SPLIT_TOKEN in reply:
                parts = [p.strip() for p in reply.split(SPLIT_TOKEN) if p.strip()]
                parts = parts[:MAX_SPLIT_PARTS]
            else:
                parts = [reply.replace(SPLIT_TOKEN, " ").strip()]

            for i, part in enumerate(parts):
                # Typing time scales with what's being "typed"; between bubbles
                # there's always at least a small human gap.
                delay = typing_delay_for(part)
                if i > 0:
                    delay = max(delay, random.uniform(SPLIT_GAP_MIN, SPLIT_GAP_MAX))
                if delay > 0:
                    await self._type_for(delay, sender_id, tenant.page_access_token)

                sent = await messaging_service.send_message(
                    recipient_id=sender_id,
                    message_text=part,
                    access_token=tenant.page_access_token,
                )
                if not sent:
                    logger.error(f"[{sender_id}] Send failed on bubble {i + 1}/{len(parts)} — stopping")
                    break

                # Persist each bubble as its own row so the dashboard mirrors
                # exactly what the customer saw.
                persistence_service.log_message_bg(tenant, "bot", part)

            logger.info(
                f"[{sender_id}] ✅ Reply sent ({len(reply)} chars, {len(parts)} bubble(s))"
            )

        except Exception as e:
            # Errors are logged only — NEVER sent to the user.
            logger.error(f"[{sender_id}] Text handler error: {e}", exc_info=True)


text_handler = TextHandler()
