"""Handler for processing text messages from Facebook Messenger."""

import asyncio

from app.core.logging_config import get_logger
from app.core.tenant_context import TenantContext
from app.services.agent_service import agent_service
from app.services.memory_service import memory_service
from app.services.messaging_service import messaging_service
from app.services.persistence_service import persistence_service

logger = get_logger(__name__)


class TextHandler:
    """Handler for processing text-based messages using Agentic orchestration."""

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
                logger.info(f"[{sender_id}] 🙋 Human agent active — bot staying silent")
                return

            # Show typing indicator
            await messaging_service.send_typing_on(sender_id, access_token=tenant.page_access_token)

            # Let the agent handle the entire multi-turn logic
            reply = await agent_service.process(sender_id, message_text, image_urls=image_urls, tenant=tenant)

            # Empty reply = internal error already logged upstream. Errors are
            # NEVER surfaced to the user — stay silent and let them retry.
            if not reply or not reply.strip():
                logger.warning(f"[{sender_id}] Agent produced no reply — staying silent (error logged upstream)")
                return

            # Artificial human typing delay (e.g., 50 chars per sec, bounded 1.5s to 4s)
            delay = min(4.0, max(1.5, len(reply) / 50.0))
            await messaging_service.send_typing_on(sender_id, access_token=tenant.page_access_token)
            await asyncio.sleep(delay)

            # Send the final reply
            await messaging_service.send_message(
                recipient_id=sender_id,
                message_text=reply,
                access_token=tenant.page_access_token,
            )

            # Persist the bot's reply for the dashboard transcript
            persistence_service.log_message_bg(tenant, "bot", reply)

            logger.info(f"[{sender_id}] ✅ Reply sent ({len(reply)} chars)")

        except Exception as e:
            # Errors are logged only — NEVER sent to the user.
            logger.error(f"[{sender_id}] Text handler error: {e}", exc_info=True)


text_handler = TextHandler()
