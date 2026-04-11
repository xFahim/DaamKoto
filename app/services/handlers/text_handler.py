"""Handler for processing text messages from Facebook Messenger."""

from app.core.logging_config import get_logger
from app.services.agent_service import agent_service
from app.services.messaging_service import messaging_service

logger = get_logger(__name__)


class TextHandler:
    """Handler for processing text-based messages using Agentic orchestration."""

    async def process(
        self,
        sender_id: str,
        message_text: str,
        page_id: str,
        image_urls: list[str] = None
    ) -> None:
        """
        Pass the message to the central Agent Service, get the reply, and send it.
        """
        try:
            import asyncio
            # Show typing indicator immediately
            await messaging_service.send_typing_on(sender_id)

            # Let the agent handle the entire multi-turn logic
            reply = await agent_service.process(sender_id, message_text, image_urls=image_urls, page_id=page_id)

            # Artificial human typing delay (e.g., 50 chars per sec, bounded 1.5s to 4s)
            delay = min(4.0, max(1.5, len(reply) / 50.0))
            await messaging_service.send_typing_on(sender_id)
            await asyncio.sleep(delay)

            # Step 3: Send the final reply
            await messaging_service.send_message(
                recipient_id=sender_id,
                message_text=reply,
            )

            logger.info(f"[{sender_id}] ✅ Reply sent ({len(reply)} chars)")

        except Exception as e:
            logger.error(f"[{sender_id}] Text handler error: {e}", exc_info=True)
            await messaging_service.send_message(
                recipient_id=sender_id,
                message_text=(
                    "Sorry, I'm having trouble processing your message right now! "
                    "Please try again later!"
                ),
            )


text_handler = TextHandler()
