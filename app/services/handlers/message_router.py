"""Router for categorizing and routing messages to appropriate handlers."""

from typing import Any
from app.services.batching_service import message_batcher
from app.services.handlers.image_handler import image_handler
from app.services.messaging_service import messaging_service
from app.services.input_guard import input_guard
from app.core.config import settings


class MessageRouter:
    """Router that categorizes messages and routes them to appropriate handlers."""

    @staticmethod
    async def route_message(
        sender_id: str,
        message: dict[str, Any] | None,
        page_id: str,
    ) -> None:
        """
        Categorize a message and route it to the appropriate handler.

        Args:
            sender_id: The Facebook user ID who sent the message
            message: The message object from Facebook webhook
            page_id: The Facebook page ID
        """
        if not message:
            return

        # Check if message has text
        if message.get("text"):
            status, payload = input_guard.check(sender_id, message["text"])

            if status == "ok":
                await message_batcher.add_message(
                    sender_id=sender_id,
                    text=payload,
                    page_id=page_id,
                )
            elif status == "reject":
                if payload == "too_long":
                    await messaging_service.send_message(
                        recipient_id=sender_id,
                        message_text=(
                            f"Please keep your message under {settings.max_message_length} characters "
                            "so I can understand you better!"
                        ),
                    )
                elif payload == "rate_limited":
                    await messaging_service.send_message(
                        recipient_id=sender_id,
                        message_text=(
                            "You're sending messages too fast! "
                            "Take a moment and try again."
                        ),
                    )

        # Check if message has image attachments
        attachments = message.get("attachments")
        if attachments:
            # We import here locally to avoid circular dependencies if any
            from app.services.handlers.image_handler import ImageHandler
            image_url = ImageHandler._extract_image_url(attachments)
            if image_url:
                await message_batcher.add_message(
                    sender_id=sender_id,
                    page_id=page_id,
                    image_url=image_url
                )

        # If message doesn't match any category, send a default response
        await MessageRouter._send_unsupported_message(sender_id)

    @staticmethod
    async def _send_unsupported_message(sender_id: str) -> None:
        """
        Send a message for unsupported message types.

        Args:
            sender_id: The Facebook user ID to send the message to
        """
        await messaging_service.send_message(
            recipient_id=sender_id,
            message_text=(
                "I can help you with text messages or product images! "
                "Please send me a message or an image of a product you're looking for."
            ),
        )


message_router = MessageRouter()

