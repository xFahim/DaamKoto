"""Router for categorizing and routing messages to appropriate handlers."""

from typing import Any
from app.services.handlers.text_handler import text_handler
from app.services.handlers.image_handler import image_handler
from app.services.messaging_service import messaging_service


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
            await text_handler.process(
                sender_id=sender_id,
                message_text=message["text"],
                page_id=page_id,
            )
            return

        # Check if message has image attachments
        attachments = message.get("attachments")
        if attachments:
            # Check if any attachment is an image
            has_image = any(
                att.get("type") == "image" for att in attachments if isinstance(att, dict)
            )
            if has_image:
                await image_handler.process(
                    sender_id=sender_id,
                    attachments=attachments,
                    page_id=page_id,
                )
                return

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

