"""Handler for processing image messages from Facebook Messenger."""

from typing import Any
from app.services.messaging_service import messaging_service
from app.services.agent_service import agent_service


class ImageHandler:
    """Handler for processing image-based messages using Agentic orchestration."""

    @staticmethod
    async def process(
        sender_id: str,
        attachments: list[dict[str, Any]],
        page_id: str,
    ) -> None:
        """
        Process an image message by providing it directly to the Agent.
        """
        try:
            # Show typing indicator immediately
            await messaging_service.send_typing_on(sender_id)

            # Extract image URL from attachments
            image_url = ImageHandler._extract_image_url(attachments)
            if not image_url:
                await messaging_service.send_message(
                    recipient_id=sender_id,
                    message_text="I couldn't process the image. Please try sending it again.",
                )
                return

            # Native Agent Integration: Feed the image straight into the brain's memory
            reply = await agent_service.process(
                sender_id=sender_id,
                message_text="Can you identify this product? I'm interested in it.",
                image_url=image_url
            )
            
            await messaging_service.send_message(
                recipient_id=sender_id,
                message_text=reply,
            )
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            await messaging_service.send_message(
                recipient_id=sender_id,
                message_text=(
                    "Sorry, I'm having trouble analyzing the image right now! "
                    "Please try again later!"
                ),
            )

    @staticmethod
    def _extract_image_url(attachments: list[dict[str, Any]]) -> str | None:
        """
        Extract the image URL from Facebook attachment payload.

        Args:
            attachments: List of attachment dictionaries from Facebook

        Returns:
            The image URL if found, None otherwise
        """
        if not attachments:
            return None

        for attachment in attachments:
            attachment_type = attachment.get("type")
            if attachment_type == "image":
                payload = attachment.get("payload", {})
                image_url = payload.get("url")
                if image_url:
                    return image_url

        return None


image_handler = ImageHandler()



