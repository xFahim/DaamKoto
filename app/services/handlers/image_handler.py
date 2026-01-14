"""Handler for processing image messages from Facebook Messenger."""

from typing import Any
from app.services.image_service import image_service
from app.services.messaging_service import messaging_service
from app.services.rag_service import rag_service


class ImageHandler:
    """Handler for processing image-based messages."""

    @staticmethod
    async def process(
        sender_id: str,
        attachments: list[dict[str, Any]],
        page_id: str,
    ) -> None:
        """
        Process an image message, analyze it, and send a response.

        Args:
            sender_id: The Facebook user ID who sent the message
            attachments: List of attachment dictionaries from Facebook
            page_id: The Facebook page ID
        """
        try:
            # Extract image URL from attachments
            image_url = ImageHandler._extract_image_url(attachments)
            if not image_url:
                await messaging_service.send_message(
                    recipient_id=sender_id,
                    message_text="I couldn't process the image. Please try sending it again.",
                )
                return

            # Analyze the image using image service
            image_description = await image_service.describe_image(image_url)
            print(f"Image analyzed: {image_description}")

            # Use RAG service to search inventory based on image description
            # We pass image_url so RAG can use CLIP image embeddings for retrieval
            # while keeping image_description for the context prompt.
            response_text = await rag_service.generate_response(
                user_query=image_description,
                page_id=page_id,
                image_url=image_url,
            )
            await messaging_service.send_message(
                recipient_id=sender_id,
                message_text=response_text,
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



