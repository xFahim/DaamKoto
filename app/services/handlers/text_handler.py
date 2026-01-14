"""Handler for processing text messages from Facebook Messenger."""

from app.services.rag_service import rag_service
from app.services.messaging_service import messaging_service


class TextHandler:
    """Handler for processing text-based messages."""

    @staticmethod
    async def process(
        sender_id: str,
        message_text: str,
        page_id: str,
    ) -> None:
        """
        Process a text message and send an AI-generated response.

        Args:
            sender_id: The Facebook user ID who sent the message
            message_text: The text content of the message
            page_id: The Facebook page ID
        """
        try:
            # Generate AI response using RAG service
            response_text = await rag_service.generate_response(
                user_query=message_text,
                page_id=page_id,
            )
            await messaging_service.send_message(
                recipient_id=sender_id,
                message_text=response_text,
            )
        except Exception as e:
            print(f"Error generating AI response: {str(e)}")
            # Fallback to a simple error message
            await messaging_service.send_message(
                recipient_id=sender_id,
                message_text=(
                    "Sorry, I'm having trouble processing your message right now! "
                    "Please try again later!"
                ),
            )


text_handler = TextHandler()



