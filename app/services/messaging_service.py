"""Service for sending messages to Facebook Messenger users."""

import httpx
from app.core.config import settings


class MessagingService:
    """Service for handling message sending to Facebook Messenger."""

    @staticmethod
    async def send_message(recipient_id: str, message_text: str) -> bool:
        """
        Send a text message to a Facebook Messenger user.

        Args:
            recipient_id: The Facebook user ID to send the message to
            message_text: The text content of the message

        Returns:
            True if the message was sent successfully, False otherwise
        """
        url = "https://graph.facebook.com/v18.0/me/messages"
        params = {"access_token": settings.facebook_page_access_token}
        payload = {
            "recipient": {"id": recipient_id},
            "message": {"text": message_text},
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(url, params=params, json=payload)
                if response.status_code == 200:
                    print(f"Message sent successfully to {recipient_id}")
                    return True
                else:
                    print(
                        f"Failed to send message. Status: {response.status_code}, "
                        f"Response: {response.text}"
                    )
                    return False
        except Exception as e:
            print(f"Error sending message: {str(e)}")
            return False


messaging_service = MessagingService()

