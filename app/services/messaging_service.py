"""Service for sending messages to Facebook Messenger users."""

import httpx
from app.core.config import settings

GRAPH_API_URL = "https://graph.facebook.com/v18.0/me/messages"


class MessagingService:
    """Service for handling message sending to Facebook Messenger."""

    @staticmethod
    async def send_typing_on(recipient_id: str) -> None:
        """
        Send a typing indicator to show the bot is processing.

        Args:
            recipient_id: The Facebook user ID to show typing to
        """
        params = {"access_token": settings.facebook_page_access_token}
        payload = {
            "recipient": {"id": recipient_id},
            "sender_action": "typing_on",
        }

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                await client.post(GRAPH_API_URL, params=params, json=payload)
        except Exception as e:
            # Non-critical — don't fail the whole flow for a typing indicator
            print(f"Failed to send typing indicator: {e}")

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
        params = {"access_token": settings.facebook_page_access_token}
        payload = {
            "recipient": {"id": recipient_id},
            "message": {"text": message_text},
        }

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(GRAPH_API_URL, params=params, json=payload)
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

    @staticmethod
    async def send_image(recipient_id: str, image_url: str) -> bool:
        """
        Send an image via Facebook Messenger API.
        """
        params = {"access_token": settings.facebook_page_access_token}
        payload = {
            "recipient": {"id": recipient_id},
            "message": {
                "attachment": {
                    "type": "image",
                    "payload": {
                        "url": image_url,
                        "is_reusable": True
                    }
                }
            }
        }
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(GRAPH_API_URL, params=params, json=payload)
                if response.status_code == 200:
                    print(f"Image sent successfully to {recipient_id}")
                    return True
                else:
                    print(f"Failed to send image. Status: {response.status_code}, Response: {response.text}")
                    return False
        except Exception as e:
            print(f"Error sending image: {str(e)}")
            return False


messaging_service = MessagingService()
