"""Service layer for Facebook Messenger webhook processing."""

import httpx
from typing import Any
from app.core.config import settings
from app.schemas.facebook import FacebookWebhookPayload


class FacebookService:
    """Service for handling Facebook Messenger webhook events."""

    @staticmethod
    def verify_webhook(
        mode: str, token: str, challenge: str, verify_token: str
    ) -> str | None:
        """
        Verify the webhook subscription request from Facebook.

        Args:
            mode: The mode parameter from the verification request
            token: The verify token from the verification request
            challenge: The challenge string from Facebook
            verify_token: The expected verify token from configuration

        Returns:
            The challenge string if verification succeeds, None otherwise
        """
        if mode == "subscribe" and token == verify_token:
            return challenge
        return None

    @staticmethod
    async def send_message(recipient_id: str, message_text: str) -> bool:
        """
        Send a message to a Facebook Messenger user.

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

    @staticmethod
    async def process_webhook_event(payload: FacebookWebhookPayload) -> None:
        """
        Process incoming Facebook webhook events.

        Args:
            payload: The validated Facebook webhook payload
        """
        print("=" * 50)
        print("Facebook Webhook Event Received")
        print("=" * 50)
        print(f"Object: {payload.object}")
        print(f"Number of entries: {len(payload.entry)}")
        print()

        for entry in payload.entry:
            print(f"Entry ID: {entry.id}")
            print(f"Entry Time: {entry.time}")
            print(f"Number of messaging events: {len(entry.messaging)}")
            print()

            for messaging in entry.messaging:
                print(f"  Sender ID: {messaging.sender.id}")
                print(f"  Recipient ID: {messaging.recipient.id}")
                print(f"  Timestamp: {messaging.timestamp}")

                if messaging.message:
                    print(f"  Message ID: {messaging.message.mid}")
                    print(f"  Message Text: {messaging.message.text}")
                    if messaging.message.attachments:
                        print(f"  Attachments: {len(messaging.message.attachments)}")

                    # Send automatic reply for text messages
                    if messaging.message.text:
                        reply_text = (
                            f"I received your message: {messaging.message.text}"
                        )
                        await FacebookService.send_message(
                            recipient_id=messaging.sender.id,
                            message_text=reply_text,
                        )

                if messaging.postback:
                    print(f"  Postback: {messaging.postback}")

                if messaging.delivery:
                    print(f"  Delivery: {messaging.delivery}")

                if messaging.read:
                    print(f"  Read: {messaging.read}")

                print()

        print("=" * 50)
        print()


facebook_service = FacebookService()
