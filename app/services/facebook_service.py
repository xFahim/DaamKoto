"""Service layer for Facebook Messenger webhook processing."""

from typing import Any
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
    def process_webhook_event(payload: FacebookWebhookPayload) -> None:
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
