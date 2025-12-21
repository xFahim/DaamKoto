"""Service layer for Facebook Messenger webhook processing."""

from app.core.config import settings
from app.schemas.facebook import FacebookWebhookPayload
from app.services.handlers.message_router import message_router


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
    async def process_webhook_event(payload: FacebookWebhookPayload) -> None:
        """
        Process incoming Facebook webhook events.

        Routes messages to appropriate handlers based on message type (text/image).

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
            # Extract page_id from entry
            page_id = entry.id

            print(f"Entry ID: {entry.id}")
            print(f"Entry Time: {entry.time}")
            print(f"Number of messaging events: {len(entry.messaging)}")
            print()

            for messaging in entry.messaging:
                sender_id = messaging.sender.id
                print(f"  Sender ID: {sender_id}")
                print(f"  Recipient ID: {messaging.recipient.id}")
                print(f"  Timestamp: {messaging.timestamp}")

                # Process message events
                if messaging.message:
                    message_dict = messaging.message.model_dump()
                    print(f"  Message ID: {message_dict.get('mid')}")
                    print(f"  Message Text: {message_dict.get('text')}")
                    if message_dict.get("attachments"):
                        print(f"  Attachments: {len(message_dict['attachments'])}")

                    # Route message to appropriate handler (text or image)
                    await message_router.route_message(
                        sender_id=sender_id,
                        message=message_dict,
                        page_id=page_id,
                    )

                # Log other event types
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
