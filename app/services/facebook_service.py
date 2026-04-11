"""Service layer for Facebook Messenger webhook processing."""

from app.core.config import settings
from app.core.logging_config import get_logger
from app.schemas.facebook import FacebookWebhookPayload
from app.services.handlers.message_router import message_router

logger = get_logger(__name__)


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
            logger.info("Webhook verification succeeded")
            return challenge
        logger.warning("Webhook verification failed")
        return None

    @staticmethod
    async def process_webhook_event(payload: FacebookWebhookPayload) -> None:
        """
        Process incoming Facebook webhook events.

        Routes messages to appropriate handlers based on message type (text/image).

        Args:
            payload: The validated Facebook webhook payload
        """
        logger.info(f"Webhook event received — object={payload.object}, entries={len(payload.entry)}")

        for entry in payload.entry:
            # TODO: Map Facebook page ID to internal store name for production
            # For now, hardcode to "goodybro" for testing
            page_id = "goodybro"  # entry.id

            for messaging in entry.messaging:
                sender_id = messaging.sender.id

                # Process message events
                if messaging.message:
                    message_dict = messaging.message.model_dump()
                    text = message_dict.get('text', '')
                    att_count = len(message_dict.get('attachments') or [])
                    logger.info(
                        f"[{sender_id}] 📨 Webhook message — "
                        f"mid={message_dict.get('mid')} | "
                        f"text=\"{text[:80]}{'…' if len(text or '') > 80 else ''}\" | "
                        f"attachments={att_count}"
                    )

                    # Route message to appropriate handler (text or image)
                    await message_router.route_message(
                        sender_id=sender_id,
                        message=message_dict,
                        page_id=page_id,
                    )

                # Log other event types at debug level
                if messaging.postback:
                    logger.debug(f"[{sender_id}] Postback: {messaging.postback}")

                if messaging.delivery:
                    logger.debug(f"[{sender_id}] Delivery receipt")

                if messaging.read:
                    logger.debug(f"[{sender_id}] Read receipt")


facebook_service = FacebookService()
