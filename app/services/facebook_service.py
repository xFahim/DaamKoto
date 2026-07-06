"""Service layer for Facebook Messenger webhook processing."""

from cachetools import TTLCache
from app.core.config import settings
from app.core.logging_config import get_logger
from app.core.tenant_context import resolve_tenant, TenantNotFoundError, TenantInactiveError
from app.schemas.facebook import FacebookWebhookPayload
from app.services.handlers.message_router import message_router

logger = get_logger(__name__)

# Cache processed message IDs (mids) to handle Facebook webhook retries.
# 5 minutes TTL is generally sufficient to catch immediate retry bursts.
_processed_mids = TTLCache(maxsize=10000, ttl=300)


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

        Resolves the tenant from bot_settings using the Facebook page ID,
        then routes messages through the pipeline with full tenant context.

        Args:
            payload: The validated Facebook webhook payload
        """
        logger.info(f"Webhook event received — object={payload.object}, entries={len(payload.entry)}")

        for entry in payload.entry:
            # Resolve tenant from bot_settings using Facebook page ID
            try:
                page_tenant = await resolve_tenant(entry.id)
            except TenantNotFoundError:
                logger.error(
                    f"No bot_settings row for facebook_page_id={entry.id} — "
                    f"skipping all messages in this entry"
                )
                continue
            except TenantInactiveError:
                logger.info(
                    f"Bot for facebook_page_id={entry.id} is inactive — "
                    f"silently skipping {len(entry.messaging)} event(s)"
                )
                continue

            for messaging in entry.messaging:
                sender_id = messaging.sender.id
                # Derive a per-message immutable context stamped with this sender
                tenant = page_tenant.for_sender(sender_id)

                # Process message events
                if messaging.message:
                    message_dict = messaging.message.model_dump()
                    mid = message_dict.get('mid')
                    
                    # --- Webhook Idempotency Check ---
                    if mid:
                        if mid in _processed_mids:
                            logger.info(f"[{sender_id}] ♻️ Idempotency check: Dropping duplicate message (mid={mid})")
                            continue
                        _processed_mids[mid] = True
                    # ---------------------------------

                    text = message_dict.get('text', '')
                    att_count = len(message_dict.get('attachments') or [])
                    logger.info(
                        f"[{sender_id}] 📨 Webhook message — "
                        f"mid={mid} | "
                        f"text=\"{text[:80]}{'…' if len(text or '') > 80 else ''}\" | "
                        f"attachments={att_count}"
                    )

                    # Route message to appropriate handler (text or image)
                    await message_router.route_message(
                        sender_id=sender_id,
                        message=message_dict,
                        tenant=tenant,
                    )

                # Log other event types at debug level
                if messaging.postback:
                    logger.debug(f"[{sender_id}] Postback: {messaging.postback}")

                if messaging.delivery:
                    logger.debug(f"[{sender_id}] Delivery receipt")

                if messaging.read:
                    logger.debug(f"[{sender_id}] Read receipt")


facebook_service = FacebookService()
