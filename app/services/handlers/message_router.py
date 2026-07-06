"""Router for categorizing and routing messages to appropriate handlers."""

from typing import Any
from app.core.logging_config import get_logger
from app.core.tenant_context import TenantContext
from app.services.batching_service import message_batcher
from app.services.messaging_service import messaging_service
from app.services.input_guard import input_guard
from app.services.reply_context import store_mid, resolve_mid
from app.core.config import settings

logger = get_logger(__name__)


class MessageRouter:
    """Router that categorizes messages and routes them to appropriate handlers."""

    @staticmethod
    async def route_message(
        sender_id: str,
        message: dict[str, Any] | None,
        tenant: TenantContext,
    ) -> None:
        """
        Categorize a message and route it to the appropriate handler.

        Args:
            sender_id: The Facebook user ID who sent the message
            message: The message object from Facebook webhook
            tenant: The resolved tenant context for this request
        """
        if not message:
            return

        # Store this message's mid → text for future reply-to lookups
        mid = message.get("mid")
        raw_text = message.get("text", "")
        if mid and raw_text:
            store_mid(mid, raw_text)

        # Resolve reply-to context if present
        reply_to = message.get("reply_to")
        reply_context_prefix = ""
        if reply_to and isinstance(reply_to, dict):
            ref_mid = reply_to.get("mid")
            if ref_mid:
                original_text = resolve_mid(ref_mid)
                if original_text:
                    reply_context_prefix = f"[Replying to: \"{original_text}\"]\n"
                    logger.info(f"[{sender_id}] ↩️ Reply-to resolved: \"{original_text[:80]}\"")
                else:
                    reply_context_prefix = "[Replying to an earlier message]\n"
                    logger.info(f"[{sender_id}] ↩️ Reply-to mid={ref_mid} (text expired/unknown)")

        handled = False

        # Check if message has text
        if message.get("text"):
            status, payload = input_guard.check(sender_id, message["text"])

            if status == "ok":
                # Prepend reply context so the agent knows what's being referenced
                enriched_text = reply_context_prefix + payload if reply_context_prefix else payload
                logger.info(f"[{sender_id}] 📩 TEXT received — \"{payload[:100]}{'…' if len(payload) > 100 else ''}\"")
                await message_batcher.add_message(
                    sender_id=sender_id,
                    text=enriched_text,
                    tenant=tenant,
                )
                handled = True
            elif status == "reject":
                handled = True
                if payload == "too_long":
                    logger.warning(f"[{sender_id}] 🚫 Rejected: message too long ({len(message['text'])} chars)")
                    await messaging_service.send_message(
                        recipient_id=sender_id,
                        message_text=(
                            f"Please keep your message under {settings.max_message_length} characters "
                            "so I can understand you better!"
                        ),
                        access_token=tenant.page_access_token,
                    )
                elif payload == "rate_limited_notify":
                    logger.warning(f"[{sender_id}] 🚫 Rejected: rate limited (notifying once)")
                    await messaging_service.send_message(
                        recipient_id=sender_id,
                        message_text=(
                            "You're sending messages too fast! "
                            "Take a moment and try again."
                        ),
                        access_token=tenant.page_access_token,
                    )
                elif payload == "rate_limited_silent":
                    logger.warning(f"[{sender_id}] 🚫 Rejected: rate limited (silent)")
            elif status == "silent_drop":
                logger.debug(f"[{sender_id}] Silent drop — empty/stripped message")
                handled = True

        # Check if message has image attachments
        attachments = message.get("attachments")
        if attachments:
            for att in attachments:
                if isinstance(att, dict) and att.get("type") == "image":
                    url = att.get("payload", {}).get("url")
                    if url:
                        logger.info(f"[{sender_id}] 📷 IMAGE received — {url[:80]}…")
                        await message_batcher.add_message(
                            sender_id=sender_id,
                            tenant=tenant,
                            image_url=url
                        )
                        handled = True

        # If message doesn't match any category, send a default response
        if not handled:
            logger.info(f"[{sender_id}] ❓ Unsupported message type — sending fallback")
            await MessageRouter._send_unsupported_message(sender_id, tenant)

    @staticmethod
    async def _send_unsupported_message(sender_id: str, tenant: TenantContext) -> None:
        """
        Send a message for unsupported message types.

        Args:
            sender_id: The Facebook user ID to send the message to
            tenant: The resolved tenant context
        """
        await messaging_service.send_message(
            recipient_id=sender_id,
            message_text=(
                "I can help you with text messages or product images! "
                "Please send me a message or an image of a product you're looking for."
            ),
            access_token=tenant.page_access_token,
        )


message_router = MessageRouter()
