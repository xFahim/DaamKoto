"""Service for sending messages to Facebook Messenger users."""

import httpx
from app.core.logging_config import get_logger
from app.services.reply_context import store_mid

logger = get_logger(__name__)

# Matches the Graph API version used by the tormoose dashboard token exchange.
GRAPH_API_BASE = "https://graph.facebook.com/v22.0"
GRAPH_API_URL = f"{GRAPH_API_BASE}/me/messages"

# Facebook rejects text messages longer than 2000 characters.
MAX_MESSAGE_CHARS = 2000

# Shared client — connection pooling instead of a new TLS handshake per send.
_http_client = httpx.AsyncClient(timeout=10.0)


def split_message(text: str, limit: int = MAX_MESSAGE_CHARS) -> list[str]:
    """Split a long message into Facebook-sized chunks.

    Prefers paragraph breaks, then line breaks, then sentence ends, and only
    hard-cuts when a single unbroken run exceeds the limit.
    """
    text = text.strip()
    if len(text) <= limit:
        return [text] if text else []

    chunks: list[str] = []
    remaining = text
    separators = ["\n\n", "\n", ". ", "! ", "? ", " "]

    while len(remaining) > limit:
        window = remaining[:limit]
        cut = -1
        for sep in separators:
            idx = window.rfind(sep)
            if idx > 0:
                cut = idx + len(sep)
                break
        if cut <= 0:
            cut = limit  # single unbroken run — hard cut
        chunks.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()

    if remaining:
        chunks.append(remaining)
    return [c for c in chunks if c]


class MessagingService:
    """Service for handling message sending to Facebook Messenger."""

    @staticmethod
    async def send_typing_on(recipient_id: str, access_token: str) -> None:
        """
        Send a typing indicator to show the bot is processing.

        Args:
            recipient_id: The Facebook user ID to show typing to
            access_token: The Facebook page access token for this tenant
        """
        params = {"access_token": access_token}
        payload = {
            "recipient": {"id": recipient_id},
            "sender_action": "typing_on",
        }

        try:
            await _http_client.post(GRAPH_API_URL, params=params, json=payload)
        except Exception as e:
            # Non-critical — don't fail the whole flow for a typing indicator
            logger.debug(f"Failed to send typing indicator: {e}")

    @staticmethod
    async def get_profile_name(psid: str, access_token: str) -> str | None:
        """Fetch the customer's real name from the Graph User Profile API.

        Works for anyone who has messaged the page (pages_messaging grants
        first_name/last_name access). Best effort: unverified apps or privacy
        settings can deny it — return None and the dashboard falls back to
        'Customer XXXXXX'.
        """
        try:
            response = await _http_client.get(
                f"{GRAPH_API_BASE}/{psid}",
                params={"fields": "first_name,last_name", "access_token": access_token},
            )
            if response.status_code == 200:
                data = response.json()
                name = " ".join(
                    part for part in [data.get("first_name"), data.get("last_name")] if part
                ).strip()
                return name or None
            logger.debug(
                f"Profile fetch for {psid} returned {response.status_code}: {response.text[:200]}"
            )
        except Exception as e:
            logger.debug(f"Profile fetch for {psid} failed: {e}")
        return None

    @staticmethod
    async def send_message(recipient_id: str, message_text: str, access_token: str) -> bool:
        """
        Send a text message to a Facebook Messenger user.

        Messages over Facebook's 2000-char limit are split on natural
        boundaries and sent sequentially.

        Args:
            recipient_id: The Facebook user ID to send the message to
            message_text: The text content of the message
            access_token: The Facebook page access token for this tenant

        Returns:
            True if every chunk was sent successfully, False otherwise
        """
        chunks = split_message(message_text)
        if not chunks:
            return True
        if len(chunks) > 1:
            logger.info(f"Message to {recipient_id} split into {len(chunks)} chunks")

        params = {"access_token": access_token}
        all_ok = True

        for chunk in chunks:
            payload = {
                "recipient": {"id": recipient_id},
                "message": {"text": chunk},
            }
            try:
                response = await _http_client.post(GRAPH_API_URL, params=params, json=payload)
                if response.status_code == 200:
                    # Store bot reply mid → text for reply-to resolution
                    resp_data = response.json()
                    bot_mid = resp_data.get("message_id")
                    if bot_mid:
                        store_mid(bot_mid, chunk)
                    logger.debug(f"Message delivered to {recipient_id}")
                else:
                    logger.error(
                        f"Failed to send message to {recipient_id}. "
                        f"Status: {response.status_code}, Response: {response.text}"
                    )
                    all_ok = False
                    break  # don't send later chunks out of order after a failure
            except Exception as e:
                logger.error(f"Error sending message to {recipient_id}: {e}")
                all_ok = False
                break

        return all_ok

    @staticmethod
    async def send_image(recipient_id: str, image_url: str, access_token: str) -> bool:
        """
        Send an image via Facebook Messenger API.

        Args:
            recipient_id: The Facebook user ID to send the image to
            image_url: The URL of the image to send
            access_token: The Facebook page access token for this tenant
        """
        params = {"access_token": access_token}
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
            response = await _http_client.post(GRAPH_API_URL, params=params, json=payload)
            if response.status_code == 200:
                # Store bot image mid → description for reply-to resolution
                resp_data = response.json()
                bot_mid = resp_data.get("message_id")
                if bot_mid:
                    store_mid(bot_mid, f"[Bot sent a product image: {image_url}]")
                logger.debug(f"Image delivered to {recipient_id}")
                return True
            else:
                logger.error(f"Failed to send image to {recipient_id}. Status: {response.status_code}, Response: {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error sending image to {recipient_id}: {e}")
            return False


messaging_service = MessagingService()
