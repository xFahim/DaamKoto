"""Facebook Messenger webhook endpoints."""

import asyncio
import hashlib
import hmac
from fastapi import APIRouter, Query, Request, HTTPException, status
from fastapi.responses import PlainTextResponse
from pydantic import ValidationError
from app.core.config import settings
from app.core.logging_config import get_logger
from app.services.facebook_service import facebook_service
from app.schemas.facebook import FacebookWebhookPayload

logger = get_logger(__name__)

router = APIRouter()


def _verify_signature(raw_body: bytes, signature_header: str | None) -> bool:
    """Validate the X-Hub-Signature-256 header against the raw request body.

    Facebook signs every webhook delivery with HMAC-SHA256 using the app secret.
    Without this check, anyone who discovers the URL can forge messages with
    arbitrary page/sender IDs.
    """
    if not signature_header or not signature_header.startswith("sha256="):
        return False
    expected = hmac.new(
        settings.facebook_app_secret.encode("utf-8"), raw_body, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(signature_header.removeprefix("sha256="), expected)


@router.get("/webhook")
async def verify_webhook(
    mode: str = Query(..., alias="hub.mode"),
    token: str = Query(..., alias="hub.verify_token"),
    challenge: str = Query(..., alias="hub.challenge"),
) -> PlainTextResponse:
    """
    Facebook webhook verification endpoint.

    This endpoint is called by Facebook to verify the webhook subscription.
    It validates the verify token and returns the challenge string as plain text.

    Args:
        mode: The mode parameter from Facebook (should be 'subscribe')
        token: The verify token from Facebook
        challenge: The challenge string from Facebook

    Returns:
        PlainTextResponse containing the challenge string if verification succeeds

    Raises:
        HTTPException: If verification fails
    """
    result = facebook_service.verify_webhook(
        mode=mode,
        token=token,
        challenge=challenge,
        verify_token=settings.facebook_verify_token,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Verification failed",
        )

    return PlainTextResponse(content=result)


@router.post("/webhook")
async def receive_webhook(request: Request) -> dict[str, str]:
    """
    Facebook webhook message reception endpoint.

    Verifies the X-Hub-Signature-256 HMAC on the raw body, then returns 200
    immediately and processes the event in the background. This prevents
    Facebook from timing out and retrying the request.

    Returns:
        A success response (returned immediately)
    """
    raw_body = await request.body()

    if settings.facebook_app_secret:
        signature = request.headers.get("X-Hub-Signature-256")
        if not _verify_signature(raw_body, signature):
            logger.warning(
                f"Webhook signature verification FAILED — "
                f"header={'present' if signature else 'missing'}, dropping payload"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid signature",
            )
    else:
        # Fail open so a missing env var doesn't take the bot down, but make noise.
        logger.warning(
            "FACEBOOK_APP_SECRET is not set — webhook signature NOT verified. "
            "Set it in production; forged payloads are otherwise accepted."
        )

    try:
        payload = FacebookWebhookPayload.model_validate_json(raw_body)
    except ValidationError as e:
        logger.warning(f"Webhook payload failed validation: {e.error_count()} error(s)")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid payload",
        )

    # Fire and forget — process in background so Facebook gets 200 instantly
    task = asyncio.create_task(_process_webhook_safe(payload))
    # Log any unhandled errors from the background task
    task.add_done_callback(_log_task_exception)

    return {"status": "ok"}


async def _process_webhook_safe(payload: FacebookWebhookPayload) -> None:
    """Wrapper to catch and log exceptions from background webhook processing."""
    try:
        await facebook_service.process_webhook_event(payload)
    except Exception as e:
        logger.error(f"Background webhook processing failed: {e}", exc_info=True)


def _log_task_exception(task: asyncio.Task) -> None:
    """Callback to log unhandled exceptions from background tasks."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc:
        logger.error(f"Unhandled error in background task: {exc}", exc_info=True)
