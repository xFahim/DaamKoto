"""Facebook Messenger webhook endpoints."""

import asyncio
from fastapi import APIRouter, Query, Request, HTTPException, status
from fastapi.responses import PlainTextResponse
from app.core.config import settings
from app.services.facebook_service import facebook_service
from app.schemas.facebook import FacebookWebhookPayload

router = APIRouter()


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
async def receive_webhook(payload: FacebookWebhookPayload) -> dict[str, str]:
    """
    Facebook webhook message reception endpoint.

    Returns 200 immediately and processes the event in the background.
    This prevents Facebook from timing out and retrying the request.

    Args:
        payload: The Facebook webhook payload

    Returns:
        A success response (returned immediately)
    """
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
        print(f"❌ Background webhook processing failed: {e}")
        import traceback
        traceback.print_exc()


def _log_task_exception(task: asyncio.Task) -> None:
    """Callback to log unhandled exceptions from background tasks."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc:
        print(f"❌ Unhandled error in background task: {exc}")
