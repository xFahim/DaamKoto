"""Facebook Messenger webhook endpoints."""

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

    This endpoint receives incoming messages and events from Facebook Messenger.
    It processes the payload asynchronously and returns immediately.

    Args:
        payload: The Facebook webhook payload

    Returns:
        A success response
    """
    # Process the webhook event (logs to console)
    facebook_service.process_webhook_event(payload)

    # Return 200 immediately
    return {"status": "ok"}
