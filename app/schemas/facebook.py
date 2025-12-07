"""Pydantic models for Facebook Webhook payloads."""

from typing import Any, Optional
from pydantic import BaseModel, Field


class Sender(BaseModel):
    """Sender information in a messaging event."""

    id: str


class Recipient(BaseModel):
    """Recipient information in a messaging event."""

    id: str


class Message(BaseModel):
    """Message content in a messaging event."""

    mid: Optional[str] = None
    text: Optional[str] = None
    attachments: Optional[list[dict[str, Any]]] = None


class MessagingItem(BaseModel):
    """Individual messaging event within an entry."""

    sender: Sender
    recipient: Recipient
    timestamp: int
    message: Optional[Message] = None
    postback: Optional[dict[str, Any]] = None
    delivery: Optional[dict[str, Any]] = None
    read: Optional[dict[str, Any]] = None


class WebhookEntry(BaseModel):
    """Entry object in Facebook webhook payload."""

    id: str
    time: int
    messaging: list[MessagingItem] = Field(default_factory=list)


class FacebookWebhookPayload(BaseModel):
    """Root model for Facebook webhook payload."""

    object: str
    entry: list[WebhookEntry] = Field(default_factory=list)
