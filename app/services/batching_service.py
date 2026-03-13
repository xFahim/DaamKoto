"""Debouncing service that batches rapid sequential text messages per sender."""

import asyncio
from app.core.config import settings
from app.services.messaging_service import messaging_service
from app.services.handlers.text_handler import text_handler


class MessageBatcher:

    def __init__(self) -> None:
        self._pending_messages: dict[str, list[str]] = {}
        self._timers: dict[str, asyncio.Task] = {}

    async def add_text_message(self, sender_id: str, text: str, page_id: str) -> None:
        """Append text to sender's batch, reset debounce timer. Sends typing_on on first message."""
        is_first = sender_id not in self._pending_messages
        self._pending_messages.setdefault(sender_id, []).append(text)

        if is_first:
            await messaging_service.send_typing_on(sender_id)

        existing = self._timers.get(sender_id)
        if existing and not existing.done():
            existing.cancel()

        self._timers[sender_id] = asyncio.create_task(
            self._process_batch(sender_id, page_id)
        )

    async def _process_batch(self, sender_id: str, page_id: str) -> None:
        """Wait for debounce window, then flush batch to TextHandler."""
        try:
            await asyncio.sleep(settings.message_batch_timeout)
        except asyncio.CancelledError:
            raise  # Timer reset by new message — leave state alone, new task owns it

        # Timer fired normally — clean up state before processing
        messages = self._pending_messages.pop(sender_id, [])
        self._timers.pop(sender_id, None)

        if not messages:
            return

        combined_text = "\n".join(messages)
        try:
            await text_handler.process(
                sender_id=sender_id,
                message_text=combined_text,
                page_id=page_id,
            )
        except Exception as e:
            print(f"[MessageBatcher] Error processing batch for {sender_id}: {e}")

    async def shutdown(self) -> None:
        """Cancel all pending timers. Called during app shutdown."""
        for task in list(self._timers.values()):
            if not task.done():
                task.cancel()
        if self._timers:
            await asyncio.gather(*self._timers.values(), return_exceptions=True)
        self._timers.clear()
        self._pending_messages.clear()


message_batcher = MessageBatcher()
