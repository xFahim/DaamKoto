"""Debouncing service that batches rapid sequential text messages per sender."""

import asyncio
from app.core.config import settings
from app.services.messaging_service import messaging_service
from app.services.handlers.text_handler import text_handler


class MessageBatcher:

    def __init__(self) -> None:
        self._pending_items: dict[str, dict] = {}
        self._timers: dict[str, asyncio.Task] = {}

    async def add_message(self, sender_id: str, page_id: str, text: str = None, image_url: str = None) -> None:
        """Append text or image to sender's batch, reset debounce timer."""
        is_first = sender_id not in self._pending_items
        
        if is_first:
            self._pending_items[sender_id] = {"texts": [], "image_urls": []}
            
        if text:
            self._pending_items[sender_id]["texts"].append(text)
        if image_url:
            self._pending_items[sender_id]["image_urls"].append(image_url)

        # Reset timer BEFORE any awaits so slow Meta API calls can't hijack the timer
        existing = self._timers.get(sender_id)
        if existing and not existing.done():
            existing.cancel()

        self._timers[sender_id] = asyncio.create_task(
            self._process_batch(sender_id, page_id)
        )

        # Non-critical — send after timer is locked in; failure doesn't affect batching
        if is_first:
            await messaging_service.send_typing_on(sender_id)

    async def _process_batch(self, sender_id: str, page_id: str) -> None:
        """Wait for debounce window, then flush batch to TextHandler."""
        try:
            await asyncio.sleep(settings.message_batch_timeout)
        except asyncio.CancelledError:
            raise  # Timer reset by new message — leave state alone, new task owns it

        # Timer fired normally — clean up state before processing
        batch = self._pending_items.pop(sender_id, {"texts": [], "image_urls": []})
        self._timers.pop(sender_id, None)

        if not batch["texts"] and not batch["image_urls"]:
            return

        combined_text = "\n".join(batch["texts"]) if batch["texts"] else ""
        
        try:
            await asyncio.wait_for(
                text_handler.process(
                    sender_id=sender_id,
                    message_text=combined_text,
                    page_id=page_id,
                    image_urls=batch["image_urls"]
                ),
                timeout=60.0,
            )
        except asyncio.TimeoutError:
            print(f"[MessageBatcher] Batch processing timed out for {sender_id}")
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
        self._pending_items.clear()


message_batcher = MessageBatcher()
