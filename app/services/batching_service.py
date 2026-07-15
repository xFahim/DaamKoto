"""Debouncing service that batches rapid sequential text messages per sender."""

import asyncio
from app.core.config import settings
from app.core.logging_config import get_logger
from app.core.tenant_context import TenantContext
from app.services.messaging_service import messaging_service
from app.services.handlers.text_handler import text_handler

logger = get_logger(__name__)

# Hard ceiling for one agent run (LLM turns + tools + typing delay).
PROCESSING_TIMEOUT = 90.0


class MessageBatcher:
    """Per-conversation debounce + serialization.

    Keys are '{shop_id}:{sender_psid}' — PSIDs are page-scoped, so the shop
    namespace prevents any cross-tenant bleed.

    Lifecycle per conversation:
      - add_message() appends to the pending batch and (re)starts a debounce timer.
      - When the timer fires, it atomically takes the batch, then processes it
        under a per-conversation lock.
      - New messages arriving DURING processing start a fresh batch + timer;
        that timer then waits on the lock, so batches are processed in order
        and an in-flight LLM call is never cancelled (the old code cancelled
        it and silently dropped the popped batch).
    """

    def __init__(self) -> None:
        self._pending_items: dict[str, dict] = {}
        self._timers: dict[str, asyncio.Task] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def add_message(self, sender_id: str, tenant: TenantContext, text: str = None, image_url: str = None) -> None:
        """Append text or image to sender's batch, reset debounce timer."""
        key = f"{tenant.shop_id}:{sender_id}"
        is_first = key not in self._pending_items

        if is_first:
            self._pending_items[key] = {"texts": [], "image_urls": []}

        if text:
            self._pending_items[key]["texts"].append(text)
        if image_url:
            self._pending_items[key]["image_urls"].append(image_url)

        batch = self._pending_items[key]
        logger.debug(
            f"[{sender_id}] Batch updated — {len(batch['texts'])} text(s), "
            f"{len(batch['image_urls'])} image(s) | debounce={settings.message_batch_timeout}s"
        )

        # Reset the debounce timer. Safe: a timer removes itself from _timers
        # (with no await in between) the moment its sleep completes, so a task
        # still present here is guaranteed to be sleeping — cancelling it can
        # never kill in-flight processing.
        existing = self._timers.get(key)
        if existing and not existing.done():
            existing.cancel()

        self._timers[key] = asyncio.create_task(
            self._process_batch(key, sender_id, tenant)
        )

        # Non-critical — send after timer is locked in; failure doesn't affect batching
        if is_first:
            await messaging_service.send_typing_on(sender_id, access_token=tenant.page_access_token)

    async def _process_batch(self, key: str, sender_id: str, tenant: TenantContext) -> None:
        """Wait for debounce window, then flush batch to TextHandler."""
        try:
            await asyncio.sleep(settings.message_batch_timeout)
        except asyncio.CancelledError:
            raise  # Timer reset by new message — leave state alone, new task owns it

        # Timer fired — atomically claim the batch and deregister this timer.
        # No awaits between these lines, so add_message can't interleave.
        batch = self._pending_items.pop(key, {"texts": [], "image_urls": []})
        self._timers.pop(key, None)

        if not batch["texts"] and not batch["image_urls"]:
            return

        combined_text = "\n".join(batch["texts"]) if batch["texts"] else ""

        logger.info(
            f"[{sender_id}] 📦 Batch flushed — {len(batch['texts'])} text(s), "
            f"{len(batch['image_urls'])} image(s)"
        )

        # Serialize per conversation: replies go out in order, and memory
        # writes for one customer never interleave.
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            try:
                await asyncio.wait_for(
                    text_handler.process(
                        sender_id=sender_id,
                        message_text=combined_text,
                        tenant=tenant,
                        image_urls=batch["image_urls"]
                    ),
                    timeout=PROCESSING_TIMEOUT,
                )
            except asyncio.TimeoutError:
                # Errors are logged only — NEVER sent to the user. The typing
                # indicator expires on its own within ~20s.
                logger.error(f"[{sender_id}] Batch processing timed out after {PROCESSING_TIMEOUT}s")
            except Exception as e:
                logger.error(f"[{sender_id}] Batch processing error: {e}", exc_info=True)

    async def shutdown(self) -> None:
        """Cancel all pending timers. Called during app shutdown."""
        for task in list(self._timers.values()):
            if not task.done():
                task.cancel()
        if self._timers:
            await asyncio.gather(*self._timers.values(), return_exceptions=True)
        self._timers.clear()
        self._pending_items.clear()
        self._locks.clear()


message_batcher = MessageBatcher()
