"""Quick test for message batching / debouncing logic."""

import asyncio
import sys
from unittest.mock import AsyncMock, patch

# ── Patch external services before importing batching_service ──────────────
mock_text_handler = AsyncMock()
mock_messaging_service = AsyncMock()
mock_settings = type("S", (), {"message_batch_timeout": 0.3})()  # 0.3s so test is fast

patches = [
    patch("app.services.handlers.text_handler.text_handler", mock_text_handler),
    patch("app.services.messaging_service.messaging_service", mock_messaging_service),
    patch("app.core.config.settings", mock_settings),
]
for p in patches:
    p.start()

from app.services.batching_service import MessageBatcher  # noqa: E402


async def run_tests():
    passed = 0
    failed = 0

    def ok(label):
        nonlocal passed
        print(f"  PASS  {label}")
        passed += 1

    def fail(label, reason):
        nonlocal failed
        print(f"  FAIL  {label}: {reason}")
        failed += 1

    # ── Test 1: 3 rapid messages → 1 combined AI call ─────────────────────
    batcher = MessageBatcher()
    batcher._pending_messages = {}
    batcher._timers = {}

    text_calls = []

    async def fake_process(sender_id, message_text, page_id):
        text_calls.append(message_text)

    with patch.object(
        type(batcher).__mro__[0],  # won't work — patch directly
        "__init__",
        lambda s: None,
    ):
        pass  # skip, just use instance directly

    # Monkey-patch text_handler.process on the module
    import app.services.batching_service as bs
    original_handler = bs.text_handler
    bs.text_handler = type("H", (), {"process": staticmethod(fake_process)})()
    bs.messaging_service = type("M", (), {"send_typing_on": AsyncMock()})()
    bs.settings = mock_settings

    batcher = MessageBatcher()
    text_calls.clear()

    await batcher.add_text_message("user1", "I want a shirt", "page1")
    await asyncio.sleep(0.05)
    await batcher.add_text_message("user1", "blue colour", "page1")
    await asyncio.sleep(0.05)
    await batcher.add_text_message("user1", "under 500 taka", "page1")

    await asyncio.sleep(0.5)  # wait for debounce to fire

    if len(text_calls) == 1:
        ok("3 rapid messages = 1 AI call")
    else:
        fail("3 rapid messages = 1 AI call", f"got {len(text_calls)} calls")

    expected = "I want a shirt\nblue colour\nunder 500 taka"
    if text_calls and text_calls[0] == expected:
        ok("Combined text is correct")
    else:
        fail("Combined text is correct", f"got: {repr(text_calls[0] if text_calls else None)}")

    # ── Test 2: typing_on sent only once (on first message) ───────────────
    typing_mock = AsyncMock()
    bs.messaging_service = type("M", (), {"send_typing_on": typing_mock})()
    batcher2 = MessageBatcher()

    await batcher2.add_text_message("user2", "msg1", "page1")
    await asyncio.sleep(0.05)
    await batcher2.add_text_message("user2", "msg2", "page1")
    await asyncio.sleep(0.05)
    await batcher2.add_text_message("user2", "msg3", "page1")
    await asyncio.sleep(0.5)

    call_count = typing_mock.call_count
    if call_count == 1:
        ok("typing_on sent exactly once per batch")
    else:
        fail("typing_on sent exactly once per batch", f"called {call_count} times")

    # ── Test 3: 2 messages >timeout apart → 2 independent AI calls ────────
    calls3 = []

    async def fake_process3(sender_id, message_text, page_id):
        calls3.append(message_text)

    bs.messaging_service = type("M", (), {"send_typing_on": AsyncMock()})()
    bs.text_handler = type("H", (), {"process": staticmethod(fake_process3)})()
    batcher3 = MessageBatcher()

    await batcher3.add_text_message("user3", "first message", "page1")
    await asyncio.sleep(0.5)  # let first batch fire
    await batcher3.add_text_message("user3", "second message", "page1")
    await asyncio.sleep(0.5)  # let second batch fire

    if len(calls3) == 2:
        ok("Messages far apart = 2 independent calls")
    else:
        fail("Messages far apart = 2 independent calls", f"got {len(calls3)} calls")

    if calls3 == ["first message", "second message"]:
        ok("Both messages processed independently with correct text")
    else:
        fail("Both messages processed independently with correct text", f"got: {calls3}")

    # ── Test 4: shutdown cancels pending timers cleanly ───────────────────
    bs.messaging_service = type("M", (), {"send_typing_on": AsyncMock()})()
    batcher4 = MessageBatcher()

    await batcher4.add_text_message("user4", "pending msg", "page1")
    # Don't wait — shut down immediately
    await batcher4.shutdown()

    if not batcher4._timers and not batcher4._pending_messages:
        ok("Shutdown clears all state cleanly")
    else:
        fail("Shutdown clears all state cleanly", f"timers={batcher4._timers}, msgs={batcher4._pending_messages}")

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{passed + failed} tests | {passed} passed | {failed} failed")
    return failed


if __name__ == "__main__":
    failed = asyncio.run(run_tests())
    sys.exit(1 if failed else 0)
