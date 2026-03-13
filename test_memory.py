"""Standalone tests for MemoryService."""

import sys
from unittest.mock import patch

# Patch settings before importing memory_service
mock_settings = type("S", (), {"conversation_ttl": 60, "conversation_max_turns": 5})()
with patch("app.core.config.settings", mock_settings):
    from cachetools import TTLCache
    import app.services.memory_service as ms
    ms._cache = TTLCache(maxsize=1000, ttl=mock_settings.conversation_ttl)
    ms.settings = mock_settings
    from app.services.memory_service import MemoryService

svc = MemoryService()
passed = 0
failed = 0

def ok(label):
    global passed
    print(f"  PASS  {label}")
    passed += 1

def fail(label, reason):
    global failed
    print(f"  FAIL  {label}: {reason}")
    failed += 1

# Test 1: Unknown sender returns empty string
result = svc.get_history("unknown_user")
if result == "":
    ok("Unknown sender returns empty string")
else:
    fail("Unknown sender returns empty string", f"got: {repr(result)}")

# Test 2: Save one turn, get correct format
svc.save_turn("user1", "do you have blue shirts?", "Yes, we do!")
result = svc.get_history("user1")
expected = "User: do you have blue shirts?\nBot: Yes, we do!"
if result == expected:
    ok("Single turn formatted correctly")
else:
    fail("Single turn formatted correctly", f"got: {repr(result)}")

# Test 3: Save 6 turns, verify only last 5 are kept
ms._cache.clear()
svc2 = MemoryService()
for i in range(6):
    svc2.save_turn("user2", f"msg {i}", f"reply {i}")

history = svc2.get_history("user2")
lines = history.split("\n")
# 5 turns = 10 lines (5 user + 5 bot)
if len(lines) == 10:
    ok("6 turns saved, only last 5 kept (10 lines)")
else:
    fail("6 turns saved, only last 5 kept (10 lines)", f"got {len(lines)} lines")

# Check first line is turn 1 (not turn 0)
if lines[0] == "User: msg 1":
    ok("Oldest turn correctly trimmed (starts at turn 1, not 0)")
else:
    fail("Oldest turn correctly trimmed", f"first line: {repr(lines[0])}")

# Test 4: Two senders have independent histories
ms._cache.clear()
svc3 = MemoryService()
svc3.save_turn("alice", "I want a dress", "Sure!")
svc3.save_turn("bob", "show me shoes", "Got it!")
alice_hist = svc3.get_history("alice")
bob_hist = svc3.get_history("bob")
if "dress" in alice_hist and "shoes" not in alice_hist:
    ok("Alice's history is independent of Bob's")
else:
    fail("Alice's history is independent", f"alice: {repr(alice_hist)}")
if "shoes" in bob_hist and "dress" not in bob_hist:
    ok("Bob's history is independent of Alice's")
else:
    fail("Bob's history is independent", f"bob: {repr(bob_hist)}")

# Summary
print(f"\n{passed + failed} tests | {passed} passed | {failed} failed")
sys.exit(1 if failed else 0)
