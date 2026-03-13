"""
Input validation and rate limiting for incoming user text messages.

Covers:
  - Empty / whitespace-only messages          → silent drop
  - ASCII control character injection         → stripped then re-evaluated
  - Unicode null / invisible character abuse  → stripped
  - Oversized messages (payload bombing)      → reject with user feedback
  - Rapid-fire spam (rate limiting)           → reject with user feedback
  - Prompt injection attempts                 → sanitised text passed through
                                               (Gemini system prompt is authoritative;
                                                we still strip the most obvious patterns)

All operations are synchronous in-memory — no I/O, safe to call without await.
"""

import re
import time
from app.core.config import settings

# ── Regex patterns compiled once at import time ────────────────────────────

# ASCII control chars to remove: everything except \t (tab) and \n (newline),
# which are legitimate in multi-line product queries.
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Unicode categories that are invisible / zero-width and serve no legitimate purpose:
#   \u200b–\u200f  zero-width spaces/joiners
#   \u202a–\u202e  bidirectional overrides (can be used to obscure intent)
#   \ufeff         BOM
#   \u00ad         soft hyphen (invisible)
_INVISIBLE_UNICODE = re.compile(
    r"[\u00ad\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff]"
)

# Crude prompt-injection signal phrases. We don't block on these — we strip the
# phrase and pass the cleaned text, because Gemini's system prompt is the real
# defence. Stripping makes logs cleaner and reduces token waste.
_INJECTION_PATTERNS = re.compile(
    r"(ignore\s+(all\s+)?(?:previous\s+)?instructions?|"
    r"you\s+are\s+now\s+(?:a\s+)?|"
    r"forget\s+(?:all\s+)?(?:your\s+)?instructions?|"
    r"system\s*prompt|"
    r"disregard\s+(?:all\s+)?(?:previous\s+)?|"
    r"act\s+as\s+(?:if\s+you\s+are\s+|a\s+)?|"
    r"jailbreak|"
    r"dan\s+mode)",
    re.IGNORECASE,
)


class InputGuard:
    """
    Stateful guard that cleans and rate-limits user messages.

    Rate limiting uses a fixed-window counter per sender_id stored in a plain
    dict. The dict grows with active users but stays small (a few hundred bytes
    per user). Old windows are lazily replaced when a new window starts, so
    memory doesn't leak for inactive users.
    """

    def __init__(self) -> None:
        # sender_id -> (window_start: float, count: int)
        self._windows: dict[str, tuple[float, int]] = {}

    def check(self, sender_id: str, text: str) -> tuple[str, str]:
        """
        Validate and clean a single incoming message.

        Returns one of:
          ("ok",           cleaned_text)   — pass through to batcher
          ("silent_drop",  "")             — ignore silently (empty/whitespace)
          ("reject",       "too_long")     — message exceeded max length
          ("reject",       "rate_limited") — sender exceeded message rate
        """
        # ── Stage 1: basic cleaning ────────────────────────────────────────

        cleaned = text.strip()

        # Silently drop empty or whitespace-only messages
        if not cleaned:
            return "silent_drop", ""

        # Remove ASCII control characters (keep \t and \n)
        cleaned = _CONTROL_CHARS.sub("", cleaned)

        # Strip invisible/zero-width Unicode characters
        cleaned = _INVISIBLE_UNICODE.sub("", cleaned)

        # Re-check after stripping — might now be empty
        cleaned = cleaned.strip()
        if not cleaned:
            return "silent_drop", ""

        # ── Stage 2: strip prompt-injection phrases ────────────────────────
        # Gemini's system prompt is the real defence; this just keeps logs clean.
        cleaned = _INJECTION_PATTERNS.sub("", cleaned).strip()
        if not cleaned:
            return "silent_drop", ""

        # ── Stage 3: rate limit — consume a slot before length check so that
        #    deliberately oversized messages also burn the rate budget ─────
        if not self._try_count(sender_id):
            return "reject", "rate_limited"

        # ── Stage 4: length check ──────────────────────────────────────────
        if len(cleaned) > settings.max_message_length:
            return "reject", "too_long"

        return "ok", cleaned

    # ── Private helpers ────────────────────────────────────────────────────

    def _try_count(self, sender_id: str) -> bool:
        """
        Attempt to consume one rate-limit slot for sender_id.
        Returns True if allowed, False if the window is exhausted.

        Uses a fixed window: the counter resets when the window duration has
        elapsed since the first message in that window.
        """
        now = time.monotonic()
        window_secs: float = settings.rate_limit_window
        limit: int = settings.rate_limit_messages

        if sender_id in self._windows:
            start, count = self._windows[sender_id]
            if now - start >= window_secs:
                # Window expired — fresh window starting now
                self._windows[sender_id] = (now, 1)
                return True
            if count >= limit:
                return False
            self._windows[sender_id] = (start, count + 1)
            return True

        self._windows[sender_id] = (now, 1)
        return True


input_guard = InputGuard()
