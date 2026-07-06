"""
Input validation and rate limiting for incoming user text messages.

Covers:
  - Empty / whitespace-only messages          → silent drop
  - ASCII control character injection         → stripped then re-evaluated
  - Unicode null / invisible character abuse  → stripped
  - Oversized messages (payload bombing)      → reject with user feedback
  - Rapid-fire spam (rate limiting)           → reject; user notified ONCE per window
  - Prompt injection attempts                 → detected and logged, text passed
                                               through UNCHANGED (the system prompt
                                               is the real defence — deleting phrases
                                               mid-sentence corrupts legitimate
                                               messages like 'can this act as a raincoat?')

All operations are synchronous in-memory — no I/O, safe to call without await.
"""

import re
import time
from app.core.config import settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)

# ── Regex patterns compiled once at import time ────────────────────────────

# ASCII control chars to remove: everything except \t (tab) and \n (newline),
# which are legitimate in multi-line product queries.
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Unicode categories that are invisible / zero-width and serve no legitimate purpose:
#   \u200b-\u200f  zero-width spaces/joiners
#   \u202a-\u202e  bidirectional overrides (can be used to obscure intent)
#   \ufeff         BOM
#   \u00ad         soft hyphen (invisible)
_INVISIBLE_UNICODE = re.compile(
    r"[\u00ad\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff]"
)

# Crude prompt-injection signal phrases — English-only, so they are a
# monitoring signal, not a defence. We LOG matches for observability but never
# mutate the user's text; the system prompt and server-side tool guards are
# the actual protection.
_INJECTION_PATTERNS = re.compile(
    r"(ignore\s+(all\s+)?(?:previous\s+)?instructions?|"
    r"forget\s+(?:all\s+)?(?:your\s+)?instructions?|"
    r"system\s*prompt|"
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
        # sender_id -> (window_start: float, count: int, notified: bool)
        self._windows: dict[str, tuple[float, int, bool]] = {}

    def check(self, sender_id: str, text: str) -> tuple[str, str]:
        """
        Validate and clean a single incoming message.

        Returns one of:
          ("ok",           cleaned_text)          — pass through to batcher
          ("silent_drop",  "")                    — ignore silently (empty/whitespace)
          ("reject",       "too_long")            — message exceeded max length
          ("reject",       "rate_limited_notify") — rate limited; tell the user (once per window)
          ("reject",       "rate_limited_silent") — rate limited; drop without replying
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

        # ── Stage 2: detect (don't mutate) prompt-injection phrases ────────
        match = _INJECTION_PATTERNS.search(cleaned)
        if match:
            logger.warning(
                f"[{sender_id}] ⚠️ Possible prompt-injection phrase detected: "
                f"\"{match.group(0)}\" — passing through unchanged"
            )

        # ── Stage 3: rate limit — consume a slot before length check so that
        #    deliberately oversized messages also burn the rate budget ─────
        allowed, should_notify = self._try_count(sender_id)
        if not allowed:
            return "reject", "rate_limited_notify" if should_notify else "rate_limited_silent"

        # ── Stage 4: length check ──────────────────────────────────────────
        if len(cleaned) > settings.max_message_length:
            return "reject", "too_long"

        return "ok", cleaned

    # ── Private helpers ────────────────────────────────────────────────────

    def _try_count(self, sender_id: str) -> tuple[bool, bool]:
        """
        Attempt to consume one rate-limit slot for sender_id.

        Returns (allowed, should_notify). should_notify is True exactly once
        per exhausted window, so the bot doesn't spam "slow down" replies at
        someone pasting many messages.
        """
        now = time.monotonic()
        window_secs: float = settings.rate_limit_window
        limit: int = settings.rate_limit_messages

        if sender_id in self._windows:
            start, count, notified = self._windows[sender_id]
            if now - start >= window_secs:
                # Window expired — fresh window starting now
                self._windows[sender_id] = (now, 1, False)
                return True, False
            if count >= limit:
                self._windows[sender_id] = (start, count, True)
                return False, not notified
            self._windows[sender_id] = (start, count + 1, notified)
            return True, False

        self._windows[sender_id] = (now, 1, False)
        return True, False


input_guard = InputGuard()
