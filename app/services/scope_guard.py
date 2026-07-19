"""Off-topic / spam policy engine.

Split of responsibilities (2026-07-19, after the silent-"hi" incident):
  - The LLM only CLASSIFIES: it prefixes replies to hard-off-topic requests
    (do-my-homework, write code, gibberish) with OFFTOPIC_TAG. It never
    decides silence itself — prompts are bad at counting.
  - THIS module owns the policy: it counts tagged strikes per conversation
    and decides send vs mute against the shop's configurable threshold
    (bot_settings.spam_mute_threshold, edited from the dashboard).

Policy:
  - Untagged reply (shopping, greetings, small talk, hype about the match):
    always sent; the conversation's strike count RESETS — re-engaging on
    topic earns back goodwill.
  - Tagged reply, strike <= threshold: the model's polite redirect is sent
    (tag stripped).
  - Tagged reply, strike > threshold: muted — the customer already got
    redirect(s); we don't spam apologies back at spam.
  - Strikes expire after STRIKE_TTL_SECONDS (fresh start after a break).

In-process state — consistent with the single-worker constraint shared by
batching, memory, and rate limiting (move to Redis together, if ever).
"""

from cachetools import TTLCache

from app.core.logging_config import get_logger

logger = get_logger(__name__)

# The model prefixes hard-off-topic redirects with this exact token.
OFFTOPIC_TAG = "[OFFTOPIC]"

# Forgiveness window: strikes older than this are forgotten.
STRIKE_TTL_SECONDS = 30 * 60

# Used when the shop has no configured threshold (column missing / null).
DEFAULT_MUTE_THRESHOLD = 3

# "{shop_id}:{sender_id}" -> consecutive hard-off-topic strike count
_strikes: TTLCache = TTLCache(maxsize=5000, ttl=STRIKE_TTL_SECONDS)


class ScopeGuard:
    """Counts off-topic strikes and applies the shop's mute policy."""

    def apply(
        self,
        conversation_key: str,
        reply: str,
        mute_threshold: int | None = None,
    ) -> str:
        """Return the text to actually send; empty string means stay silent.

        mute_threshold = how many hard-off-topic messages still get a polite
        redirect. 0 mutes immediately; strikes beyond the threshold are
        dropped without a reply.
        """
        threshold = (
            mute_threshold
            if isinstance(mute_threshold, int) and mute_threshold >= 0
            else DEFAULT_MUTE_THRESHOLD
        )

        tagged = OFFTOPIC_TAG in reply
        text = reply.replace(OFFTOPIC_TAG, "").strip()

        if not tagged:
            if _strikes.pop(conversation_key, None):
                logger.info(f"[{conversation_key}] 🚧 Back on topic — off-topic strikes reset")
            return text

        strikes = (_strikes.get(conversation_key) or 0) + 1
        _strikes[conversation_key] = strikes

        if strikes > threshold:
            logger.info(
                f"[{conversation_key}] 🔇 Off-topic strike {strikes} > threshold {threshold} — muted"
            )
            return ""

        logger.info(
            f"[{conversation_key}] 🚧 Off-topic strike {strikes}/{threshold} — redirect sent"
        )
        return text

    def reset(self, conversation_key: str) -> None:
        """Clear strikes (e.g. when a human agent takes over)."""
        _strikes.pop(conversation_key, None)


scope_guard = ScopeGuard()
