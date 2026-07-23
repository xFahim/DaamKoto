"""Configuration settings for the application."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    facebook_verify_token: str
    facebook_app_secret: str | None = None      # Meta App Secret — enables X-Hub-Signature-256 verification
    internal_webhook_secret: str | None = None  # Shared secret for /internal/webhook/* endpoints
    gemini_api_key: str
    openai_api_key: str | None = None
    supabase_url: str
    supabase_service_role_key: str
    llm_provider: str = "gemini"  # "gemini" or "openai"
    # Agent models (override via OPENAI_MODEL / GEMINI_MODEL env vars).
    # Both verified available on the live keys 2026-07-15.
    openai_model: str = "gpt-5.4-mini"
    gemini_model: str = "gemini-3.6-flash"
    # Debounce window (seconds) for combining rapid multi-message bursts into
    # one agent run. Resets on every new message from the sender. Each reset
    # picks a fresh random duration in [min, max] so the bot's response timing
    # doesn't feel mechanical.
    message_batch_timeout: float = 8       # min of the window; env MESSAGE_BATCH_TIMEOUT
    message_batch_timeout_max: float = 12  # max of the window; env MESSAGE_BATCH_TIMEOUT_MAX
    conversation_ttl: int = 600
    conversation_max_turns: int = 5
    # Token-cost knobs (all env-overridable). History window: hard cap on
    # in-memory entries; summarization fires past the threshold and keeps the
    # last N verbatim. Images older than image_stale_after entries are demoted
    # to a text placeholder so they stop being re-billed on every model call.
    memory_max_messages: int = 20       # was 30; env MEMORY_MAX_MESSAGES
    summarize_threshold: int = 10       # was 15; env SUMMARIZE_THRESHOLD
    summarize_keep_last: int = 6        # was 8;  env SUMMARIZE_KEEP_LAST
    image_stale_after: int = 4          # history entries; env IMAGE_STALE_AFTER
    # Incoming photos are downscaled so the longest side ≤ this before upload.
    # 768 = one Gemini billing tile (~258 tokens) per image.
    image_max_dimension: int = 768      # env IMAGE_MAX_DIMENSION
    max_message_length: int = 500       # chars per individual message; override via MAX_MESSAGE_LENGTH
    rate_limit_messages: int = 15       # max messages per window; override via RATE_LIMIT_MESSAGES
    rate_limit_window: int = 60         # window in seconds; override via RATE_LIMIT_WINDOW

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()
