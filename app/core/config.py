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
    gemini_model: str = "gemini-3.5-flash"
    # Debounce window (seconds) for combining rapid multi-message bursts into
    # one agent run. Resets on every new message from the sender. Each reset
    # picks a fresh random duration in [min, max] so the bot's response timing
    # doesn't feel mechanical.
    message_batch_timeout: float = 8       # min of the window; env MESSAGE_BATCH_TIMEOUT
    message_batch_timeout_max: float = 12  # max of the window; env MESSAGE_BATCH_TIMEOUT_MAX
    conversation_ttl: int = 600
    conversation_max_turns: int = 5
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
