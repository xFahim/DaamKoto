"""Configuration settings for the application."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    facebook_verify_token: str
    facebook_page_access_token: str
    gemini_api_key: str
    openai_api_key: str | None = None
    pinecone_api_key: str
    gcp_service_account_json: str | None = None
    llm_provider: str = "gemini"  # "gemini" or "openai"
    message_batch_timeout: float = 4
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
