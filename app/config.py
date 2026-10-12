from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_JWT_SECRET = "change-me-in-dev-only"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Audio Streaming & RAG Discovery API"
    environment: str = "dev"

    # Security
    jwt_secret: str = DEFAULT_JWT_SECRET
    jwt_algorithm: str = "HS256"
    access_token_minutes: int = 60
    bcrypt_rounds: int = 12

    # Data stores
    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db: str = "audio_rag"
    redis_url: str | None = "redis://localhost:6379/0"
    vector_backend: str = "pinecone"  # pinecone | memory (offline dev/tests, nothing persisted)
    pinecone_api_key: str | None = None
    pinecone_index: str = "audio-rag"  # the embedder name is appended, e.g. audio-rag-pinecone-e5
    pinecone_namespace: str = "songs"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    # RAG
    embedding_backend: str = "pinecone"  # pinecone (hosted multilingual-e5-large) | hash (offline)
    embedding_dim: int = 384  # used by the hash backend only
    embedding_cache_ttl: int = 86400
    llm_provider: str = "none"  # none | anthropic | openai
    llm_api_key: str | None = None
    llm_model: str = "claude-haiku-5-5"
    llm_base_url: str = "https://api.openai.com/v1"
    llm_timeout_seconds: float = 8.0

    # Audio
    audio_dir: str = "./data/audio"
    max_upload_mb: int = 50
    stream_chunk_size: int = 64 * 1024

    # Rate limiting
    rate_limit_requests: int = 120
    rate_limit_window_seconds: int = 60
    auth_rate_limit_requests: int = 10
    trust_proxy_headers: bool = False

    cors_origins: str = "http://localhost:5173"

    def model_post_init(self, __context) -> None:  # noqa: ANN001
        if self.environment != "dev" and self.jwt_secret == DEFAULT_JWT_SECRET:
            raise ValueError("JWT_SECRET must be set to a unique value outside the dev environment")
        if not self.redis_url:
            self.redis_url = None

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
