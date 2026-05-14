"""Application configuration from environment variables."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    gemini_api_key: str = ""
    # gemini-2.0-flash is not offered to new API keys; use a current Flash model.
    gemini_model: str = "gemini-2.5-flash"

    catalog_path: Path = Path("app/data/catalog.json")
    faiss_index_path: Path = Path("app/data/faiss.index")
    catalog_meta_path: Path = Path("app/data/catalog_meta.json")
    embeddings_path: Path = Path("app/data/embeddings.npy")

    max_conversation_turns: int = 8
    request_timeout_seconds: int = 25

    # Retrieval / ranking
    retrieval_top_k: int = 40
    final_recommendation_max: int = 10
    final_recommendation_min: int = 1


@lru_cache
def get_settings() -> Settings:
    return Settings()


def clear_settings_cache() -> None:
    get_settings.cache_clear()
