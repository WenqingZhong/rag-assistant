from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class OpenSearchSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OPENSEARCH__")
    host: str = "http://localhost:9200"
    index_name: str = "arxiv-papers"
    max_text_size: int = 1000000


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    debug: bool = True
    environment: str = "development"
    postgres_database_url: str = "postgresql+psycopg2://rag_user:rag_password@localhost:5432/rag_db"
    opensearch_host: str = "http://localhost:9200"
    ollama_host: str = "http://localhost:11434"

    opensearch: OpenSearchSettings = Field(default_factory=OpenSearchSettings)


def get_settings() -> Settings:
    return Settings()