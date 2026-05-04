from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class OpenSearchSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OPENSEARCH__")
    host: str = "http://localhost:9200"
    index_name: str = "documents"
    max_text_size: int = 1000000


class ArxivSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ARXIV__")
    max_results: int = 5
    base_url: str = "https://export.arxiv.org/api/query"
    pdf_cache_dir: str = "./data/arxiv_pdfs"
    rate_limit_delay: float = 3.0
    timeout_seconds: int = 30
    search_category: str = "cs.AI"
    download_max_retries: int = 3
    download_retry_delay_base: float = 5.0
    max_concurrent_downloads: int = 5


class PDFParserSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PDF_PARSER__")
    max_pages: int = 30
    max_file_size_mb: int = 20
    do_ocr: bool = False
    do_table_structure: bool = False


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    debug: bool = True
    environment: str = "development"
    postgres_database_url: str = "postgresql+psycopg2://rag_user:rag_password@localhost:5432/rag_db"
    opensearch_host: str = "http://localhost:9200"
    ollama_host: str = "http://localhost:11434"

    opensearch: OpenSearchSettings = Field(default_factory=OpenSearchSettings)
    arxiv: ArxivSettings = Field(default_factory=ArxivSettings)
    pdf_parser: PDFParserSettings = Field(default_factory=PDFParserSettings)


def get_settings() -> Settings:
    return Settings()