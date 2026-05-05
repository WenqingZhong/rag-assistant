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


class ChunkingSettings(BaseSettings):
    """
    Controls how papers are split into chunks before embedding.

    WHY these defaults?
    - chunk_size=600: ~one dense paragraph. Large enough for semantic context,
      small enough to stay within embedding model token limits and return
      focused results (not half a paper).
    - overlap_size=100: ~one short paragraph of overlap. Prevents key sentences
      near chunk boundaries from being split across two chunks with no shared
      context. Rule: overlap must be < chunk_size.
    - min_chunk_size=100: Chunks shorter than this get merged with neighbors.
      A 50-word chunk has too little signal for meaningful embedding.
    """
    model_config = SettingsConfigDict(env_prefix="CHUNKING__")
    chunk_size: int = 600
    overlap_size: int = 100
    min_chunk_size: int = 100


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_version: str = "0.1.0"
    service_name: str = "rag-api"
    debug: bool = True
    environment: str = "development"
    postgres_database_url: str = "postgresql+psycopg2://rag_user:rag_password@localhost:5432/rag_db"
    opensearch_host: str = "http://localhost:9200"
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:1b"
    ollama_timeout: int = 300

    # Jina AI embeddings API key — required for hybrid search.
    # Get a free key at https://jina.ai
    # Set via JINA_API_KEY in .env
    jina_api_key: str = ""

    opensearch: OpenSearchSettings = Field(default_factory=OpenSearchSettings)
    arxiv: ArxivSettings = Field(default_factory=ArxivSettings)
    pdf_parser: PDFParserSettings = Field(default_factory=PDFParserSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)


def get_settings() -> Settings:
    return Settings()