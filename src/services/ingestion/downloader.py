import time
import logging
from pathlib import Path
import requests
from src.config import get_settings

logger = logging.getLogger(__name__)


class PDFDownloader:
    """
    Downloads PDFs to a local cache directory on disk.

    The key design principle here is CACHING — if we already downloaded
    a PDF, we never download it again. This matters because:
    1. arXiv rate-limits us
    2. PDFs can be 10-20MB each
    3. The pipeline might be re-run (manually or by Airflow retrying)

    """

    def __init__(self):
        self.settings = get_settings()
        self.cache_dir = Path(self.settings.arxiv.pdf_cache_dir)

        # mkdir(parents=True) creates the full path including parent dirs.
        # exist_ok=True means "don't throw an error if it already exists."
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def download(self, document_id: str, pdf_url: str) -> str | None:
        """
        Download a PDF. Returns the local file path as a string, or None if failed.

        Parameters:
            document_id: the arXiv ID, e.g. "2301.07041"
            pdf_url: the direct URL to the PDF on arXiv's servers
        """

        # Build the local file path where we'll save this PDF.
        # We replace "/" with "_" because arXiv IDs like "2301.07041v1"
        # are safe, but some IDs have slashes which would create subdirectories.
        # e.g. document_id="2301.07041" → filename="2301.07041.pdf"
        local_path = self.cache_dir / f"{document_id.replace('/', '_')}.pdf"
    
        # Cache hit — we already have this file, no need to download again.
        if local_path.exists():
            logger.info(f"PDF already cached: {local_path}")
            return str(local_path)  # return path as plain string for storage in DB

        # Retry loop — network requests fail sometimes. We try up to
        # download_max_retries times (default: 3) before giving up.
        for attempt in range(self.settings.arxiv.download_max_retries):
            try:
                logger.info(f"Downloading PDF: {pdf_url} (attempt {attempt + 1})")

                # stream=True means: don't download the whole file into memory
                # at once. Instead, stream it in chunks. Important for large files.
                response = requests.get(
                    pdf_url,
                    timeout=self.settings.arxiv.timeout_seconds,
                    stream=True
                )
                response.raise_for_status()  # throws if HTTP 4xx/5xx

                # Read the full content into memory now.
                # For very large files you'd stream to disk directly,
                # but for PDFs up to 20MB this is fine.
                content = response.content

                # Size check — skip PDFs that are too large to parse reasonably.
                # 1024 * 1024 = 1MB in bytes, so this converts bytes → MB.
                size_mb = len(content) / (1024 * 1024)
                if size_mb > self.settings.pdf_parser.max_file_size_mb:
                    logger.warning(f"PDF too large ({size_mb:.1f}MB), skipping")
                    return None

                # Write the bytes to disk.
                # write_bytes() creates the file if it doesn't exist.
                local_path.write_bytes(content)
                logger.info(f"Downloaded PDF: {local_path} ({size_mb:.1f}MB)")
                return str(local_path)

            except Exception as e:
                # Exponential backoff — each retry waits longer than the last.
                # attempt=0 → wait 5s, attempt=1 → wait 10s, attempt=2 → wait 20s
                # Formula: base_delay * (2 ^ attempt)
                # This is standard practice for retrying network requests —
                # if the server is struggling, hammering it immediately makes it worse.
                wait = self.settings.arxiv.download_retry_delay_base * (2 ** attempt)
                logger.warning(
                    f"Download failed (attempt {attempt + 1}): {e}. Waiting {wait}s"
                )
                # Only sleep if there are more retries left 
                if attempt < self.settings.arxiv.download_max_retries - 1:
                    time.sleep(wait)

        # All retries exhausted — log and return None.
        # Returning None (instead of throwing) lets the orchestrator decide
        # what to do: mark the document as "failed" and move on.
        logger.error(f"Failed to download PDF after all retries: {pdf_url}")
        return None