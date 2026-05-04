import logging
from src.config import get_settings

logger = logging.getLogger(__name__)


class PDFParser:
    """
    Converts a PDF file into plain text using pypdfium2 (Google PDFium).

    WHY pypdfium2 instead of Docling?
    Docling's StandardPdfPipeline loads a ~1-2 GB layout-analysis ML model.
    In a Docker container on a laptop (typically 4-8 GB total RAM shared with
    PostgreSQL, OpenSearch, and Airflow itself), that model consistently
    triggers the OOM killer. pypdfium2 is a thin Python binding over PDFium
    — Google's production PDF renderer — and extracts text in milliseconds
    with negligible memory usage and no ML inference at all.

    The tradeoff: text ordering in complex multi-column layouts may be less
    accurate than Docling's ML-based approach. For BM25 keyword search (the
    current use case), clean extracted text is all that matters — layout
    perfection is not required.

    Docling can be re-enabled by swapping this class out once the deployment
    environment has sufficient RAM (16+ GB available to the container).
    """

    def __init__(self):
        self.settings = get_settings().pdf_parser

    def parse(self, pdf_path: str) -> dict | None:
        """
        Parse a PDF file into plain text.

        Returns:
            {"full_text": str, "sections": []} on success
            None if the file cannot be read (corrupt, missing, etc.)

        WHY return a dict instead of a dataclass?
        Stored directly as JSON in PostgreSQL — plain dict is the most
        convenient form for both SQLAlchemy and JSON serialisation.
        """
        try:
            # Lazy import: pypdfium2 is fast to import but we keep the pattern
            # consistent — nothing at module level that delays the DAG processor.
            import pypdfium2 as pdfium

            logger.info(f"Parsing PDF: {pdf_path}")

            pdf = pdfium.PdfDocument(pdf_path)
            num_pages = min(len(pdf), self.settings.max_pages)

            page_texts = []
            for i in range(num_pages):
                page = pdf[i]
                textpage = page.get_textpage()
                text = textpage.get_text_range()
                if text and text.strip():
                    page_texts.append(text.strip())

            full_text = "\n\n".join(page_texts)

            if not full_text:
                logger.warning(f"No text extracted from {pdf_path} — may be image-only")
                return None

            logger.info(
                f"Successfully parsed PDF: {len(full_text)} chars, "
                f"{num_pages} pages"
            )
            return {"full_text": full_text, "sections": []}

        except Exception as e:
            logger.error(f"Failed to parse PDF {pdf_path}: {e}")
            return None
