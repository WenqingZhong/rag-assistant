import logging
from src.config import get_settings

logger = logging.getLogger(__name__)


class PDFParser:
    """
    Converts a PDF file into structured text using Docling.

    Why Docling and not PyPDF2 or pdfplumber?
    Academic PDFs (and legal documents) are NOT simple text files.
    They have multi-column layouts, complex tables, footnotes,
    mathematical notation, and headers/footers that break naive parsers.
    PyPDF2 often returns garbled text for these. Docling is a purpose-built
    ML-based parser that understands document structure.

    The tradeoff: Docling is slow (2-3 minutes per paper on CPU).
    That's acceptable for a background pipeline but never for a live request.
    This is why parsing happens in Airflow (scheduled, async) not in FastAPI.
    """

    def __init__(self):
        self.settings = get_settings().pdf_parser

    def parse(self, pdf_path: str) -> dict | None:
        """
        Parse a PDF file into structured content.

        Returns a dict with two keys:
            {
                "full_text": "entire document as markdown string",
                "sections": [{"title": "Introduction", "content": "..."}, ...]
            }
        Returns None if parsing fails entirely.

        Why return a dict instead of a dataclass?
        This gets stored directly as JSON in PostgreSQL (the 'sections' column),
        so a plain dict is the most convenient form.
        """
        try:
            # WHY import inside the function instead of at the top of the file?
            # Docling is a large ML library — importing it at module load time
            # adds several seconds to startup even when you're not parsing anything.
            # By importing lazily (only when parse() is actually called),
            # the API server starts up fast and only pays the import cost
            # when a parse job actually runs.
            from docling.document_converter import DocumentConverter

            logger.info(f"Parsing PDF: {pdf_path}")

            # DocumentConverter is Docling's main entry point.
            # It auto-detects the file format and applies the right pipeline.
            converter = DocumentConverter()

            # .convert() does the heavy lifting:
            # - reads the PDF bytes
            # - runs layout analysis (ML model)
            # - identifies sections, tables, figures
            # - extracts text in reading order
            result = converter.convert(pdf_path)

            # Export to markdown — this gives us clean, structured text
            # where headers are "# Introduction", tables are markdown tables, etc.
            # Much better than raw extracted text for LLM consumption later.
            full_text = result.document.export_to_markdown()

            # Attempt to extract section structure.
            # This is wrapped in its own try/except because sections are
            # OPTIONAL — if extraction fails, we still want the full_text.
            # Better to have imperfect data than no data.
            sections = []
            try:
                # iterate_items() walks the document tree node by node.
                # Each item has a 'label' (e.g. "section_header", "paragraph", "table").
                for item in result.document.iterate_items():
                    if hasattr(item, 'label') and 'section' in str(item.label).lower():
                        sections.append({
                            "title": getattr(item, 'text', ''),
                            "content": ""
                            # Note: content is empty for now — we populate it
                            # in Week 4 when we do proper chunking.
                        })
            except Exception:
                # Silently skip section extraction failure.
                pass

            logger.info(
                f"Successfully parsed PDF: {len(full_text)} chars, "
                f"{len(sections)} sections"
            )
            return {"full_text": full_text, "sections": sections}

        except Exception as e:
            # Outer exception handler — if Docling itself crashes
            # (malformed PDF, out of memory, unsupported format), we
            # return None rather than crashing the whole pipeline.
            # The orchestrator will mark this document as parse_status="failed".
            logger.error(f"Failed to parse PDF {pdf_path}: {e}")
            return None