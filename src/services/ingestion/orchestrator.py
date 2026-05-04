import logging
from datetime import datetime, timezone
from sqlalchemy.dialects.postgresql import insert
from src.services.database import get_session, create_tables
from src.services.ingestion.fetcher import ArxivFetcher
from src.services.ingestion.downloader import PDFDownloader
from src.services.ingestion.parser import PDFParser
from src.models.document import Document

logger = logging.getLogger(__name__)


class IngestionOrchestrator:
    """
    Coordinates the full ingestion pipeline end to end:
        Fetch metadata → Download PDF → Parse PDF → Store in PostgreSQL

    This is the only class the outside world (Airflow DAG, API endpoints,
    manual scripts) needs to interact with. The fetcher, downloader, and
    parser are internal implementation details.

    Design pattern: this is a Facade — it hides the complexity of
    four separate steps behind one simple .run() method.
    This follow the Single Responsibility Principle
    """

    def __init__(self):
        self.fetcher = ArxivFetcher()
        self.downloader = PDFDownloader()
        self.parser = PDFParser()

    def run(self, days_back: int = 1) -> dict:
        """
        Run the full pipeline. Returns a stats summary dict.

        Parameters:
            days_back: how many days of papers to fetch.
                       1 = yesterday's papers (used by daily Airflow DAG)
                       7 = last week (useful for initial backfill)

        Returns:
            {"fetched": 15, "stored": 15, "parsed": 12, "failed": 3}
        """
        # Ensure the 'documents' table exists in PostgreSQL.
        # Safe to call every time — it's a no-op if tables already exist.
        create_tables()

        logger.info(f"Starting ingestion pipeline (last {days_back} days)")

        # Step 1: Fetch metadata from arXiv API.
        # Returns a list of DocumentMetadata objects — no DB or files yet.
        docs = self.fetcher.fetch_recent(days_back=days_back)

        # Tracking counters for the summary report.
        stats = {
            "fetched": len(docs),   # how many papers the API returned
            "stored": 0,            # how many we successfully saved to DB
            "parsed": 0,            # how many PDFs were fully parsed
            "failed": 0,            # how many had errors
        }

        # Process each document through the pipeline.
        # We process sequentially (one at a time) for simplicity.
        for doc_meta in docs:
            try:
                # Step 2: Download the PDF to local disk.
                # Returns the local file path, or None if download failed.
                pdf_path = None
                if doc_meta.url:
                    pdf_path = self.downloader.download(doc_meta.id, doc_meta.url)

                # Step 3: Parse the PDF into structured text.
                # We only attempt parsing if we have a local file.
                parsed = None
                parse_status = "pending"    # default — means "not attempted"
                parse_error = None

                if pdf_path:
                    parsed = self.parser.parse(pdf_path)
                    if parsed:
                        parse_status = "success"
                        stats["parsed"] += 1
                    else:
                        # Parser returned None — it logged the reason internally.
                        parse_status = "failed"
                        parse_error = "Docling returned no content"
                        stats["failed"] += 1

                # Step 4: Store everything in PostgreSQL.
                # We store the document even if PDF parsing failed —
                # we still have the metadata (title, authors, abstract)
                # which is useful for search even without full text.
                self._upsert_document(
                    doc_meta, pdf_path, parsed, parse_status, parse_error
                )
                stats["stored"] += 1

            except Exception as e:
                # If anything unexpected goes wrong with one document,
                # log it and continue to the next one.
                logger.error(f"Failed to process document {doc_meta.id}: {e}")
                stats["failed"] += 1

        logger.info(f"Ingestion complete: {stats}")
        return stats

    def _upsert_document(
        self,
        doc_meta,
        pdf_path,
        parsed,
        parse_status,
        parse_error
    ):
        """
        Insert a new document or update it if it already exists.

        WHY UPSERT instead of INSERT?
        If you run the pipeline twice on the same day, or Airflow retries
        a failed run, you'd get duplicate rows with plain INSERT.
        Upsert means: "insert if new, update if already exists."
        SQL equivalent: INSERT ... ON CONFLICT DO UPDATE
        """
        # Get a new database session (connection + transaction).
        session = get_session()
        try:
            now = datetime.now(timezone.utc)

            # Build the full row as a plain dict.
            # Every key maps to a column in the 'documents' table.
            values = {
                "id": doc_meta.id,
                "source": doc_meta.source,
                "title": doc_meta.title,
                "authors": doc_meta.authors,          # stored as JSON array
                "abstract": doc_meta.abstract,
                "published_date": doc_meta.published_date,
                "url": doc_meta.url,
                "extra_metadata": doc_meta.extra,     # stored as JSON object
                "pdf_path": pdf_path,
                "pdf_parsed": parse_status,           # "success", "failed", "pending"
                "parse_error": parse_error,
                # parsed is None if PDF download/parse failed,
                # so we safely fall back to None/empty list
                "full_text": parsed["full_text"] if parsed else None,
                "sections": parsed["sections"] if parsed else [],
                "created_at": now,
                "updated_at": now,
            }

            # Build the INSERT statement using SQLAlchemy's PostgreSQL-specific
            # insert() — this supports ON CONFLICT which standard insert() doesn't.
            stmt = insert(Document).values(values)

            # ON CONFLICT DO UPDATE:
            # "If a row with this 'id' already exists, update these columns."
            # index_elements=["id"] tells it which column to check for conflict.
            # set_= is a dict of {column: new_value} to update on conflict.
            # We exclude "id" from the update dict because the ID never changes.
            stmt = stmt.on_conflict_do_update(
                index_elements=["id"],
                set_={k: v for k, v in values.items() if k != "id"}
                # dict comprehension: Java equivalent of values.entrySet().stream()
                #     .filter(e -> !e.getKey().equals("id"))
                #     .collect(Collectors.toMap(...))
            )

            # Execute the statement — sends the SQL to PostgreSQL.
            session.execute(stmt)

            # Commit the transaction — makes the change permanent.
            # Without commit(), the change would be rolled back when
            # the session closes.
            session.commit()

        except Exception as e:
            # If anything goes wrong, roll back the transaction.
            # This ensures we never leave the DB in a partial/corrupt state.
            session.rollback()
            # Re-raise so the caller (run()) knows this document failed
            # and can update the stats accordingly.
            raise e

        finally:
            # Always close the session whether we succeeded or failed.
            session.close()