from sqlalchemy import Column, String, Text, DateTime, JSON, Integer
from sqlalchemy.orm import DeclarativeBase
from datetime import datetime, timezone


class Base(DeclarativeBase):
    pass


class Document(Base):
    """
    Represents one document in the system.
    For now: an arXiv paper.
    Later: a legal document, contract, case file, etc.
    """
    __tablename__ = "documents"

    # Identity
    id = Column(String, primary_key=True)          # e.g. "2301.07041" for arXiv
    source = Column(String, nullable=False)         # "arxiv", "upload", "legal" etc.

    # Core metadata (source-agnostic)
    title = Column(String, nullable=False)
    authors = Column(JSON, default=list)            # list of author names
    abstract = Column(Text, nullable=True)
    published_date = Column(DateTime, nullable=True)
    url = Column(String, nullable=True)             # link to original document

    # Source-specific metadata (flexible JSON blob)
    extra_metadata = Column(JSON, default=dict)     # categories, journal, etc.

    # Parsed content
    full_text = Column(Text, nullable=True)         # full extracted text
    sections = Column(JSON, default=list)           # list of {title, content} dicts
    pdf_path = Column(String, nullable=True)        # local path to cached PDF

    # Processing status
    pdf_parsed = Column(String, default="pending")  # "pending", "success", "failed"
    parse_error = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc),
                        onupdate=lambda: datetime.now(timezone.utc))