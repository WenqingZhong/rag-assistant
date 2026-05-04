from fastapi import APIRouter, HTTPException
from src.services.database import get_session, create_tables
from src.models.document import Document

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


@router.get("/")
def list_documents(limit: int = 20, offset: int = 0):
    """List all documents in the database."""
    create_tables()
    session = get_session()
    try:
        docs = session.query(Document).offset(offset).limit(limit).all()
        return {
            "total": session.query(Document).count(),
            "documents": [
                {
                    "id": d.id,
                    "source": d.source,
                    "title": d.title,
                    "authors": d.authors,
                    "published_date": str(d.published_date),
                    "pdf_parsed": d.pdf_parsed,
                }
                for d in docs
            ]
        }
    finally:
        session.close()


@router.get("/{document_id}")
def get_document(document_id: str):
    """Get a single document by ID."""
    session = get_session()
    try:
        doc = session.query(Document).filter(Document.id == document_id).first()
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        return {
            "id": doc.id,
            "title": doc.title,
            "authors": doc.authors,
            "abstract": doc.abstract,
            "full_text": doc.full_text[:500] + "..." if doc.full_text else None,
            "sections": doc.sections,
            "pdf_parsed": doc.pdf_parsed,
        }
    finally:
        session.close()