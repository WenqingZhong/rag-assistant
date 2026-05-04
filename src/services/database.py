from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from src.models.document import Base
from src.config import get_settings
import logging

logger = logging.getLogger(__name__)


def get_engine():
    settings = get_settings()
    return create_engine(settings.postgres_database_url)


def create_tables():
    """Create all tables if they don't exist."""
    engine = get_engine()
    Base.metadata.create_all(engine)
    logger.info("Database tables created/verified")


def get_session() -> Session:
    engine = get_engine()
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()