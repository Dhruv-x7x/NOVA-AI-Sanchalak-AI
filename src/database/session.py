
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from .models import Base

# Database Configuration - PostgreSQL ONLY
# No SQLite fallback - fail fast if PostgreSQL is not configured

# Get PostgreSQL URL from environment or use default local connection
DB_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/a6on-i_ai")

if not DB_URL.startswith("postgresql"):
    raise ValueError(
        "DATABASE_URL must be a PostgreSQL connection string. "
        "SQLite is no longer supported. "
        f"Current value: {DB_URL}"
    )

# Create PostgreSQL engine with connection pooling
engine = create_engine(
    DB_URL,
    pool_pre_ping=True,  # Verify connections before using
    pool_size=10,
    max_overflow=20,
    echo=False  # Set to True for SQL debugging
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """Initialize database schema."""
    Base.metadata.create_all(bind=engine)

def get_db():
    """Dependency for DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
