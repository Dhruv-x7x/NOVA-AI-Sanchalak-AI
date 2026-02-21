"""
SANCHALAK AI - Database Connection Manager
===============================================
Connection pool and session management for PostgreSQL.
"""

import logging
from typing import Optional, Generator
from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool

from .config import DatabaseConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Database connection manager with connection pooling.
    
    Features:
    - Connection pooling for performance
    - Session management with context managers
    - Health checks and connection validation
    - Automatic reconnection on failure
    """
    
    _instance: Optional['DatabaseManager'] = None
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        """
        Initialize database manager.
        
        Args:
            config: Database configuration (uses default if not provided)
        """
        self.config = config or DEFAULT_CONFIG
        self._engine = None
        self._session_factory = None
        self._initialized = False
    
    def initialize(self) -> None:
        """Initialize database engine and session factory."""
        if self._initialized:
            return
        
        try:
            self._engine = create_engine(
                self.config.connection_url,
                poolclass=QueuePool,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_timeout=self.config.pool_timeout,
                pool_recycle=self.config.pool_recycle,
                echo=self.config.echo,
            )
            
            self._session_factory = sessionmaker(
                bind=self._engine,
                autocommit=False,
                autoflush=False,
            )
            
            self._initialized = True
            logger.info(f"Database connected: {self.config.host}:{self.config.port}/{self.config.database}")
            
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    @property
    def engine(self):
        """Get SQLAlchemy engine."""
        if not self._initialized:
            self.initialize()
        return self._engine
    
    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        """
        Get a database session with automatic cleanup.
        
        Usage:
            with db_manager.session() as session:
                patients = session.query(Patient).all()
        """
        if not self._initialized:
            self.initialize()
        
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Session error: {e}")
            raise
        finally:
            session.close()
    
    def get_session(self) -> Session:
        """Get a new session (caller must manage lifecycle)."""
        if not self._initialized:
            self.initialize()
        return self._session_factory()
    
    def health_check(self) -> bool:
        """
        Check database connectivity.
        
        Returns:
            True if database is accessible
        """
        try:
            with self.session() as session:
                session.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def close(self) -> None:
        """Close database connections."""
        if self._engine:
            self._engine.dispose()
            self._initialized = False
            logger.info("Database connections closed")
    
    def create_tables(self, drop_existing: bool = False) -> None:
        """
        Create all database tables.
        
        Args:
            drop_existing: Whether to drop existing tables first
        """
        from .models import Base
        
        if drop_existing:
            Base.metadata.drop_all(self.engine)
            logger.warning("Dropped all existing tables")
        
        Base.metadata.create_all(self.engine)
        logger.info("Created all database tables")


# Singleton instance
_db_manager: Optional[DatabaseManager] = None


def get_db_manager() -> DatabaseManager:
    """
    Get singleton database manager instance.
    
    Returns:
        DatabaseManager instance
    """
    global _db_manager
    
    if _db_manager is None:
        _db_manager = DatabaseManager()
        _db_manager.initialize()
    
    return _db_manager


def reset_db_manager() -> None:
    """Reset the singleton instance (useful for testing)."""
    global _db_manager
    
    if _db_manager:
        _db_manager.close()
        _db_manager = None
