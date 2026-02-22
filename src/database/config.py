"""
a6on-i - Database Configuration
==========================================
Database connection settings and configuration management.
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(env_path)


@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    
    host: str = os.getenv('DB_HOST', 'localhost')
    port: int = int(os.getenv('DB_PORT', '5432'))
    database: str = os.getenv('DB_NAME', 'a6on-i_ai')
    username: str = os.getenv('DB_USER', 'postgres')
    password: str = os.getenv('DB_PASSWORD', 'postgres')
    
    # Connection pool settings
    pool_size: int = int(os.getenv('DB_POOL_SIZE', '10'))
    max_overflow: int = int(os.getenv('DB_MAX_OVERFLOW', '20'))
    pool_timeout: int = int(os.getenv('DB_POOL_TIMEOUT', '30'))
    pool_recycle: int = int(os.getenv('DB_POOL_RECYCLE', '1800'))
    
    # Echo SQL statements (for debugging)
    echo: bool = os.getenv('DB_ECHO', 'false').lower() == 'true'
    
    @property
    def connection_url(self) -> str:
        """Get SQLAlchemy connection URL."""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
    
    @property
    def async_connection_url(self) -> str:
        """Get async SQLAlchemy connection URL."""
        return f"postgresql+asyncpg://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"


def get_database_url(async_mode: bool = False) -> str:
    """
    Get database connection URL.
    
    Args:
        async_mode: Whether to use async driver
        
    Returns:
        SQLAlchemy connection URL
    """
    config = DatabaseConfig()
    return config.async_connection_url if async_mode else config.connection_url


# Default configuration instance
DEFAULT_CONFIG = DatabaseConfig()
