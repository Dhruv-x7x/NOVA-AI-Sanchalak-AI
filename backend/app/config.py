"""
FastAPI Application Configuration
"""
import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment"""
    
    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env"),
        case_sensitive=True,
        extra="ignore"
    )
    
    # App
    APP_NAME: str = "Sanchalak AI API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # Production Mode - enforces strict authentication
    TEST_MODE: bool = False  # Set to True for testing, False for production
    ALLOW_ANONYMOUS_ACCESS: bool = False  # Allow requests without auth in test mode
    
    # Security
    SECRET_KEY: str = "prod-sanchalak-ai-secure-key-2024-v1"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # Database
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "trialpulse_nexus"
    DB_USER: str = "postgres"
    DB_PASSWORD: str = "postgres"
    
    # Neo4j
    NEO4J_ENABLED: bool = False
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "password"
    
    # CORS
    CORS_ORIGINS: list = ["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173", "*"]
    
    @property
    def DATABASE_URL(self) -> str:
        return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
