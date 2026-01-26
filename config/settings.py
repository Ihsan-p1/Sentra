"""
Sentra Configuration Settings
"""
import os
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Database - PostgreSQL with pgvector
    # Format: postgresql://user:password@host:port/dbname
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/Sentra1")
    DATABASE_NAME: str = "Sentra1"
    
    # Embedding Model (local, free)
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    
    # LLM Configuration
    OPENAI_API_KEY: Optional[str] = None
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_TEMPERATURE: float = 0.3
    
    # RAG Settings
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    TOP_K_RETRIEVAL: int = 5
    
    # Hallucination Detection
    SIMILARITY_THRESHOLD: float = 0.65  # For rule-based model
    MIN_WORDS_FOR_CHECK: int = 4
    
    # Media Sources (English-language Indonesian news)
    SUPPORTED_MEDIA: list = ["jakarta_post", "tempo", "antara", "jakarta_globe"]
    
    # Paths
    DATA_DIR: str = "data"
    MODELS_DIR: str = "data/models"
    LABELED_DIR: str = "data/labeled"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()
