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
    EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    EMBEDDING_DIMENSION: int = 384
    
    # LLM Configuration
    @property
    def GOOGLE_API_KEYS(self) -> list:
        # Default to empty list if None
        if not self.GOOGLE_API_KEY: return []
        return [k.strip() for k in self.GOOGLE_API_KEY.split(",") if k.strip()]
        
    GOOGLE_API_KEY: Optional[str] = None  # Loaded from .env (comma-separated)
    
    # Legacy/Unused
    GROQ_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    LLM_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta" 
    LLM_MODEL: str = "gemini-2.0-flash"
    LLM_TEMPERATURE: float = 0.3
    
    # RAG Settings
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    TOP_K_RETRIEVAL: int = 5
    
    # Hallucination Detection
    SIMILARITY_THRESHOLD: float = 0.65  # For rule-based model
    MIN_WORDS_FOR_CHECK: int = 4
    
    # Media Sources (English-language Indonesian news)
    SUPPORTED_MEDIA: list = ["antaranews", "tempo", "abc_news"]
    
    # Paths
    DATA_DIR: str = "data"
    MODELS_DIR: str = "data/models"
    LABELED_DIR: str = "data/labeled"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()
