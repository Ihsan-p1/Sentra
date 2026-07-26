"""
Sentra Configuration Settings
"""
import os
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # Database - PostgreSQL (embeddings stored as FLOAT8[], no pgvector)
    # Format: postgresql://user:password@host:port/dbname
    #
    # There is deliberately no DATABASE_NAME setting. One existed, hardcoded to
    # "Sentra1", and was used only in the connect log - so the log claimed
    # "Sentra1" no matter which database was actually opened. The name is now
    # derived from the URL by database.connection.database_name().
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/Sentra1")

    # Embedding Model (local, free)
    #
    # Changing this REQUIRES re-ingesting: stored vectors are only comparable to
    # query vectors from the same model. The dimension is read from the loaded
    # model rather than configured, because a configured value that disagreed with
    # the model would be worse than no value at all.
    EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    # LLM Configuration
    # Defaults target a local Ollama server; override in .env for any other
    # OpenAI-compatible provider. Local endpoints need no API key.
    LLM_BASE_URL: str = "http://localhost:11434/v1"
    LLM_MODEL: str = "sentra-qwen:3b"  # built from ollama/Modelfile
    LLM_TEMPERATURE: float = 0.3
    LLM_MAX_TOKENS: int = 2048
    LLM_TIMEOUT: int = 300  # seconds; local CPU inference is slow
    LLM_SEED: int = 42  # fixed seed keeps runs reproducible for evaluation

    # Only needed when LLM_BASE_URL points at a remote provider
    OPENROUTER_API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None

    # API security
    # /api/ingest writes into the corpus the chatbot answers from, so it is
    # gated by a shared secret whenever one is configured.
    INGEST_API_KEY: Optional[str] = None
    CORS_ALLOW_ORIGINS: list = ["http://localhost:8000", "http://127.0.0.1:8000"]

    # RAG Settings
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    TOP_K_RETRIEVAL: int = 8

    # How much retrieved text reaches the prompt. This is a genuine trade-off,
    # tuned by experiment rather than by "more context is better":
    #
    #   3 chunks - too few. Discarded the only chunk stating who won the 2024
    #              election, which ranks 4th-7th within its outlet.
    #   8 chunks - too many. The chunk WAS included, but sat 13th of 24 and the
    #              3B model answered "the sources do not mention a winner".
    #   5 chunks - the model finds and quotes it correctly.
    #
    # Prompt size is not the constraint (8 chunks was only ~2.0k tokens against an
    # 8192-token window); the constraint is a small model's attention over a long
    # context. Re-tune if the model changes.
    MAX_CONTEXT_CHUNKS_PER_SOURCE: int = 5
    MAX_CONTEXT_CHARS_PER_CHUNK: int = 500

    # Retrieval gate: refuse to answer when the best-matching chunk scores below
    # this. Calibrated against the corpus by scripts/calibrate_threshold.py -
    # do not change these by feel, re-run the calibration.
    #
    # Measured on 25 in-corpus + 8 out-of-corpus queries:
    #   in-corpus     min 0.329, median 0.611, max 0.773
    #   out-of-corpus max 0.343
    # The distributions OVERLAP by 0.014, so no threshold is error-free. 0.35 is
    # the lowest value that lets nothing unanswerable through; it wrongly refuses
    # 2 of 25 valid queries, both scoring 0.329. Dropping to 0.30 would admit them
    # and also 3 of 8 unanswerable ones.
    #
    # These numbers only became meaningful after the corpus was re-ingested: the
    # stored embeddings had been produced by a different model than the queries,
    # which compressed every score. An earlier calibration on those broken vectors
    # recommended 0.28 - a value that leaks 5 of 8 unanswerable queries once the
    # embeddings are correct. Always re-run the calibration after re-ingesting.
    RETRIEVAL_THRESHOLD: float = 0.35
    RETRIEVAL_THRESHOLD_STRICT: float = 0.42  # reduce_hallucination mode

    # Media Sources (English-language Indonesian news)
    SUPPORTED_MEDIA: list = ["antaranews", "tempo", "abc_news"]

    # Paths
    MODELS_DIR: str = "models"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # tolerate leftover keys (e.g. GOOGLE_API_KEY) in .env


# Global settings instance
settings = Settings()
