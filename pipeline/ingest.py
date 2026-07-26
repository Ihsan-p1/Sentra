"""
Article Ingestion Pipeline
Ingests articles into PostgreSQL with embeddings.
"""
import asyncio
from typing import Any, Dict, List, Optional
from datetime import datetime
import json

import sys
sys.path.append('..')
from config.settings import settings
from database.connection import db_manager, init_database
from pipeline.preprocessing import ArticlePreprocessor, ProcessedChunk
from pipeline.embeddings import get_embedder


class ArticleIngestor:
    """
    Ingests news articles into the database.
    
    Pipeline:
    1. Preprocess article (clean, chunk)
    2. Generate embeddings for each chunk
    3. Store in PostgreSQL with pgvector
    """
    
    def __init__(self):
        self.preprocessor = ArticlePreprocessor()
        self.embedder = get_embedder()
    
    async def ingest_article(
        self,
        article: Dict[str, Any],
        db = None,
        corpus_version: Optional[str] = None
    ) -> int:
        """
        Ingest a single article.

        Args:
            article: Dict with keys:
                - title: str
                - content: str
                - media_source: str ('antaranews', 'tempo', 'abc_news')
                - url: str (optional)
                - author: str (optional)
                - published_date: datetime (optional)
                - category: str (optional)
            corpus_version: fingerprint of the corpus this article came from,
                stored on the row so the database can be traced back to it.

        Returns:
            article_id: int
        """
        db = db or db_manager

        # Validate media source
        if article.get('media_source') not in settings.SUPPORTED_MEDIA:
            raise ValueError(f"Unsupported media source: {article.get('media_source')}")

        # Insert article
        insert_article_query = """
            INSERT INTO articles (title, content, media_source, url, author, published_date, category, corpus_version)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            RETURNING id
        """

        article_id = await db.fetchval(
            insert_article_query,
            article.get('title', ''),
            article.get('content', ''),
            article.get('media_source'),
            article.get('url', ''),
            article.get('author', ''),
            article.get('published_date'),
            article.get('category', ''),
            corpus_version
        )

        # Preprocess and chunk
        chunks = self.preprocessor.chunk_article(article)

        if chunks:
            # Generate embeddings for all chunks
            chunk_texts = [c.chunk_text for c in chunks]
            embeddings = self.embedder.generate(chunk_texts, show_progress=False)

            # Insert chunks with embeddings. The model name is stored with every
            # vector - see the comment on article_chunks.embedding_model for why.
            insert_chunk_query = """
                INSERT INTO article_chunks
                    (article_id, chunk_index, chunk_text, embedding, embedding_model, embedding_dim, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
            """

            for chunk, embedding in zip(chunks, embeddings):
                await db.execute(
                    insert_chunk_query,
                    article_id,
                    chunk.chunk_index,
                    chunk.chunk_text,
                    embedding.tolist(),
                    settings.EMBEDDING_MODEL,
                    len(embedding),
                    json.dumps(chunk.metadata)
                )

        print(f"Ingested article: {article.get('title', '')[:50]}... ({len(chunks)} chunks)")
        return article_id

    async def ingest_batch(
        self,
        articles: List[Dict[str, Any]],
        db = None,
        corpus_version: Optional[str] = None
    ) -> List[int]:
        """Ingest multiple articles"""
        db = db or db_manager

        article_ids = []
        for article in articles:
            try:
                article_id = await self.ingest_article(article, db, corpus_version)
                article_ids.append(article_id)
            except Exception as e:
                print(f"[ERROR] Failed to ingest article: {e}")

        return article_ids

    async def verify_embeddings_match_model(self, db = None, sample_size: int = 5) -> Dict[str, Any]:
        """
        Check that the stored embeddings were produced by the CURRENT model.

        This failure is silent and expensive: EMBEDDING_MODEL was once changed
        from an English-only model to a multilingual one without re-ingesting.
        Both output 384 dimensions, so nothing crashed - retrieval simply compared
        query vectors against vectors from a different model, which compressed all
        similarity scores and broke cross-lingual matching.

        Two independent checks run, cheapest first:

        1. The recorded `embedding_model` label on every chunk row. Exact, covers
           the whole table, and costs one GROUP BY.
        2. Re-embedding a sample of chunk_text and comparing to the stored vector.
           Slower and only a sample, but it catches the case the label cannot: the
           name is right while the vectors are not, because the weights behind
           that name changed upstream.

        Returns a dict with `ok`, both check results, and a remediation hint.
        """
        db = db or db_manager
        expected = settings.EMBEDDING_MODEL

        label_rows = await db.fetch(
            "SELECT embedding_model, COUNT(*) AS n FROM article_chunks "
            "WHERE embedding IS NOT NULL GROUP BY embedding_model"
        )
        total = sum(row["n"] for row in label_rows)
        if total == 0:
            return {
                "ok": True,
                "checked": 0,
                "total_chunks": 0,
                "model": expected,
                "detail": "no chunks stored yet",
            }

        mismatched = {
            (row["embedding_model"] or "<unlabelled>"): row["n"]
            for row in label_rows
            if row["embedding_model"] != expected
        }
        if mismatched:
            described = ", ".join(f"{name} ({n} chunks)" for name, n in mismatched.items())
            return {
                "ok": False,
                "checked": 0,
                "total_chunks": total,
                "model": expected,
                "mismatched_models": mismatched,
                "detail": (
                    f"chunks were embedded by a different model - expected {expected}, "
                    f"found {described}. Run scripts/tools/refresh_database.py to re-ingest"
                ),
            }

        rows = await db.fetch(
            "SELECT chunk_text, embedding FROM article_chunks "
            "WHERE embedding IS NOT NULL LIMIT $1",
            sample_size,
        )

        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity

        similarities = []
        for row in rows:
            stored = np.array(row["embedding"], dtype=float)
            if len(stored) != self.embedder.dimension:
                return {
                    "ok": False,
                    "checked": len(similarities),
                    "total_chunks": total,
                    "model": expected,
                    "detail": (
                        f"dimension mismatch: stored {len(stored)}, "
                        f"model {self.embedder.dimension}"
                    ),
                }
            fresh = self.embedder.generate_single(row["chunk_text"])
            similarities.append(
                float(cosine_similarity(stored.reshape(1, -1), fresh.reshape(1, -1))[0][0])
            )

        worst = min(similarities) if similarities else 1.0
        # Re-embedding the same text with the same model is deterministic, so
        # anything below ~0.99 means the vectors came from a different model.
        ok = worst > 0.99

        return {
            "ok": ok,
            "checked": len(similarities),
            "total_chunks": total,
            "min_self_similarity": round(worst, 4),
            "model": expected,
            "detail": (
                "stored embeddings match the configured model"
                if ok
                else f"chunks are labelled {expected} but the vectors do not match it - "
                     "the model weights behind that name may have changed. "
                     "Run scripts/tools/refresh_database.py to re-ingest"
            ),
        }

    async def get_corpus_versions(self, db = None) -> Dict[str, int]:
        """Which corpus version(s) the stored articles came from, and how many each."""
        db = db or db_manager
        rows = await db.fetch(
            "SELECT corpus_version, COUNT(*) AS n FROM articles GROUP BY corpus_version"
        )
        return {(row["corpus_version"] or "<unversioned>"): row["n"] for row in rows}

    async def get_article_count(self, db = None) -> Dict[str, int]:
        """Get count of articles per media source"""
        db = db or db_manager
        
        query = """
            SELECT media_source, COUNT(*) as count
            FROM articles
            GROUP BY media_source
        """
        
        rows = await db.fetch(query)
        return {row['media_source']: row['count'] for row in rows}


async def ingest_sample_articles():
    """Ingest sample articles for testing"""
    from data.sample_articles import SAMPLE_ARTICLES

    await init_database()
    ingestor = ArticleIngestor()

    article_ids = await ingestor.ingest_batch(SAMPLE_ARTICLES, corpus_version="sample-articles")
    
    counts = await ingestor.get_article_count()
    print(f"\nArticle counts: {counts}")
    
    await db_manager.disconnect()
    return article_ids


if __name__ == "__main__":
    asyncio.run(ingest_sample_articles())
