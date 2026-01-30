"""
Article Ingestion Pipeline
Ingests articles into PostgreSQL with embeddings.
"""
import asyncio
from typing import List, Dict, Any, Optional
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
        db = None
    ) -> int:
        """
        Ingest a single article.
        
        Args:
            article: Dict with keys:
                - title: str
                - content: str
                - media_source: str ('kompas', 'tempo', 'bbc')
                - url: str (optional)
                - author: str (optional)
                - published_date: datetime (optional)
                - category: str (optional)
        
        Returns:
            article_id: int
        """
        db = db or db_manager
        
        # Validate media source
        if article.get('media_source') not in settings.SUPPORTED_MEDIA:
            raise ValueError(f"Unsupported media source: {article.get('media_source')}")
        
        # Insert article
        insert_article_query = """
            INSERT INTO articles (title, content, media_source, url, author, published_date, category)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
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
            article.get('category', '')
        )
        
        # Preprocess and chunk
        chunks = self.preprocessor.chunk_article(article)
        
        if chunks:
            # Generate embeddings for all chunks
            chunk_texts = [c.chunk_text for c in chunks]
            embeddings = self.embedder.generate(chunk_texts, show_progress=False)
            
            # Insert chunks with embeddings
            insert_chunk_query = """
                INSERT INTO article_chunks (article_id, chunk_index, chunk_text, embedding, metadata)
                VALUES ($1, $2, $3, $4, $5)
            """
            
            for chunk, embedding in zip(chunks, embeddings):
                await db.execute(
                    insert_chunk_query,
                    article_id,
                    chunk.chunk_index,
                    chunk.chunk_text,
                    embedding.tolist(),
                    json.dumps(chunk.metadata)
                )
        
        print(f"Ingested article: {article.get('title', '')[:50]}... ({len(chunks)} chunks)")
        return article_id
    
    async def ingest_batch(
        self, 
        articles: List[Dict[str, Any]],
        db = None
    ) -> List[int]:
        """Ingest multiple articles"""
        db = db or db_manager
        
        article_ids = []
        for article in articles:
            try:
                article_id = await self.ingest_article(article, db)
                article_ids.append(article_id)
            except Exception as e:
                print(f"[ERROR] Failed to ingest article: {e}")
        
        return article_ids
    
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
    
    article_ids = await ingestor.ingest_batch(SAMPLE_ARTICLES)
    
    counts = await ingestor.get_article_count()
    print(f"\nArticle counts: {counts}")
    
    await db_manager.disconnect()
    return article_ids


if __name__ == "__main__":
    asyncio.run(ingest_sample_articles())
