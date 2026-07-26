"""
Clear the database and re-ingest data/election_articles.json from scratch.

Run this whenever the corpus OR settings.EMBEDDING_MODEL changes. Changing the
embedding model without re-ingesting leaves the stored vectors in a different
embedding space from the query vectors, which degrades retrieval silently -
api/main.py now refuses to start quietly in that state.
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.settings import settings
from data.corpus import corpus_metadata, load_articles
from database.connection import db_manager, init_database
from pipeline.ingest import ArticleIngestor

async def clear_old_data():
    """Remove all existing articles and chunks from database"""
    print("Clearing old data from database...")

    await db_manager.execute("DELETE FROM article_chunks")
    await db_manager.execute("DELETE FROM articles")

    print("Old data cleared successfully.")

async def ingest_election_dataset():
    """Ingest the curated election articles dataset"""
    meta = corpus_metadata()
    articles = load_articles()

    print(f"Ingesting {meta['name']} ({len(articles)} articles)...")
    print(f"   corpus version:  {meta['version']}")
    print(f"   embedding model: {settings.EMBEDDING_MODEL}")

    ingestor = ArticleIngestor()
    article_ids = await ingestor.ingest_batch(articles, corpus_version=meta["version"])

    failed = len(articles) - len(article_ids)
    print(f"\nIngested {len(article_ids)} articles successfully.")
    if failed:
        print(f"[WARN] {failed} article(s) failed to ingest - see errors above.")

    # Show summary
    counts = await ingestor.get_article_count()
    print("\nArticles per media source:")
    for media, count in sorted(counts.items()):
        print(f"   - {media.upper()}: {count} articles")

    return ingestor

async def main():
    print("=" * 60)
    print("  SENTRA - Database Refresh Script")
    print("  Topic: Indonesia Presidential Election 2024 Aftermath")
    print("=" * 60)

    # Initialize database connection
    await init_database()

    # Clear old data
    await clear_old_data()

    # Ingest new dataset
    ingestor = await ingest_election_dataset()

    # Prove the vectors we just wrote are readable by the configured model.
    # Re-ingesting is exactly when this can go wrong, so verify here rather than
    # leaving it to the next server start.
    check = await ingestor.verify_embeddings_match_model()
    print(f"\nEmbedding check: {'OK' if check['ok'] else 'FAILED'} - {check['detail']}")
    if check.get("min_self_similarity") is not None:
        print(f"   min self-similarity: {check['min_self_similarity']} (expected 1.0)")

    versions = await ingestor.get_corpus_versions()
    print(f"   corpus versions in database: {versions}")

    # Disconnect
    await db_manager.disconnect()

    print("\nDatabase refresh complete." if check["ok"]
          else "\n[ERROR] Database refresh finished with a failed embedding check.")
    return 0 if check["ok"] else 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
