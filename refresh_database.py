"""
Script to clear old data and ingest the new election articles dataset.
"""
import asyncio
import sys
sys.path.insert(0, '.')

from database.connection import db_manager, init_database
from pipeline.ingest import ArticleIngestor
from data.election_articles import ELECTION_ARTICLES

async def clear_old_data():
    """Remove all existing articles and chunks from database"""
    print("Clearing old data from database...")
    
    await db_manager.execute("DELETE FROM article_chunks")
    await db_manager.execute("DELETE FROM articles")
    
    print("Old data cleared successfully.")

async def ingest_election_dataset():
    """Ingest the curated election articles dataset"""
    print("Ingesting election aftermath dataset...")
    
    ingestor = ArticleIngestor()
    article_ids = await ingestor.ingest_batch(ELECTION_ARTICLES)
    
    print(f"\nIngested {len(article_ids)} articles successfully.")
    
    # Show summary
    counts = await ingestor.get_article_count()
    print("\nArticles per media source:")
    for media, count in counts.items():
        print(f"   - {media.upper()}: {count} articles")
    
    return article_ids

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
    await ingest_election_dataset()
    
    # Disconnect
    await db_manager.disconnect()
    
    print("\nDatabase refresh complete.")

if __name__ == "__main__":
    asyncio.run(main())
