import asyncio
from database.connection import db_manager, init_database

async def check_database():
    print("🔌 Connecting to database Sentra1...")
    try:
        await db_manager.connect()
        
        # 1. Total Articles per Media
        print("\n📊 Total Articles per Media:")
        rows = await db_manager.fetch("""
            SELECT media_source, COUNT(*) as count 
            FROM articles 
            GROUP BY media_source
        """)
        if not rows:
            print("   (No articles found)")
        for row in rows:
            print(f"   - {row['media_source'].upper()}: {row['count']} articles")
            
        # 2. Total Embeddings (Chunks)
        chunk_count = await db_manager.fetchval("SELECT COUNT(*) FROM article_chunks")
        print(f"\n🧩 Total Processed Chunks (Embeddings): {chunk_count}")
        
        # 3. Latest 5 Articles
        print("\n📰 Latest 5 Ingested Articles:")
        latest = await db_manager.fetch("""
            SELECT id, title, media_source, created_at 
            FROM articles 
            ORDER BY created_at DESC 
            LIMIT 5
        """)
        
        if not latest:
             print("   (No data)")
        
        for art in latest:
            print(f"   [{art['id']}] {art['media_source'].upper()} - {art['title']}")
            
        await db_manager.disconnect()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(check_database())
