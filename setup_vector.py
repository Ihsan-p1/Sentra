import asyncio
import asyncpg
from config.settings import settings

async def setup_db():
    try:
        conn = await asyncpg.connect(settings.DATABASE_URL)
        print("[INFO] Database connected successfully.")
        
        # Try to enable pgvector expansion
        try:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            print("[INFO] 'vector' extension enabled successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to enable 'vector' extension: {e}")
            print("\n[WARN] Possible issue: pgvector is not installed on this PostgreSQL instance.")
            print("[INFO] Since you are on Windows, installing pgvector can be complex.")
            print("[INFO] Consider switching to 'Simulated Vector Search' mode (Python-based) which doesn't require pgvector.")
            
        await conn.close()
    except Exception as e:
        print(f"[ERROR] Connection error: {e}")

if __name__ == "__main__":
    asyncio.run(setup_db())
