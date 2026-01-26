"""
Database Connection Manager (No pgvector dependency)
"""
import asyncio
from contextlib import asynccontextmanager
from typing import Optional
import asyncpg
import sys
sys.path.append('..')
from config.settings import settings

class DatabaseManager:
    """Manages PostgreSQL connection pool"""
    
    def __init__(self, database_url: str = None):
        self.database_url = database_url or settings.DATABASE_URL
        self.pool: Optional[asyncpg.Pool] = None
    
    async def connect(self):
        """Create connection pool"""
        try:
            self.pool = await asyncpg.create_pool(
                self.database_url,
                min_size=2,
                max_size=10
            )
            print(f"✅ Connected to database: {settings.DATABASE_NAME}")
        except Exception as e:
            print(f"❌ Failed to connect to database: {e}")
            raise e
        return self
    
    async def disconnect(self):
        """Close connection pool"""
        if self.pool:
            await self.pool.close()
            print("🔌 Database connection closed")
    
    @asynccontextmanager
    async def acquire(self):
        """Acquire a connection from the pool"""
        if not self.pool:
            await self.connect()
        async with self.pool.acquire() as conn:
            yield conn
    
    async def execute(self, query: str, *args):
        """Execute a query"""
        async with self.acquire() as conn:
            return await conn.execute(query, *args)
    
    async def fetch(self, query: str, *args):
        """Fetch multiple rows"""
        async with self.acquire() as conn:
            return await conn.fetch(query, *args)
    
    async def fetchrow(self, query: str, *args):
        """Fetch single row"""
        async with self.acquire() as conn:
            return await conn.fetchrow(query, *args)
            
    async def fetchval(self, query: str, *args):
        """Fetch single value"""
        async with self.acquire() as conn:
            return await conn.fetchval(query, *args)

# Singleton instance
db_manager = DatabaseManager()

async def init_database():
    """Initialize database with schema"""
    await db_manager.connect()
    
    # Read and execute schema
    schema_path = "database/schema.sql"
    try:
        with open(schema_path, 'r') as f:
            schema = f.read()
            
        async with db_manager.acquire() as conn:
            # Simple split by statement (naive but works for this schema)
            statements = schema.split(';')
            for stmt in statements:
                if stmt.strip():
                    try:
                        await conn.execute(stmt)
                    except Exception as e:
                        # Ignore benign errors
                        if 'already exists' not in str(e):
                            print(f"⚠️ Schema execution warning: {e}")
                            
        print("✅ Database schema initialized (Standard Array Mode)")
    except FileNotFoundError:
        print(f"⚠️ Scheama file not found: {schema_path}")
        
    return db_manager
