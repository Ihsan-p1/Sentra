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

def database_name(url: str) -> str:
    """
    The database a URL actually points at.

    The connect log used to print settings.DATABASE_NAME, a separate hardcoded
    constant. Any DatabaseManager built with an explicit URL - the integration
    tests, a CI database, a staging box - therefore logged "Connected to
    database: Sentra1" while connected somewhere else entirely. A log line about
    which database you are on is worthless if it cannot be wrong.
    """
    return (url or "").rstrip("/").rsplit("/", 1)[-1].split("?")[0] or "unknown"


class DatabaseManager:
    """Manages PostgreSQL connection pool"""

    def __init__(self, database_url: str = None):
        self.database_url = database_url or settings.DATABASE_URL
        self.pool: Optional[asyncpg.Pool] = None

    @property
    def name(self) -> str:
        return database_name(self.database_url)

    async def connect(self):
        """Create connection pool"""
        try:
            self.pool = await asyncpg.create_pool(
                self.database_url,
                min_size=2,
                max_size=10
            )
            print(f"[INFO] Connected to database: {self.name}")
        except Exception as e:
            print(f"[ERROR] Failed to connect to database {self.name}: {e}")
            raise e
        return self
    
    async def disconnect(self):
        """Close connection pool"""
        if self.pool:
            await self.pool.close()
            print("Database connection closed.")
    
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


def split_sql_statements(schema: str) -> list:
    """
    Split a schema file into executable statements.

    A plain schema.split(';') breaks plpgsql function bodies, which contain their
    own semicolons between the $$ delimiters, so dollar-quoted blocks are tracked
    and kept intact.
    """
    statements = []
    current = []
    in_dollar_block = False

    for line in schema.splitlines():
        # Each line either opens or closes a $$ block (an odd number of markers)
        if line.count('$$') % 2 == 1:
            in_dollar_block = not in_dollar_block

        current.append(line)

        if not in_dollar_block and line.rstrip().endswith(';'):
            stmt = "\n".join(current).strip()
            if stmt:
                statements.append(stmt)
            current = []

    tail = "\n".join(current).strip()
    if tail:
        statements.append(tail)

    return statements

async def init_database():
    """Initialize database with schema"""
    await db_manager.connect()
    
    # Read and execute schema
    schema_path = "database/schema.sql"
    try:
        with open(schema_path, 'r') as f:
            schema = f.read()
            
        async with db_manager.acquire() as conn:
            for stmt in split_sql_statements(schema):
                try:
                    await conn.execute(stmt)
                except Exception as e:
                    # Ignore benign errors
                    if 'already exists' not in str(e):
                        print(f"[WARN] Schema execution warning: {e}")
                            
        print("Database schema initialized (Standard Array Mode).")
    except FileNotFoundError:
        print(f"[WARN] Schema file not found: {schema_path}")
        
    return db_manager
