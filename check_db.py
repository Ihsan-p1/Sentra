import asyncio
import sys
import os

# Ensure we can import from parent directory
sys.path.append(os.getcwd())

from database.connection import db_manager

async def main():
    try:
        print("Connecting to DB...")
        await db_manager.connect()
        rows = await db_manager.fetch('SELECT DISTINCT media_source FROM articles')
        sources = [r['media_source'] for r in rows]
        print(f"MEDIA SOURCES IN DB: {sources}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await db_manager.disconnect()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
