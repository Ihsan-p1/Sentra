"""
Ingest articles from JSON file into the API - one at a time.
"""
import requests
import json
import time

API_BASE = "http://localhost:8000"

def ingest_articles():
    """Load articles from JSON and send to API one by one."""
    
    # Load articles
    with open('data/scraped_articles.json', 'r', encoding='utf-8') as f:
        articles = json.load(f)
    
    print(f"Loading {len(articles)} articles...")
    
    success = 0
    failed = 0
    
    for i, article in enumerate(articles):
        try:
            # Send single article
            response = requests.post(
                f"{API_BASE}/api/ingest",
                json={
                    "title": article["title"],
                    "content": article["content"],
                    "media_source": article["media_source"],
                    "url": article.get("url", ""),
                    "category": article.get("category", "Politics")
                },
                timeout=30
            )
            
            if response.status_code == 200:
                success += 1
                if (i + 1) % 10 == 0:
                    print(f"  [OK] Processed {i+1}/{len(articles)}")
            else:
                failed += 1
                print(f"  [ERROR] Failed article {i+1}: {response.status_code}")
                
        except Exception as e:
            failed += 1
            print(f"  [ERROR] Error on article {i+1}: {e}")
        
        # Small delay to avoid overwhelming the API
        time.sleep(0.1)
    
    print(f"\nResults:")
    print(f"  Success: {success}")
    print(f"  Failed: {failed}")

if __name__ == "__main__":
    ingest_articles()
