"""
Convert election_articles.py to JSON format for API ingestion.
"""
import json
import sys
sys.path.insert(0, '.')

from data.election_articles import ELECTION_ARTICLES

def convert_to_json():
    """Convert the curated articles to JSON format."""
    articles_json = []
    
    for article in ELECTION_ARTICLES:
        articles_json.append({
            "title": article["title"],
            "content": article["content"].strip(),
            "media_source": article["media_source"],
            "url": article.get("url", ""),
            "published_date": article["published_date"].isoformat() if hasattr(article["published_date"], 'isoformat') else str(article["published_date"]),
            "category": article.get("category", "Politics")
        })
    
    # Save to JSON
    with open('data/scraped_articles.json', 'w', encoding='utf-8') as f:
        json.dump(articles_json, f, indent=2, ensure_ascii=False)
    
    # Print summary by media source
    sources = {}
    for a in articles_json:
        src = a["media_source"]
        sources[src] = sources.get(src, 0) + 1
    
    print("Dataset Summary:")
    print("=" * 40)
    for src, count in sorted(sources.items()):
        print(f"  {src}: {count} articles")
    print("=" * 40)
    print(f"  TOTAL: {len(articles_json)} articles")
    print(f"\nSaved to data/scraped_articles.json")

if __name__ == "__main__":
    convert_to_json()
