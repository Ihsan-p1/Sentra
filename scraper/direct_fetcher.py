"""
Direct Article Fetcher for Indonesian English-Language News Sources
Uses hardcoded article URLs identified from search results.
Topic: Indonesia's Presidential Election / Prabowo Administration
"""
import requests
from bs4 import BeautifulSoup
import time
import re
from typing import List, Dict, Optional
from datetime import datetime
import json

# Election-related article URLs identified from search results
ELECTION_ARTICLES = {
    "antara": [
        "https://en.antaranews.com/news/400550/election-law-revision-will-not-regulate-permanent-political-coalitions",
        "https://en.antaranews.com/news/400506/prabowo-urges-electoral-system-to-prioritize-public-interest-minister",
        "https://en.antaranews.com/news/400502/indonesian-presidential-elections-will-remain-direct-amid-law-revision",
        "https://en.antaranews.com/news/386917/one-year-on-prabowo-bets-on-political-consolidation",
        "https://en.antaranews.com/news/401154/indonesias-president-highlights-prabowonomics-at-wef-2026-davos",
    ],
    "tempo": [
        "https://en.tempo.co/read/2083048/prabowos-nephew-steps-closer-to-secure-bank-indonesia-key-role",
        "https://en.tempo.co/read/2082548/prabowo-family-expands-political-footprint-in-indonesia",
        "https://en.tempo.co/read/2081986/indonesias-prabowo-calls-for-global-peace-at-davos",
    ],
    "jakarta_post": [
        "https://www.thejakartapost.com/indonesia/2024/10/20/prabowo-subianto-inaugurated-as-indonesias-eighth-president.html",
        "https://www.thejakartapost.com/opinion/2024/10/21/prabowo-presidency-what-to-expect.html",
    ],
    "jakarta_globe": [
        "https://jakartaglobe.id/news/prabowo-promises-change-in-first-speech-as-president",
        "https://jakartaglobe.id/news/cabinet-formation-prabowo-picks-ministers",
    ]
}

class DirectArticleFetcher:
    """Fetches articles directly from known working URLs."""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
    def _clean_text(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _fetch_page(self, url: str) -> Optional[BeautifulSoup]:
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            return BeautifulSoup(response.text, 'html.parser')
        except Exception as e:
            print(f"  [WARN] Failed to fetch {url}: {e}")
            return None
    
    def _parse_generic_article(self, url: str, media_source: str) -> Optional[Dict]:
        """Parse article using multiple selector strategies."""
        soup = self._fetch_page(url)
        if not soup:
            return None
            
        try:
            # Try multiple title selectors
            title = None
            for selector in ['h1.title', 'h1.post-title', 'h1', '.article-title']:
                title_elem = soup.select_one(selector)
                if title_elem:
                    title = title_elem.get_text(strip=True)
                    break
            
            if not title:
                title = "Untitled"
            
            # Try multiple content selectors
            content = ""
            for selector in ['.post-content', '.article-content', '.detail-content', 
                           '.content-inner', 'article', '.entry-content', '.post-body']:
                content_div = soup.select_one(selector)
                if content_div:
                    paragraphs = content_div.select('p')
                    content = ' '.join([p.get_text(strip=True) for p in paragraphs])
                    if len(content) > 100:
                        break
            
            # Fallback: get all paragraphs
            if len(content) < 100:
                all_paragraphs = soup.select('p')
                content = ' '.join([p.get_text(strip=True) for p in all_paragraphs if len(p.get_text(strip=True)) > 50])
            
            if not content or len(content) < 100:
                print(f"  [WARN] Content too short for {url}")
                return None
            
            print(f"  [OK] Fetched: {title[:50]}...")
            return {
                'title': self._clean_text(title),
                'content': self._clean_text(content)[:5000],  # Limit content length
                'media_source': media_source,
                'url': url,
                'published_date': datetime.now().isoformat()
            }
        except Exception as e:
            print(f"  [WARN] Error parsing {url}: {e}")
            return None
    
    def fetch_all(self) -> List[Dict]:
        """Fetch all articles from hardcoded URLs."""
        all_articles = []
        
        for media_source, urls in ELECTION_ARTICLES.items():
            print(f"\nFetching from {media_source.upper()}...")
            for url in urls:
                article = self._parse_generic_article(url, media_source)
                if article:
                    all_articles.append(article)
                time.sleep(1)  # Rate limiting
        
        print(f"\nTotal articles fetched: {len(all_articles)}")
        return all_articles


if __name__ == "__main__":
    fetcher = DirectArticleFetcher()
    articles = fetcher.fetch_all()
    
    # Save to JSON
    with open('data/scraped_articles.json', 'w', encoding='utf-8') as f:
        json.dump(articles, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved {len(articles)} articles to data/scraped_articles.json")
