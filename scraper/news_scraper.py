"""
News Scraper for Indonesian English-Language News Sources
Topic: Indonesia's Presidential Election Aftermath
"""
import requests
from bs4 import BeautifulSoup
import time
import re
from typing import List, Dict, Optional
from datetime import datetime
import json

class NewsScraper:
    """
    Scrapes news articles from various Indonesian English news sources.
    """
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        if not text:
            return ""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:\-\'\"()\[\]]', '', text)
        return text.strip()
    
    def _fetch_page(self, url: str) -> Optional[BeautifulSoup]:
        """Fetch and parse a webpage"""
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            return BeautifulSoup(response.text, 'html.parser')
        except Exception as e:
            print(f"  ⚠️ Failed to fetch {url}: {e}")
            return None
    
    # =========================================
    # THE JAKARTA POST
    # =========================================
    def scrape_jakarta_post(self, query: str = "presidential election", max_articles: int = 10) -> List[Dict]:
        """Scrape articles from The Jakarta Post"""
        print(f"\n📰 Scraping The Jakarta Post...")
        articles = []
        
        # Search URL
        search_url = f"https://www.thejakartapost.com/search?q={query.replace(' ', '+')}"
        soup = self._fetch_page(search_url)
        
        if not soup:
            return articles
            
        # Find article links
        article_links = []
        for link in soup.select('a[href*="/paper/"]'):
            href = link.get('href', '')
            if href and href not in article_links:
                if not href.startswith('http'):
                    href = f"https://www.thejakartapost.com{href}"
                article_links.append(href)
                
        article_links = article_links[:max_articles]
        print(f"  Found {len(article_links)} article links")
        
        for url in article_links:
            article = self._parse_jakarta_post_article(url)
            if article:
                articles.append(article)
            time.sleep(1)  # Rate limiting
            
        return articles
    
    def _parse_jakarta_post_article(self, url: str) -> Optional[Dict]:
        """Parse a single Jakarta Post article"""
        soup = self._fetch_page(url)
        if not soup:
            return None
            
        try:
            title = soup.select_one('h1')
            title = title.get_text(strip=True) if title else "Untitled"
            
            # Get article body
            content_div = soup.select_one('article') or soup.select_one('.detail-content')
            if content_div:
                paragraphs = content_div.select('p')
                content = ' '.join([p.get_text(strip=True) for p in paragraphs])
            else:
                content = ""
                
            if not content or len(content) < 100:
                return None
                
            return {
                'title': self._clean_text(title),
                'content': self._clean_text(content),
                'media_source': 'jakarta_post',
                'url': url,
                'published_date': datetime.now()
            }
        except Exception as e:
            print(f"  ⚠️ Error parsing {url}: {e}")
            return None
    
    # =========================================
    # ANTARA NEWS ENGLISH
    # =========================================
    def scrape_antara_news(self, query: str = "presidential election", max_articles: int = 10) -> List[Dict]:
        """Scrape articles from ANTARA News English"""
        print(f"\n📰 Scraping ANTARA News English...")
        articles = []
        
        search_url = f"https://en.antaranews.com/search?q={query.replace(' ', '+')}"
        soup = self._fetch_page(search_url)
        
        if not soup:
            return articles
            
        article_links = []
        for link in soup.select('a[href*="/news/"]'):
            href = link.get('href', '')
            if href and '/news/' in href and href not in article_links:
                if not href.startswith('http'):
                    href = f"https://en.antaranews.com{href}"
                article_links.append(href)
                
        article_links = article_links[:max_articles]
        print(f"  Found {len(article_links)} article links")
        
        for url in article_links:
            article = self._parse_antara_article(url)
            if article:
                articles.append(article)
            time.sleep(1)
            
        return articles
    
    def _parse_antara_article(self, url: str) -> Optional[Dict]:
        """Parse a single ANTARA article"""
        soup = self._fetch_page(url)
        if not soup:
            return None
            
        try:
            title = soup.select_one('h1')
            title = title.get_text(strip=True) if title else "Untitled"
            
            content_div = soup.select_one('.post-content') or soup.select_one('article')
            if content_div:
                paragraphs = content_div.select('p')
                content = ' '.join([p.get_text(strip=True) for p in paragraphs])
            else:
                content = ""
                
            if not content or len(content) < 100:
                return None
                
            return {
                'title': self._clean_text(title),
                'content': self._clean_text(content),
                'media_source': 'antara',
                'url': url,
                'published_date': datetime.now()
            }
        except Exception as e:
            print(f"  ⚠️ Error parsing {url}: {e}")
            return None
    
    # =========================================
    # TEMPO.CO ENGLISH
    # =========================================
    def scrape_tempo_english(self, query: str = "presidential election", max_articles: int = 10) -> List[Dict]:
        """Scrape articles from Tempo.co English"""
        print(f"\n📰 Scraping Tempo.co English...")
        articles = []
        
        search_url = f"https://en.tempo.co/search?q={query.replace(' ', '+')}"
        soup = self._fetch_page(search_url)
        
        if not soup:
            return articles
            
        article_links = []
        for link in soup.select('a[href*="/read/"]'):
            href = link.get('href', '')
            if href and '/read/' in href and href not in article_links:
                if not href.startswith('http'):
                    href = f"https://en.tempo.co{href}"
                article_links.append(href)
                
        article_links = article_links[:max_articles]
        print(f"  Found {len(article_links)} article links")
        
        for url in article_links:
            article = self._parse_tempo_article(url)
            if article:
                articles.append(article)
            time.sleep(1)
            
        return articles
    
    def _parse_tempo_article(self, url: str) -> Optional[Dict]:
        """Parse a single Tempo article"""
        soup = self._fetch_page(url)
        if not soup:
            return None
            
        try:
            title = soup.select_one('h1')
            title = title.get_text(strip=True) if title else "Untitled"
            
            content_div = soup.select_one('.detail-konten') or soup.select_one('article')
            if content_div:
                paragraphs = content_div.select('p')
                content = ' '.join([p.get_text(strip=True) for p in paragraphs])
            else:
                content = ""
                
            if not content or len(content) < 100:
                return None
                
            return {
                'title': self._clean_text(title),
                'content': self._clean_text(content),
                'media_source': 'tempo',
                'url': url,
                'published_date': datetime.now()
            }
        except Exception as e:
            print(f"  ⚠️ Error parsing {url}: {e}")
            return None
    
    # =========================================
    # JAKARTA GLOBE
    # =========================================
    def scrape_jakarta_globe(self, query: str = "presidential election", max_articles: int = 10) -> List[Dict]:
        """Scrape articles from Jakarta Globe"""
        print(f"\n📰 Scraping Jakarta Globe...")
        articles = []
        
        search_url = f"https://jakartaglobe.id/?s={query.replace(' ', '+')}"
        soup = self._fetch_page(search_url)
        
        if not soup:
            return articles
            
        article_links = []
        for link in soup.select('a[href*="jakartaglobe.id/"]'):
            href = link.get('href', '')
            if href and '/?s=' not in href and href not in article_links:
                if '/news/' in href or '/politics/' in href or '/business/' in href:
                    article_links.append(href)
                    
        article_links = article_links[:max_articles]
        print(f"  Found {len(article_links)} article links")
        
        for url in article_links:
            article = self._parse_jakarta_globe_article(url)
            if article:
                articles.append(article)
            time.sleep(1)
            
        return articles
    
    def _parse_jakarta_globe_article(self, url: str) -> Optional[Dict]:
        """Parse a single Jakarta Globe article"""
        soup = self._fetch_page(url)
        if not soup:
            return None
            
        try:
            title = soup.select_one('h1')
            title = title.get_text(strip=True) if title else "Untitled"
            
            content_div = soup.select_one('.content-inner') or soup.select_one('article')
            if content_div:
                paragraphs = content_div.select('p')
                content = ' '.join([p.get_text(strip=True) for p in paragraphs])
            else:
                content = ""
                
            if not content or len(content) < 100:
                return None
                
            return {
                'title': self._clean_text(title),
                'content': self._clean_text(content),
                'media_source': 'jakarta_globe',
                'url': url,
                'published_date': datetime.now()
            }
        except Exception as e:
            print(f"  ⚠️ Error parsing {url}: {e}")
            return None


def scrape_all_sources(query: str = "presidential election aftermath", max_per_source: int = 10) -> List[Dict]:
    """
    Scrape articles from all configured sources.
    """
    scraper = NewsScraper()
    all_articles = []
    
    # Scrape each source
    all_articles.extend(scraper.scrape_jakarta_post(query, max_per_source))
    all_articles.extend(scraper.scrape_antara_news(query, max_per_source))
    all_articles.extend(scraper.scrape_tempo_english(query, max_per_source))
    all_articles.extend(scraper.scrape_jakarta_globe(query, max_per_source))
    
    print(f"\n✅ Total articles scraped: {len(all_articles)}")
    return all_articles


if __name__ == "__main__":
    # Test scraping
    articles = scrape_all_sources("Indonesia presidential election", max_per_source=5)
    
    # Save to JSON for inspection
    with open('data/scraped_articles.json', 'w', encoding='utf-8') as f:
        json.dump(articles, f, indent=2, default=str)
    
    print(f"\n📁 Saved {len(articles)} articles to data/scraped_articles.json")
