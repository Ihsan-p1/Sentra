"""
Text Preprocessing Pipeline
Cleaning, chunking, and metadata extraction for news articles.
"""
import re
from typing import List, Dict, Any
from dataclasses import dataclass
from langchain_text_splitters import RecursiveCharacterTextSplitter

import sys
sys.path.append('..')
from config.settings import settings


@dataclass
class ProcessedChunk:
    """Represents a processed text chunk"""
    chunk_text: str
    chunk_index: int
    metadata: Dict[str, Any]


class ArticlePreprocessor:
    """
    Preprocesses news articles for RAG pipeline.
    
    Steps:
    1. Clean text (remove noise, normalize whitespace)
    2. Extract metadata (title, source, date)
    3. Chunk into smaller pieces for embedding
    """
    
    def __init__(
        self, 
        chunk_size: int = None, 
        chunk_overlap: int = None
    ):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", ", ", " "],
            length_function=len
        )
        
        # Indonesian stopwords (basic)
        self.stopwords = {
            'yang', 'dan', 'di', 'dari', 'ke', 'pada', 'ini', 'itu',
            'dengan', 'untuk', 'adalah', 'dalam', 'tidak', 'akan',
            'juga', 'sudah', 'saya', 'kami', 'mereka', 'ada', 'bisa',
            'atau', 'oleh', 'setelah', 'karena', 'seperti', 'namun'
        }
    
    def clean_text(self, text: str) -> str:
        """
        Clean article text.
        
        Operations:
        - Normalize whitespace
        - Remove excessive punctuation
        - Remove URLs
        - Remove special characters (keep basic punctuation)
        """
        if not text:
            return ""
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Normalize whitespace (multiple spaces, newlines)
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:\-\'\"()\[\]]', '', text)
        
        # Remove multiple punctuation
        text = re.sub(r'([.,!?;:])\1+', r'\1', text)
        
        return text.strip()
    
    def chunk_article(self, article: Dict[str, Any]) -> List[ProcessedChunk]:
        """
        Split article into chunks with metadata.
        
        Args:
            article: Dict with keys: title, content, media_source, url, published_date
        
        Returns:
            List of ProcessedChunk objects
        """
        # Clean content
        cleaned_content = self.clean_text(article.get('content', ''))
        
        if not cleaned_content:
            return []
        
        # Split into chunks
        chunks = self.splitter.split_text(cleaned_content)
        
        # Create ProcessedChunk objects
        processed_chunks = []
        for i, chunk_text in enumerate(chunks):
            chunk = ProcessedChunk(
                chunk_text=chunk_text,
                chunk_index=i,
                metadata={
                    'title': article.get('title', ''),
                    'media_source': article.get('media_source', ''),
                    'url': article.get('url', ''),
                    'published_date': str(article.get('published_date', '')),
                    'total_chunks': len(chunks)
                }
            )
            processed_chunks.append(chunk)
        
        return processed_chunks
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Simple entity extraction (names, organizations).
        Uses pattern matching - for production, use proper NER.
        """
        entities = {
            'persons': [],
            'organizations': [],
            'locations': []
        }
        
        # Pattern for capitalized words (potential names)
        # Match 2-3 consecutive capitalized words
        name_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b'
        potential_names = re.findall(name_pattern, text)
        
        # Filter common words that are capitalized at sentence start
        entities['persons'] = list(set([
            name for name in potential_names 
            if name.lower() not in self.stopwords
            and len(name) > 3
        ]))[:10]  # Limit to top 10
        
        return entities
    
    def preprocess_batch(
        self, 
        articles: List[Dict[str, Any]]
    ) -> List[ProcessedChunk]:
        """Process multiple articles"""
        all_chunks = []
        
        for article in articles:
            chunks = self.chunk_article(article)
            all_chunks.extend(chunks)
        
        return all_chunks


# Convenience function
def preprocess_article(article: Dict[str, Any]) -> List[ProcessedChunk]:
    """Preprocess a single article"""
    preprocessor = ArticlePreprocessor()
    return preprocessor.chunk_article(article)
