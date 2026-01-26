"""
Framing Analysis Model (TF-IDF & Frequency Analysis)
Custom analyzer built from scratch without LLM.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from collections import Counter
import re
from typing import List, Dict, Tuple

class FramingAnalyzer:
    """
    Analyzes media framing using classical NLP techniques.
    
    Approaches:
    1. TF-IDF to find unique keywords per media
    2. Actor Mention Frequency (Who is talked about?)
    3. Sentiment/Tone Dictionary (Optional expansion)
    """
    
    def __init__(self):
        # Indonesian Stopwords (Simple list, can be expanded)
        self.stop_words = [
            'yang', 'dan', 'di', 'dari', 'ke', 'ini', 'itu', 'untuk', 'adalah',
            'dengan', 'tidak', 'akan', 'juga', 'pada', 'ia', 'dia', 'mereka',
            'kami', 'kita', 'saya', 'anda', 'bisa', 'ada', 'sebagai', 'sudah'
        ]
        
    def analyze_media_framing(
        self, 
        articles_by_media: Dict[str, List[str]]
    ) -> Dict[str, Dict]:
        """
        Compare framing across different media sources.
        
        Args:
            articles_by_media: {'kompas': ['text1', ...], 'tempo': [...]}
            
        Returns:
            Dictionary containing analysis results per media
        """
        results = {}
        all_texts = []
        media_map = [] # Track which text belongs to which media
        
        # Prepare data
        for media, texts in articles_by_media.items():
            results[media] = {}
            if not texts: 
                continue
                
            combined_text = " ".join(texts)
            results[media]['actor_frequency'] = self._extract_actors(combined_text)
            
            # Store for TF-IDF
            all_texts.extend(texts)
            media_map.extend([media] * len(texts))
            
        if not all_texts:
            return results
            
        # 1. TF-IDF Analysis (Find distinctive words)
        tfidf_results = self._compute_tfidf_keywords(all_texts, media_map)
        
        for media in results:
            if media in tfidf_results:
                results[media]['top_keywords'] = tfidf_results[media]
                
        return results
    
    def _extract_actors(self, text: str, top_k: int = 10) -> List[Tuple[str, int]]:
        """
        Extract most frequent proper nouns (Actors).
        Uses simple heuristic: Capitalized words not at start of sentence.
        """
        # Simple heuristic for potential names (Capitalized words)
        # In production, training a CRF NER model from scratch would be better
        # but this suffices for "Rule-based/Classic NLP" approach.
        
        words = text.split()
        potential_actors = []
        
        for i, word in enumerate(words):
            clean_word = re.sub(r'[^\w]', '', word)
            if not clean_word: continue
            
            # If capitalized and not first word of sentence (rough check)
            if word[0].isupper() and i > 0 and not words[i-1].endswith('.'):
                if clean_word.lower() not in self.stop_words:
                    potential_actors.append(clean_word)
                    
        return Counter(potential_actors).most_common(top_k)

    def _compute_tfidf_keywords(
        self, 
        all_texts: List[str], 
        media_map: List[str],
        top_k: int = 10
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Compute top TF-IDF keywords per media.
        """
        # Fit TF-IDF on all texts
        vectorizer = TfidfVectorizer(
            stop_words=self.stop_words, 
            max_features=1000,
            token_pattern=r'\b[a-zA-Z]{3,}\b' # Min 3 chars
        )
        
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        feature_names = np.array(vectorizer.get_feature_names_out())
        
        media_keywords = {}
        unique_media = set(media_map)
        
        for media in unique_media:
            # Get indices for this media
            indices = [i for i, m in enumerate(media_map) if m == media]
            
            if not indices:
                continue
                
            # Average TF-IDF vector for this media
            avg_vector = np.mean(tfidf_matrix[indices], axis=0).A1
            
            # Get top indices
            top_indices = avg_vector.argsort()[-top_k:][::-1]
            
            keywords = [
                (feature_names[i], float(avg_vector[i])) 
                for i in top_indices
            ]
            media_keywords[media] = keywords
            
        return media_keywords
