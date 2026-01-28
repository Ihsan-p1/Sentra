"""
Baseline Models for Comparison (Model B)
These are simple rule-based approaches to compare against our trained models.
"""
import re
from typing import List, Dict, Any
import random

class BaselineHallucinationDetector:
    """
    Baseline: Simple keyword overlap check.
    Much simpler than our trained Logistic Regression model.
    """
    
    def __init__(self):
        self.name = "Baseline (Keyword Overlap)"
    
    def check_sentence(self, sentence: str, source_texts: List[str]) -> Dict[str, Any]:
        """Check if sentence keywords appear in sources"""
        # Extract significant words (>4 chars)
        sentence_words = set(word.lower() for word in re.findall(r'\b\w+\b', sentence) if len(word) > 4)
        
        if not sentence_words:
            return {"is_supported": True, "confidence": 0.5, "method": "baseline_keyword"}
        
        # Check overlap with each source
        max_overlap = 0
        for source in source_texts:
            source_words = set(word.lower() for word in re.findall(r'\b\w+\b', source) if len(word) > 4)
            if source_words:
                overlap = len(sentence_words & source_words) / len(sentence_words)
                max_overlap = max(max_overlap, overlap)
        
        # Simple threshold: 30% overlap = supported
        is_supported = max_overlap >= 0.3
        
        return {
            "is_supported": is_supported,
            "confidence": max_overlap,
            "method": "baseline_keyword"
        }
    
    def check_response(self, response: str, source_texts: List[str]) -> Dict[str, Any]:
        """Check entire response"""
        # Split by punctuation, but keep structure
        sentences = [s.strip() for s in re.split(r'[.!?]', response) if len(s.strip()) > 10]
        
        results = []
        for sentence in sentences:
            # Clean sentence (remove citations and bullets)
            clean_sentence = re.sub(r'\[.*?\]', '', sentence).strip()
            clean_sentence = re.sub(r'^[\-\*•]\s+', '', clean_sentence).strip()
            
            # Skip headers/markdown artifacts
            if clean_sentence.startswith("##") or len(clean_sentence) < 10:
                continue
                
            # Skip refusal sentences
            if "sufficient information" in clean_sentence.lower():
                continue

            result = self.check_sentence(clean_sentence, source_texts)
            results.append({
                # Store original sentence for display context (or clean if preferred)
                "sentence": clean_sentence[:50] + "..." if len(clean_sentence) > 50 else clean_sentence,
                **result
            })
        
        supported_count = sum(1 for r in results if r["is_supported"])
        total = len(results) if results else 1
        unsupported_count = total - supported_count if results else 0
        
        return {
            "overall_score": supported_count / total,
            "supported_ratio": f"{unsupported_count}/{total}", # Return UNsupported/Total to match Model A
            "details": results,
            "method": "baseline_keyword"
        }


class BaselineFramingAnalyzer:
    """
    Baseline: Simple word frequency count.
    Much simpler than our TF-IDF based analyzer.
    """
    
    def __init__(self):
        self.name = "Baseline (Word Count)"
        self.stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
                         'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                         'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                         'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                         'as', 'into', 'through', 'during', 'before', 'after', 'above',
                         'below', 'between', 'under', 'again', 'further', 'then', 'once',
                         'and', 'but', 'or', 'nor', 'so', 'yet', 'both', 'either',
                         'neither', 'not', 'only', 'own', 'same', 'than', 'too', 'very',
                         'just', 'also', 'now', 'here', 'there', 'when', 'where', 'why',
                         'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some',
                         'such', 'no', 'any', 'this', 'that', 'these', 'those', 'it', 'its'}
    
    def analyze(self, texts_by_media: Dict[str, List[str]]) -> Dict[str, Any]:
        """Simple word frequency per media"""
        results = {}
        
        for media, texts in texts_by_media.items():
            combined = " ".join(texts).lower()
            words = re.findall(r'\b[a-z]+\b', combined)
            
            # Count words (excluding stopwords)
            word_counts = {}
            for word in words:
                if word not in self.stopwords and len(word) > 3:
                    word_counts[word] = word_counts.get(word, 0) + 1
            
            # Get top 5
            sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            
            results[media] = {
                "top_keywords": [w[0] for w in sorted_words],
                "method": "baseline_wordcount"
            }
        
        return {
            "framing_by_media": results,
            "method": "baseline_wordcount"
        }


class BaselineConfidenceScorer:
    """
    Baseline: Random or fixed confidence score.
    Much simpler than our trained Random Forest model.
    """
    
    def __init__(self):
        self.name = "Baseline (Random/Fixed)"
    
    def score(self, query: str, retrieved_chunks: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Return a semi-random confidence score"""
        # Count total chunks retrieved
        total_chunks = sum(len(chunks) for chunks in retrieved_chunks.values())
        
        # Simple heuristic: more chunks = higher confidence (capped)
        base_score = min(total_chunks * 0.1, 0.7)
        
        # Add some randomness
        noise = random.uniform(-0.1, 0.1)
        score = max(0.2, min(0.9, base_score + noise))
        
        return {
            "confidence": score,
            "confidence_percent": f"{int(score * 100)}%",
            "method": "baseline_random",
            "reasoning": f"Based on {total_chunks} retrieved chunks"
        }


# Convenience function to get all baseline models
def get_baseline_models():
    return {
        "hallucination": BaselineHallucinationDetector(),
        "framing": BaselineFramingAnalyzer(),
        "confidence": BaselineConfidenceScorer()
    }
