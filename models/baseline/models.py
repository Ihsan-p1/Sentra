"""
Baseline Models for Comparison (Model B)
These are simple rule-based approaches to compare against our trained models.
"""
import re
from typing import List, Dict, Any

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
            "unsupported_ratio": f"{unsupported_count}/{total}",  # same orientation as Model A
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

    def keyword_weights(self, texts_by_media: Dict[str, List[str]], top_k: int = 5) -> Dict[str, List]:
        """
        Same extraction as analyze(), but returns (word, count) pairs and an
        adjustable depth. Used by the framing evaluation, which needs weights to
        score articles and a comparable top_k across extractors.
        """
        profiles = {}

        for media, texts in texts_by_media.items():
            combined = " ".join(texts).lower()
            words = re.findall(r'\b[a-z]+\b', combined)

            counts = {}
            for word in words:
                if word not in self.stopwords and len(word) > 3:
                    counts[word] = counts.get(word, 0) + 1

            ranked = sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:top_k]
            profiles[media] = [(word, float(count)) for word, count in ranked]

        return profiles


class BaselineConfidenceScorer:
    """
    Baseline: deterministic rule-based confidence.

    Much simpler than the trained Random Forest model. Deliberately contains no
    randomness: the A/B comparison against Model A must be reproducible, so the
    same query on the same corpus has to yield the same baseline score every run.
    """

    def __init__(self):
        self.name = "Baseline (Heuristic)"

    def score(self, query: str, retrieved_chunks: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Return a deterministic heuristic confidence score"""
        # Count total chunks retrieved
        total_chunks = sum(len(chunks) for chunks in retrieved_chunks.values())

        # Number of distinct sources that actually returned something
        num_sources = sum(1 for chunks in retrieved_chunks.values() if chunks)

        # Simple weighted rule: more chunks and more corroborating sources
        # both raise confidence, each capped so neither can dominate.
        chunk_component = min(total_chunks * 0.08, 0.55)
        source_component = min(num_sources * 0.1, 0.25)
        score = round(min(0.9, max(0.2, chunk_component + source_component)), 3)

        return {
            "confidence": score,
            "confidence_percent": f"{int(score * 100)}%",
            "method": "baseline_heuristic",
            "reasoning": f"Based on {total_chunks} chunks across {num_sources} sources"
        }


# Convenience function to get all baseline models
def get_baseline_models():
    return {
        "hallucination": BaselineHallucinationDetector(),
        "framing": BaselineFramingAnalyzer(),
        "confidence": BaselineConfidenceScorer()
    }
