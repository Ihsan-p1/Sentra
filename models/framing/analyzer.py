"""
Framing Analysis Model (TF-IDF & Frequency Analysis)
Custom analyzer built from scratch without LLM.
Updated to include BertFramingAnalyzer (Neural Fine-tuned).
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from collections import Counter
import re
from typing import List, Dict, Tuple
import os
import json
import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

class FramingAnalyzer:
    """
    Analyzes media framing using classical NLP techniques (TF-IDF).
    Model A (Legacy) or Model B (Baseline) in the comparison.
    """
    
    def __init__(self):
        # Indonesian Stopwords (Simple list, can be expanded)
        # English Stopwords (Simple list)
        self.stop_words = [
            'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but', 
            'in', 'to', 'of', 'for', 'with', 'from', 'by', 'that', 'this', 'it',
            'as', 'are', 'was', 'were', 'be', 'been', 'has', 'have', 'had',
            'not', 'news', 'said', 'report', 'jakarta', 'indonesia' # Domain specific stops
        ]
        
    def analyze_media_framing(
        self, 
        articles_by_media: Dict[str, List[str]]
    ) -> Dict[str, Dict]:
        """
        Compare framing across different media sources.
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
        """
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
        vectorizer = TfidfVectorizer(
            stop_words=self.stop_words, 
            max_features=1000,
            token_pattern=r'\b[a-zA-Z]{3,}\b' # Min 3 chars
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(all_texts)
            feature_names = np.array(vectorizer.get_feature_names_out())
            
            media_keywords = {}
            unique_media = set(media_map)
            
            for media in unique_media:
                indices = [i for i, m in enumerate(media_map) if m == media]
                if not indices: continue
                    
                avg_vector = np.mean(tfidf_matrix[indices], axis=0).A1
                top_indices = avg_vector.argsort()[-top_k:][::-1]
                
                keywords = [
                    (feature_names[i], float(avg_vector[i])) 
                    for i in top_indices
                ]
                media_keywords[media] = keywords
                
            return media_keywords
        except ValueError:
            return {}


class BertFramingAnalyzer:
    """
    Fine-tuned Neural Model for Framing Classification.
    Predicts which media source a text sounds like.
    """
    def __init__(self, model_path="data/models/framing_bert"):
        self.is_loaded = False
        try:
            if os.path.exists(model_path):
                self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
                self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.model.to(self.device)
                self.model.eval()
                
                # Load label map
                with open(os.path.join(model_path, 'label_map.json'), 'r') as f:
                    self.id_to_label = json.load(f)
                    # Convert keys to int because JSON makes them strings
                    self.id_to_label = {int(k): v for k, v in self.id_to_label.items()}
                
                self.is_loaded = True
                print("[INFO] BertFramingAnalyzer loaded.")
            else:
                print(f"[WARN] BERT model not found at {model_path}")
        except Exception as e:
            print(f"[ERROR] Loading BERT model: {e}")

    def predict_source_style(self, text: str) -> Dict[str, float]:
        """
        Predict the probability that the text matches the style of each media source.
        """
        if not self.is_loaded:
            return {}
            
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=128
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)
            
        probs_dict = {}
        for idx, prob in enumerate(probs[0]):
            label = self.id_to_label.get(idx, f"Unknown_{idx}")
            probs_dict[label] = float(prob)
            
        return probs_dict
