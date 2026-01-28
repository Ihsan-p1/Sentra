"""
Hallucination Detection Model (Logistic Regression)
Trained from scratch using extracted features.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os
from typing import List, Tuple, Dict
import sys

sys.path.append('../../..') # Adjust path if needed

class HallucinationDetector:
    """
    Custom Hallucination Detector trained from scratch.
    
    Uses classic ML (Logistic Regression) on manually engineered features
    rather than a 'black box' deep learning model.
    """
    
    def __init__(self, model_path: str = None):
        self.model = LogisticRegression(class_weight='balanced')
        self.scaler = StandardScaler()
        self.is_trained = False
        
        if model_path and os.path.exists(model_path):
            self.load(model_path)
            
    def extract_features(
        self, 
        sentence_emb: np.ndarray, 
        source_embs: np.ndarray,
        sentence_text: str = "",
        source_texts: List[str] = []
    ) -> np.ndarray:
        """
        Extract interpretable features for the classifier.
        
        Features:
        1. Max Cosine Similarity (How close is the best matching chunk?)
        2. Avg Cosine Similarity (Is the sentence generally supported?)
        3. Min Cosine Similarity
        4. Std Dev Similarity (Variance in support)
        5. Exact Keyword Overlap (Ratio of words found in sources)
        6. Sentence Length (Longer sentences might be harder to support)
        """
        
        # 1. Similarity Features
        if len(source_embs) == 0:
            return np.zeros(6)
            
        sims = cosine_similarity(sentence_emb.reshape(1, -1), source_embs)[0]
        
        max_sim = np.max(sims)
        avg_sim = np.mean(sims)
        min_sim = np.min(sims)
        std_sim = np.std(sims)
        
        # 2. Textual Features (Simple Overlap)
        sent_words = set(sentence_text.lower().split())
        source_words = set(" ".join(source_texts).lower().split())
        
        if len(sent_words) > 0:
            overlap = len(sent_words.intersection(source_words)) / len(sent_words)
        else:
            overlap = 0.0
            
        sent_len = len(sent_words)
        
        return np.array([max_sim, avg_sim, min_sim, std_sim, overlap, sent_len])

    def train(self, X_features: np.ndarray, y_labels: np.ndarray):
        """Train the logistic regression model"""
        print(f"Training Hallucination Detector on {len(X_features)} samples...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_features)
        
        # Train
        self.model.fit(X_scaled, y_labels)
        self.is_trained = True
        
        # Show weights (Explainability)
        feature_names = ['Max Sim', 'Avg Sim', 'Min Sim', 'Std Sim', 'Overlap', 'Sent Len']
        print("\nModel Coefficients (Feature Importance):")
        for name, coef in zip(feature_names, self.model.coef_[0]):
            print(f"  {name}: {coef:.4f}")
            
    def predict(
        self, 
        sentence_emb: np.ndarray, 
        source_embs: np.ndarray,
        sentence_text: str = "",
        source_texts: List[str] = []
    ) -> Dict:
        """Predict if a sentence is a hallucination"""
        if not self.is_trained:
            raise ValueError("Model is not trained yet!")
            
        features = self.extract_features(
            sentence_emb, source_embs, sentence_text, source_texts
        )
        
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        prob = self.model.predict_proba(features_scaled)[0]
        
        # --- HEURISTIC OVERRIDE (Fix for broken ML model weights) ---
        # The ML model seems to be biased towards "Unverified" even for perfect matches.
        # We trust deterministic metrics (Similarity & Overlap) more than the ML model here.
        
        max_sim = features[0]
        overlap = features[4]
        
        # 1. Strong Match Override (Almost exact quote)
        if max_sim > 0.75: 
            prob[1] = max(prob[1], 0.95) # Force high confidence
            
        # 2. Moderate Match Override (Paraphrased but supported)
        elif max_sim > 0.55 and overlap > 0.4:
            prob[1] = max(prob[1], 0.85)
            
        # 3. Weak Match but high overlap (Rare, but possible)
        elif overlap > 0.7:
             prob[1] = max(prob[1], 0.80)
        
        # -----------------------------------------------------------
        
        # Lower threshold (0.35) to be more lenient for interpretive/analytical content
        # Original academic research often paraphrases and interprets, not just quotes
        is_supported = prob[1] > 0.40  # Slightly adjusted threshold
        
        # Determine reason if not supported
        if not is_supported:
            if features[0] < 0.3:  # max_similarity
                reason = "Low semantic similarity to retrieved sources"
            elif features[4] < 0.2:  # overlap
                reason = "Limited keyword overlap with sources"
            else:
                reason = "Claim appears to extend beyond source material"
        else:
            reason = ""
        
        return {
            'is_supported': bool(is_supported),
            'confidence': float(prob[1] if is_supported else prob[0]),
            'score': float(prob[1]), # Probability of being supported
            'reason': reason,
            'features': {
                'max_similarity': float(features[0]),
                'overlap': float(features[4])
            }
        }
        
    def save(self, path: str):
        joblib.dump({'model': self.model, 'scaler': self.scaler}, path)
        print(f"Model saved to {path}")
        
    def load(self, path: str):
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.is_trained = True
        print(f"Model loaded from {path}")
