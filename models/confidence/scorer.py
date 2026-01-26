"""
Confidence Scoring Model (Random Forest)
Estimates confidence of the answer based on retrieval metrics.
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os
from typing import Dict, List
import sys

sys.path.append('../../..')

class ConfidenceScorer:
    """
    ML-based Confidence Scorer.
    
    Predicts a confidence score (0-1) representing how likely the answer 
    constructible from retrieved chunks is sufficient/correct.
    """
    
    def __init__(self, model_path: str = None):
        self.model = RandomForestRegressor(n_estimators=100, max_depth=5)
        self.scaler = StandardScaler()
        self.is_trained = False
        
        if model_path and os.path.exists(model_path):
            self.load(model_path)
            
    def extract_features(
        self, 
        similarities: List[float], 
        num_sources: int
    ) -> np.ndarray:
        """
        Extract features from retrieval results.
        
        Features:
        1. Max Similarity (Best chunk match)
        2. Mean Similarity (Average match quality)
        3. Min Similarity
        4. Number of Sources (Diversity)
        5. Similarity Variance (Consistency)
        """
        if not similarities:
            return np.zeros(5)
            
        sims = np.array(similarities)
        
        max_sim = np.max(sims)
        mean_sim = np.mean(sims)
        min_sim = np.min(sims)
        std_sim = np.std(sims)
        
        return np.array([max_sim, mean_sim, min_sim, num_sources, std_sim])
        
    def train(self, X_features: np.ndarray, y_confidence: np.ndarray):
        """Train Random Forest Regressor"""
        print(f"Training Confidence Scorer on {len(X_features)} samples...")
        
        X_scaled = self.scaler.fit_transform(X_features)
        self.model.fit(X_scaled, y_confidence)
        self.is_trained = True
        
        print("Feature Importances:")
        feature_names = ['Max Sim', 'Mean Sim', 'Min Sim', 'Num Sources', 'Sim Std']
        for name, imp in zip(feature_names, self.model.feature_importances_):
            print(f"  {name}: {imp:.4f}")
            
    def predict(self, similarities: List[float], num_sources: int) -> float:
        """Predict confidence score (0.0 to 1.0)"""
        if not self.is_trained:
            # Fallback heuristic if not trained
            if not similarities: return 0.0
            return float(np.mean(similarities))
            
        features = self.extract_features(similarities, num_sources)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        score = self.model.predict(features_scaled)[0]
        return float(np.clip(score, 0.0, 1.0))

    def save(self, path: str):
        joblib.dump({'model': self.model, 'scaler': self.scaler}, path)
        print(f"Confidence model saved to {path}")
        
    def load(self, path: str):
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.is_trained = True
        print(f"Confidence model loaded from {path}")
