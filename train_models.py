"""
Script to generate synthetic training data and train the custom evaluation models.
Since we don't have a large labeled dataset, we will generate synthetic examples
that mimic the properties of supported vs unsupported sentences.
"""
import numpy as np
import os
import sys
from typing import List, Tuple

sys.path.append('..') # Adjust path to project root
from models.hallucination.detector import HallucinationDetector
from models.confidence.scorer import ConfidenceScorer
from pipeline.embeddings import get_embedder

# Paths
MODELS_DIR = "data/models"
os.makedirs(MODELS_DIR, exist_ok=True)

def generate_hallucination_data(n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data for Hallucination Detector.
    
    Strategy:
    1. Positive samples (Supported): 
       - Take a 'source' text.
       - Create a 'sentence' that is a slightly modified version of the source.
       - Expected high similarity and overlap.
    
    2. Negative samples (Unsupported/Hallucination):
       - Take a 'source' text.
       - Create a 'sentence' that is completely different (random other text).
       - Expected low similarity and overlap.
    """
    print("Generating synthetic hallucination data...")
    embedder = get_embedder()
    
    # Mock data source
    base_texts = [
        "President Jokowi inaugurated the new toll road in North Sumatra yesterday afternoon.",
        "Indonesia's inflation reached its lowest figure in the last five years.",
        "The Indonesian National Team won convincingly against Vietnam in the World Cup qualifiers.",
        "Rice prices experienced a significant increase approaching the month of Ramadan.",
        " The KPK conducted a sting operation against regional officials.",
        "A 5.0 magnitude earthquake shook the West Java region.",
        "Major technology companies announced mass layoffs affecting local employees.",
        "The government will provide electric vehicle subsidies starting next year.",
        "Extreme weather is predicted to hit Jakarta for the next week."
    ]
    
    X_features_list = []
    y_labels_list = []
    
    detector = HallucinationDetector()
    
    for _ in range(n_samples // 2):
        # --- Positive Sample (Supported) ---
        source_idx = np.random.randint(0, len(base_texts))
        source_text = base_texts[source_idx]
        
        # Simulate slight paraphrase/extraction (Supported)
        # e.g., "Jokowi meresmikan jalan tol" from "Jokowi meresmikan jalan tol baru..."
        words = source_text.split()
        if len(words) > 3:
            cut_point = np.random.randint(3, len(words))
            sentence_text = " ".join(words[:cut_point])
        else:
            sentence_text = source_text
            
        # Get embeddings
        sent_emb = embedder.generate_single(sentence_text)
        source_emb = embedder.generate([source_text])[0]
        
        # Extract features
        feats = detector.extract_features(
            sent_emb, 
            np.array([source_emb]), 
            sentence_text, 
            [source_text]
        )
        X_features_list.append(feats)
        y_labels_list.append(1) # Label 1 = Supported
        
        # --- Negative Sample (Unsupported) ---
        source_idx_neg = np.random.randint(0, len(base_texts))
        source_text_neg = base_texts[source_idx_neg]
        
        # Pick a completely different text as the 'sentence'
        other_idx = (source_idx_neg + np.random.randint(1, len(base_texts)-1)) % len(base_texts)
        sentence_text_neg = base_texts[other_idx]
        
        # Get embeddings
        sent_emb_neg = embedder.generate_single(sentence_text_neg)
        source_emb_neg = embedder.generate([source_text_neg])[0]
        
        # Extract features
        feats_neg = detector.extract_features(
            sent_emb_neg, 
            np.array([source_emb_neg]), 
            sentence_text_neg, 
            [source_text_neg]
        )
        X_features_list.append(feats_neg)
        y_labels_list.append(0) # Label 0 = Unsupported
        
    return np.array(X_features_list), np.array(y_labels_list)

def generate_confidence_data(n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data for Confidence Scorer.
    
    Strategy:
    - Simulate retrieval metrics (similarities).
    - Confidence label logic: High max_sim and low variance -> High confidence.
    """
    print("Generating synthetic confidence data...")
    scorer = ConfidenceScorer()
    
    X_features_list = []
    y_labels_list = []
    
    for _ in range(n_samples):
        # Simulate retrieval results (5 chunks)
        # Randomly decide if this is a "good" retrieval or "bad" retrieval
        is_good = np.random.random() > 0.5
        
        if is_good:
            # Good retrieval: High similarities (e.g., 0.7 - 0.9)
            sims = np.random.uniform(0.7, 0.95, size=5).tolist()
            num_sources = np.random.randint(2, 4) # Multiple sources
            # Label = High confidence
            label = np.mean(sims) * 0.9 + (0.1 if num_sources > 1 else 0)
        else:
            # Bad retrieval: Low similarities (e.g., 0.2 - 0.6)
            sims = np.random.uniform(0.2, 0.6, size=5).tolist()
            num_sources = np.random.randint(1, 2) # Few sources
            label = np.mean(sims) * 0.8
            
        label = np.clip(label, 0.0, 1.0)
        
        feats = scorer.extract_features(sims, num_sources)
        X_features_list.append(feats)
        y_labels_list.append(label)
        
    return np.array(X_features_list), np.array(y_labels_list)

def main():
    print("Starting Model Training (From Scratch)...")
    
    # 1. Train Hallucination Detector
    print("\n--- Training Hallucination Detector ---")
    X_halluc, y_halluc = generate_hallucination_data()
    
    detector = HallucinationDetector()
    detector.train(X_halluc, y_halluc)
    detector.save(f"{MODELS_DIR}/hallucination_detector.pkl")
    
    # 2. Train Confidence Scorer
    print("\n--- Training Confidence Scorer ---")
    X_conf, y_conf = generate_confidence_data()
    
    scorer = ConfidenceScorer()
    scorer.train(X_conf, y_conf)
    scorer.save(f"{MODELS_DIR}/confidence_scorer.pkl")
    
    print("\nAll custom models trained and saved.")

if __name__ == "__main__":
    main()
