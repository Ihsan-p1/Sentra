"""
Integration of RAG + Custom Models + LLM Generator
The 'Brain' of the chatbot with A/B Model Comparison.
"""
from typing import Dict, List, Any
import asyncio
import re

import sys
sys.path.append('..')
from config.settings import settings
from rag.retrieval import RAGRetriever
from models.hallucination.detector import HallucinationDetector
from models.framing.analyzer import FramingAnalyzer
from models.confidence.scorer import ConfidenceScorer
from models.baseline.models import get_baseline_models
from pipeline.embeddings import get_embedder

# For LLM generation (using OpenAI or Ollama)
from chatbot.llm_client import LLMClient 

class ChatbotEngine:
    """
    Orchestrates the RAG flow with A/B Model Comparison:
    1. Retrieve documents (RAGRetriever)
    2. Analyze framing (Model A: TF-IDF vs Model B: Word Count)
    3. Generate response (LLM)
    4. Check hallucinations (Model A: Logistic Regression vs Model B: Keyword Overlap)
    5. Score confidence (Model A: Random Forest vs Model B: Heuristic)
    """
    
    def __init__(self):
        self.retriever = RAGRetriever()
        self.embedder = get_embedder()
        self.llm = LLMClient()
        
        # Load custom trained models (Model A)
        try:
            self.hallucination_detector = HallucinationDetector("data/models/hallucination_detector.pkl")
            self.confidence_scorer = ConfidenceScorer("data/models/confidence_scorer.pkl")
            print("✅ Loaded custom evaluation models (Model A)")
        except:
            print("⚠️ Custom models not found, using untrained instance")
            self.hallucination_detector = HallucinationDetector()
            self.confidence_scorer = ConfidenceScorer()
            
        self.framing_analyzer = FramingAnalyzer()
        
        # Load baseline models (Model B)
        self.baseline_models = get_baseline_models()
        print("✅ Loaded baseline models (Model B)")
        
    async def process_query(self, query: str) -> Dict[str, Any]:
        """
        Main pipeline execution with A/B comparison.
        """
        # 1. RETRIEVAL
        retrieved_chunks = await self.retriever.retrieve(query)
        
        # Flatten for certain analyses
        all_chunks_flat = self.retriever.flatten_results(retrieved_chunks)
        source_texts = [c['text'] for c in all_chunks_flat]
        
        if not source_texts:
            return {
                "answer": "Sorry, I couldn't find relevant information in the news database.",
                "analysis": {},
                "confidence": 0.0,
                "model_comparison": {}
            }
            
        # 2. FRAMING ANALYSIS
        texts_by_media = {
            media: [c['text'] for c in chunks]
            for media, chunks in retrieved_chunks.items()
        }
        
        # Model A: TF-IDF based
        framing_model_a = self.framing_analyzer.analyze_media_framing(texts_by_media)
        
        # Model B: Simple word count
        framing_model_b = self.baseline_models["framing"].analyze(texts_by_media)
        
        # 3. TEXT GENERATION (LLM) - Use Model A framing
        raw_response = await self.llm.generate_comparative_answer(
            query, 
            retrieved_chunks, 
            framing_model_a
        )
        
        # 4. HALLUCINATION DETECTION
        # Model A: Trained Logistic Regression
        annotated_response_a, unsupported_a, stats_a = self._check_hallucinations_model_a(
            raw_response, source_texts
        )
        
        # Model B: Simple keyword overlap
        hallucination_result_b = self.baseline_models["hallucination"].check_response(
            raw_response, source_texts
        )
        
        # 5. CONFIDENCE SCORING
        all_sims = [c['similarity'] for c in all_chunks_flat]
        
        # Model A: Trained Random Forest
        confidence_a = self.confidence_scorer.predict(all_sims, len(retrieved_chunks))
        
        # Model B: Heuristic-based
        confidence_b = self.baseline_models["confidence"].score(query, retrieved_chunks)
        
        # Build comparison result
        model_comparison = {
            "hallucination": {
                "model_a": {
                    "name": "Trained Logistic Regression",
                    "supported_ratio": stats_a["supported_ratio"],
                    "accuracy_estimate": f"{stats_a['support_rate']:.0%}",
                    "method": "embedding_similarity + ML"
                },
                "model_b": {
                    "name": "Baseline Keyword Overlap",
                    "supported_ratio": hallucination_result_b["supported_ratio"],
                    "accuracy_estimate": f"{hallucination_result_b['overall_score']:.0%}",
                    "method": "simple_keyword_matching"
                }
            },
            "framing": {
                "model_a": {
                    "name": "TF-IDF Analyzer",
                    "keywords": {m: data.get("top_keywords", [])[:3] for m, data in framing_model_a.get("framing_by_media", {}).items()},
                    "method": "tf-idf + entity_extraction"
                },
                "model_b": {
                    "name": "Word Count Baseline",
                    "keywords": {m: data.get("top_keywords", [])[:3] for m, data in framing_model_b.get("framing_by_media", {}).items()},
                    "method": "simple_frequency"
                }
            },
            "confidence": {
                "model_a": {
                    "name": "Trained Random Forest",
                    "score": confidence_a,
                    "score_percent": f"{int(confidence_a * 100)}%",
                    "method": "ML_regression"
                },
                "model_b": {
                    "name": "Heuristic Baseline",
                    "score": confidence_b["confidence"],
                    "score_percent": confidence_b["confidence_percent"],
                    "method": "chunk_count_heuristic"
                }
            }
        }
        
        # Extract unsupported claims from Model B
        unsupported_b = [
            {"sentence": d["sentence"], "confidence": d["confidence"]}
            for d in hallucination_result_b.get("details", [])
            if not d.get("is_supported", True)
        ]
        
        return {
            "answer": annotated_response_a,
            "raw_answer": raw_response,
            "framing_analysis": framing_model_a,
            "unsupported_claims": unsupported_a,
            "unsupported_claims_b": unsupported_b,
            "confidence_score": confidence_a,
            "sources": retrieved_chunks,
            "model_comparison": model_comparison
        }
        
    def _check_hallucinations_model_a(
        self, 
        response_text: str, 
        source_texts: List[str]
    ) -> tuple:
        """
        Check each sentence using custom HallucinationDetector (Model A).
        Returns: (annotated_text, unsupported_list, stats_dict)
        """
        sentences = re.split(r'(?<=[.!?])\s+', response_text)
        
        annotated_sentences = []
        unsupported = []
        supported_count = 0
        total_checked = 0
        
        # Pre-compute source embeddings once
        source_embs = self.embedder.generate(source_texts, show_progress=False)
        
        for sentence in sentences:
            if len(sentence.split()) < 4:
                annotated_sentences.append(sentence)
                continue
            
            total_checked += 1
            sent_emb = self.embedder.generate_single(sentence)
            
            result = self.hallucination_detector.predict(
                sent_emb, 
                source_embs, 
                sentence, 
                source_texts
            )
            
            if result['is_supported']:
                annotated_sentences.append(sentence)
                supported_count += 1
            else:
                annotated_sentences.append(f"{sentence} [⚠️ Unverified]")
                unsupported.append({
                    'sentence': sentence,
                    'confidence': result['confidence']
                })
        
        stats = {
            "supported_count": supported_count,
            "total_checked": total_checked,
            "supported_ratio": f"{supported_count}/{total_checked}",
            "support_rate": supported_count / total_checked if total_checked > 0 else 1.0
        }
                
        return " ".join(annotated_sentences), unsupported, stats
