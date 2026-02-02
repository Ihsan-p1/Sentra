"""
Integration of RAG + Custom Models + LLM Generator
The 'Brain' of the chatbot with A/B Model Comparison.

Improved: Aggregated hallucination warnings instead of per-sentence spam.
"""
from typing import Dict, List, Any
import asyncio
import re

import sys
sys.path.append('..')
from config.settings import settings
from rag.retrieval import RAGRetriever
from models.hallucination.detector import HallucinationDetector
from models.framing.analyzer import FramingAnalyzer, BertFramingAnalyzer
from models.confidence.scorer import ConfidenceScorer
from models.baseline.models import get_baseline_models
from pipeline.embeddings import get_embedder

# For LLM generation (using OpenAI or Ollama)
from chatbot.llm_client import LLMClient 
from chatbot.conversation_memory import conversation_memory 

class ChatbotEngine:
    """
    Orchestrates the RAG flow with A/B Model Comparison:
    1. Retrieve documents (RAGRetriever)
    2. Analyze framing (Model A: DistilBERT [Neural] vs Model B: TF-IDF [Statistical])
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
            print("[INFO] Loaded custom evaluation models (Model A)")
        except:
            print("[WARN] Custom models not found, using untrained instance")
            self.hallucination_detector = HallucinationDetector()
            self.confidence_scorer = ConfidenceScorer()
            
        self.framing_analyzer = FramingAnalyzer() # Statistical (TF-IDF)
        self.bert_framing = BertFramingAnalyzer() # Neural (Fine-tuned DistilBERT)
        
        # Load baseline models (Model B)
        self.baseline_models = get_baseline_models()
        print("[INFO] Loaded baseline models (Model B)")
        
        # Initialize conversation memory
        self.memory = conversation_memory
        
    async def process_query(
        self, 
        query: str, 
        session_id: str = "default",
        mode: str = "default"
    ) -> Dict[str, Any]:
        """
        Main pipeline execution with A/B comparison.
        Now with AGGREGATED hallucination reporting for better UX.
        """
        # 0. QUERY SANITY CHECK (Reduce Hallucination Mode)
        if mode == "reduce_hallucination":
            # Check length ( < 3 words is prone to ambiguity)
            if len(query.strip().split()) < 3:
                 return {
                    "answer": "This question cannot be reliably answered using the current news corpus. Please ask a more specific question (at least 3 words).",
                    "analysis": {"error": "query_too_short"},
                    "confidence": 0.0,
                    "model_comparison": {}, 
                    "mode": mode,
                    "session_id": session_id
                }

        # 0.5 CONTEXTUAL QUERY ENHANCEMENT
        # If query is too short (continuation phrases), extract topic from conversation
        retrieval_query = query
        extracted_topic = None  # Store the specific topic user is asking about
        affirmative_phrases = ["yes", "yeah", "yep", "sure", "ok", "okay", "please", "go ahead", "ya", "iya"]
        continuation_phrases = ["tell me more", "more", "continue", "go on", "what else", "explain", "elaborate"]
        
        query_lower = query.strip().lower()
        is_short_query = len(query.strip().split()) <= 3
        is_affirmative = any(phrase in query_lower for phrase in affirmative_phrases)
        is_continuation = any(phrase in query_lower for phrase in continuation_phrases)
        
        if is_short_query and (is_affirmative or is_continuation):
            history = self.memory.get_history(session_id)
            if history:
                # PRIORITY 1: Extract topic from assistant's "Would you like to know more about X?" question
                import re
                for msg in reversed(history):
                    if msg["role"] == "assistant":
                        assistant_text = msg["content"]
                        
                        # Pattern to match "Would you like to know more about X?"
                        patterns = [
                            r"would you like to (?:know|learn|hear) more about (.+?)\?",
                            r"want (?:me )?to (?:tell|explain) (?:you )?(?:more )?about (.+?)\?",
                            r"interested in (?:learning|knowing) (?:more )?about (.+?)\?",
                            r"shall i (?:tell|explain) (?:you )?(?:more )?about (.+?)\?",
                        ]
                        
                        for pattern in patterns:
                            match = re.search(pattern, assistant_text.lower())
                            if match:
                                extracted_topic = match.group(1).strip()
                                # Clean up
                                extracted_topic = extracted_topic.rstrip('.,!?')
                                if len(extracted_topic) > 10:  # Valid topic
                                    retrieval_query = extracted_topic
                                    print(f"[INFO] ✓ Extracted follow-up topic: '{extracted_topic}'")
                                    break
                                else:
                                    extracted_topic = None
                        
                        if extracted_topic:
                            break
                
                # PRIORITY 2: Fallback to last substantive user query
                if not extracted_topic:
                    for msg in reversed(history):
                        if msg["role"] == "user" and len(msg["content"].split()) > 3:
                            retrieval_query = msg["content"]
                            print(f"[INFO] Fallback to previous query: '{retrieval_query[:50]}...'")
                            break
        
        # 1. RETRIEVAL
        # Context Saturation Control
        top_k = 4 if mode == "reduce_hallucination" else None
        
        retrieved_chunks = await self.retriever.retrieve(retrieval_query, top_k=top_k)
        
        # Flatten for certain analyses
        all_chunks_flat = self.retriever.flatten_results(retrieved_chunks)
        source_texts = [c['text'] for c in all_chunks_flat]
        
        if not source_texts:
            return {
                "answer": "Sorry, I couldn't find relevant information in the news database.",
                "analysis": {},
                "confidence": 0.0,
                "model_comparison": {},
                "session_id": session_id
            }
        
        # 1.5 RETRIEVAL CONFIDENCE GATE - Refuse if source quality too low
        similarities = [c['similarity'] for c in all_chunks_flat]
        max_similarity = max(similarities)
        avg_similarity = sum(similarities) / len(similarities)
        
        # Dynamic Threshold
        threshold = 0.40 if mode == "reduce_hallucination" else 0.35
        
        if max_similarity < threshold:
            refusal_msg = "Insufficient source coverage." if mode == "reduce_hallucination" else "I cannot answer this based on the available news sources. The retrieved articles do not contain sufficiently relevant information. Please try asking about specific Indonesian political events or candidates covered in the database."
            
            return {
                "answer": refusal_msg,
                "analysis": {"retrieval_quality": "insufficient"},
                "confidence": 0.0,
                "model_comparison": {},
                "source_quality": {
                    "max_similarity": round(max_similarity, 3),
                    "avg_similarity": round(avg_similarity, 3),
                    "total_sources": len(all_chunks_flat),
                    "refusal_reason": f"max_sim={max_similarity:.2f} < {threshold}",
                    "mode": mode
                },
                "session_id": session_id
            }

        # Reduce Chunks Per Media (Context Saturation Control)
        if mode == "reduce_hallucination":
            for media in retrieved_chunks:
                retrieved_chunks[media] = retrieved_chunks[media][:2] # Max 2 per media
            
        # 2. FRAMING ANALYSIS
        texts_by_media = {
            media: [c['text'] for c in chunks]
            for media, chunks in retrieved_chunks.items()
        }
        
        # Statistical Analysis (TF-IDF)
        framing_statistical = self.framing_analyzer.analyze_media_framing(texts_by_media)
        
        # Neural Analysis (DistilBERT)
        # We process the combined text per media to see if it matches the expected style
        framing_neural = {}
        for media, texts in texts_by_media.items():
            if texts:
                combined_text = " ".join(texts)[:512] # Limit len for BERT
                prediction = self.bert_framing.predict_source_style(combined_text)
                # Store the probability of the ACTUAL media source
                framing_neural[media] = prediction.get(media, 0.0)
        
        # 2.5 GET CONVERSATION HISTORY for context
        conversation_history = self.memory.format_for_llm(session_id, num_turns=5)
        
        # 3. TEXT GENERATION (LLM) - Use Conversational Prompt
        raw_response = await self.llm.generate_conversational_response(
            user_query=query, 
            retrieved_chunks=retrieved_chunks, 
            conversation_history=conversation_history,
            mode=mode,
            focus_topic=extracted_topic
        )
        
        # 4. HALLUCINATION DETECTION (Internal scoring, NOT per-sentence annotation)
        # Model A: Trained Logistic Regression
        verification_result_a, unsupported_a, stats_a = self._analyze_hallucinations_aggregated(
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
        
        # 6. BUILD FINAL ANSWER (with aggregated disclaimer, not per-sentence spam)
        final_answer = self._build_final_answer(raw_response, verification_result_a, unsupported_a)
        
        # Build comparison result
        model_comparison = {
            "hallucination": {
                "model_a": {
                    "name": "Logistic Regression (Fine-tuned)",
                    "supported_ratio": f"{stats_a['unsupported_count']}/{stats_a['total_checked']}",
                    "accuracy_estimate": f"{stats_a['support_rate']:.0%}",
                    "method": "Embedding Similarity Features"
                },
                "model_b": {
                    "name": "Keyword Overlap (Baseline)",
                    "supported_ratio": hallucination_result_b["supported_ratio"],
                    "accuracy_estimate": f"{hallucination_result_b['overall_score']:.0%}",
                    "method": "N-gram Token Matching"
                }
            },
            "framing": {
                "model_a": {
                    "name": "DistilBERT Classifier (Neural)",
                    # Show style consistency score
                    "keywords": {
                        m: [f"Style Match: {score:.1%}"] 
                        for m, score in framing_neural.items()
                    },
                    "method": "Fine-tuned Transformer"
                },
                "model_b": {
                    "name": "TF-IDF Analyzer (Statistical)",
                    "keywords": {
                        m: [k[0] for k in data.get("top_keywords", [])][:3] 
                        for m, data in framing_statistical.items()
                    },
                    "method": "Keyword Frequency Analysis"
                }
            },
            "confidence": {
                "model_a": {
                    "name": "Random Forest Regressor",
                    "score": confidence_a,
                    "score_percent": f"{int(confidence_a * 100)}%",
                    "method": "Trained on Retreival Features"
                },
                "model_b": {
                    "name": "Heuristic Scorer",
                    "score": confidence_b["confidence"],
                    "score_percent": confidence_b["confidence_percent"],
                    "method": "Rule-based Weighted Sum"
                }
            }
        }
        
        # Extract unsupported claims from Model B (filtered)
        unsupported_b = []
        for d in hallucination_result_b.get("details", []):
            if not d.get("is_supported", True):
                # Apply similar filtering as Model A (skip refusals/headers)
                s_lower = d["sentence"].lower()
                if "retrieved sources do not contain" in s_lower or "sufficient information" in s_lower:
                    continue
                if d["sentence"].strip().startswith("##"):
                    continue
                    
                unsupported_b.append({
                    "sentence": d["sentence"], 
                    "confidence": d["confidence"]
                })
        
        return {
            "answer": final_answer,
            "raw_answer": raw_response,
            "framing_analysis": framing_statistical,
            "verification_summary": verification_result_a,
            "unsupported_claims": unsupported_a,
            "unsupported_claims_b": unsupported_b,
            "confidence_score": confidence_a,
            "sources": retrieved_chunks,
            "model_comparison": model_comparison,
            "source_quality": {
                "max_similarity": round(max([c['similarity'] for c in all_chunks_flat]) if all_chunks_flat else 0, 3),
                "avg_similarity": round(sum([c['similarity'] for c in all_chunks_flat]) / len(all_chunks_flat) if all_chunks_flat else 0, 3),
                "total_sources": len(all_chunks_flat),
                "coverage_note": "Articles are stored as snapshots; original URLs may have changed."
            }
        }
        
    def _analyze_hallucinations_aggregated(
        self, 
        response_text: str, 
        source_texts: List[str]
    ) -> tuple:
        """
        Analyze hallucinations INTERNALLY per-sentence, but DON'T annotate each one.
        Returns aggregated results for UX-friendly display.
        
        Returns: (verification_summary, unsupported_list, stats_dict)
        """
        # Split on punctuation lookbehind OR newlines (to handle markdown headers)
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n+', response_text) if s.strip()]
        
        unsupported = []
        supported_count = 0
        total_checked = 0
        
        # Pre-compute source embeddings once
        source_embs = self.embedder.generate(source_texts, show_progress=False)
        
        for sentence in sentences:
            # Skip short sentences
            if len(sentence.split()) < 4:
                continue
                
            # Skip refusal/direct answer headers or standard refusal phrases
            sentence_lower = sentence.lower()
            if "retrieved sources do not contain" in sentence_lower or "sufficient information" in sentence_lower:
                continue
            if "framing analysis" in sentence_lower or "explicit framing" in sentence_lower:
                continue
            if sentence.strip().startswith("##"):
                continue
            
            # Clean sentence for detection (remove citations and markdown)
            # 1. Remove [Source] type citations
            clean_sentence = re.sub(r'\[.*?\]', '', sentence).strip()
            # 2. Remove leading bullets/dashes
            clean_sentence = re.sub(r'^[\-\*•]\s+', '', clean_sentence).strip()
            
            # Skip if empty after cleaning
            if not clean_sentence or len(clean_sentence.split()) < 3:
                continue

            total_checked += 1
            sent_emb = self.embedder.generate_single(clean_sentence)
            
            result = self.hallucination_detector.predict(
                sent_emb, 
                source_embs, 
                clean_sentence, # Use clean text for overlap check
                source_texts
            )
            
            if result['is_supported']:
                supported_count += 1
            else:
                unsupported.append({
                    'sentence': sentence,
                    'confidence': result['confidence'],
                    'reason': result.get('reason', 'Low similarity to source material')
                })
        
        # Calculate verification level
        support_rate = supported_count / total_checked if total_checked > 0 else 1.0
        
        # Determine overall verification status
        if support_rate >= 0.8:
            verification_level = "high"
            verification_note = ""  # No disclaimer needed
        elif support_rate >= 0.5:
            verification_level = "moderate"
            verification_note = "\n\n---\n*[!] Framing Inference: Some analytical claims may not be explicitly stated in the retrieved sources.*"
        else:
            verification_level = "low"
            verification_note = "\n\n---\n*[!] Hallucination Risk: Due to limited source coverage, some claims in this analysis require additional confirmation from original sources.*"
        
        verification_summary = {
            "level": verification_level,
            "support_rate": support_rate,
            "supported_count": supported_count,
            "total_checked": total_checked,
            "unsupported_count": len(unsupported)
        }
        
        stats = {
            "supported_count": supported_count,
            "unsupported_count": len(unsupported),
            "total_checked": total_checked,
            "supported_ratio": f"{supported_count}/{total_checked}",
            "support_rate": support_rate,
            "verification_note": verification_note
        }
                
        return verification_summary, unsupported, stats
    
    def _build_final_answer(
        self, 
        raw_response: str, 
        verification_summary: Dict, 
        unsupported_claims: List[Dict]
    ) -> str:
        """
        Build final answer with academic disclaimer and categorized claim warnings.
        NO per-sentence [Unverified] spam!
        Uses softer language and claim categories instead of full sentences.
        """
        final_answer = raw_response
        
        # Always add global academic disclaimer at the end
        academic_disclaimer = "\n\n---\n*This analysis focuses on media framing and emphasis. Some conclusions are interpretive statements and depend on the scope of retrieved articles.*"
        
        # For low verification, add categorized claims (not full sentences)
        if verification_summary["level"] == "low" and unsupported_claims:
            # Extract claim categories from sentences
            claim_categories = self._extract_claim_categories(unsupported_claims)
            
            if claim_categories:
                final_answer += "\n\n---\n**Interpretive Claims with Limited Source Support:**\n"
                for category in claim_categories[:4]:  # Max 4 categories
                    final_answer += f"- {category}\n"
                
                # Add interpretive note
                final_answer += "\n*Interpretive statements (Framing Inferences) are evaluated conservatively and may be flagged despite being reasonable in context.*"
        elif verification_summary["level"] == "moderate":
             final_answer += "\n\n---\n*Note: Contains some Framing Inferences not explicitly in source text.*"
        
        # Add global disclaimer
        final_answer += academic_disclaimer
        
        return final_answer
    
    def _extract_claim_categories(self, unsupported_claims: List[Dict]) -> List[str]:
        """
        Extract high-level claim categories from unsupported sentences.
        Instead of showing full sentences, show thematic categories.
        """
        categories = []
        
        # Keywords to category mapping
        category_patterns = [
            (["jokowi", "widodo", "president", "facilitat"], "Role of President Jokowi in the process"),
            (["opposition", "rival", "opponent", "accept"], "Acceptance by former opposition figures"),
            (["campaign", "strateg", "victory", "success"], "Attribution of campaign success factors"),
            (["cabinet", "minister", "appoint", "position"], "Cabinet appointment decisions"),
            (["coalition", "party", "political", "alliance"], "Political coalition dynamics"),
            (["economic", "invest", "market", "business"], "Economic implications and outlook"),
            (["diplomatic", "foreign", "international"], "Diplomatic and foreign relations"),
            (["military", "defense", "security"], "Military and security aspects"),
            (["social", "program", "meal", "welfare"], "Social program initiatives"),
            (["constitutional", "court", "legal", "law"], "Constitutional and legal matters"),
        ]
        
        seen_categories = set()
        
        for claim in unsupported_claims:
            sentence_lower = claim['sentence'].lower()
            
            for keywords, category in category_patterns:
                if any(kw in sentence_lower for kw in keywords):
                    if category not in seen_categories:
                        categories.append(category)
                        seen_categories.add(category)
                    break
        
        # If no patterns matched, add generic category
        if not categories and unsupported_claims:
            categories.append("General interpretive framing statements")
        
        return categories

