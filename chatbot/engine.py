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
            self.hallucination_detector = HallucinationDetector("models/hallucination_detector.pkl")
            self.confidence_scorer = ConfidenceScorer("models/confidence_scorer.pkl")
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
        """
        # 1. Query Refinement & Context Extraction
        retrieval_query, extracted_topic = self._handle_query_refinement(query, session_id, mode)
        if isinstance(retrieval_query, dict): # Check for refusal (query too short)
            return retrieval_query

        # 2. Retrieval & Gating
        retrieval_result = await self._perform_retrieval(retrieval_query, mode, session_id)
        if "answer" in retrieval_result: # Check for retrieval refusal
            return retrieval_result
        
        retrieved_chunks = retrieval_result["chunks"]
        all_chunks_flat = retrieval_result["flat_chunks"]
        source_texts = retrieval_result["source_texts"]

        # 3. Framing Analysis (A/B)
        framing_results = self._perform_analysis(retrieved_chunks)
        
        # 4. LLM Generation
        conversation_history = self.memory.format_for_llm(session_id, num_turns=5)
        raw_response = await self.llm.generate_conversational_response(
            user_query=query, 
            retrieved_chunks=retrieved_chunks, 
            conversation_history=conversation_history,
            mode=mode,
            focus_topic=extracted_topic
        )
        
        # 5. Hallucination Detection (A/B)
        hallucination_results = self._perform_hallucination_check(raw_response, source_texts)
        
        # 6. Confidence Scoring (A/B)
        confidence_results = self._score_confidence(query, retrieved_chunks, all_chunks_flat)
        
        # 7. Final Response Assembly
        final_answer = self._build_final_answer(
            raw_response, 
            hallucination_results["verification_result_a"], 
            hallucination_results["unsupported_a"]
        )
        
        # 8. Build Comparison Metadata
        model_comparison = self._build_comparison_metadata(
            hallucination_results, 
            framing_results, 
            confidence_results
        )
        
        return {
            "answer": final_answer,
            "raw_answer": raw_response,
            "framing_analysis": framing_results["statistical"],
            "verification_summary": hallucination_results["verification_result_a"],
            "unsupported_claims": hallucination_results["unsupported_a"],
            "unsupported_claims_b": hallucination_results["unsupported_b"],
            "confidence_score": confidence_results["score_a"],
            "sources": retrieved_chunks,
            "model_comparison": model_comparison,
            "source_quality": {
                "max_similarity": round(max([c['similarity'] for c in all_chunks_flat]) if all_chunks_flat else 0, 3),
                "avg_similarity": round(sum([c['similarity'] for c in all_chunks_flat]) / len(all_chunks_flat) if all_chunks_flat else 0, 3),
                "total_sources": len(all_chunks_flat),
                "coverage_note": "Articles are stored as snapshots; original URLs may have changed."
            }
        }

    def _handle_query_refinement(self, query: str, session_id: str, mode: str) -> tuple:
        """Handle short queries, topic extraction, and refinement."""
        if mode == "reduce_hallucination" and len(query.strip().split()) < 3:
            return {
                "answer": "This question cannot be reliably answered using the current news corpus. Please ask a more specific question (at least 3 words).",
                "analysis": {"error": "query_too_short"},
                "confidence": 0.0,
                "model_comparison": {}, 
                "mode": mode,
                "session_id": session_id
            }, None

        retrieval_query = query
        extracted_topic = None
        
        query_lower = query.strip().lower()
        is_short_query = len(query.strip().split()) <= 3
        
        # Continuation & Topic Extraction logic
        continuation_phrases = ["tell me more", "more", "continue", "go on", "what else", "explain", "elaborate"]
        is_continuation = any(phrase in query_lower for phrase in continuation_phrases)
        
        if is_continuation:
            for phrase in ["tell me more about", "tell me about", "more about", "explain about", "elaborate on", "what about"]:
                if phrase in query_lower:
                    proposed_topic = query_lower.split(phrase, 1)[1].strip().strip(".,!?")
                    if len(proposed_topic.split()) >= 1:
                        retrieval_query = proposed_topic
                        extracted_topic = retrieval_query
                        break

        # Follow-up extraction from memory
        affirmative_phrases = ["yes", "yeah", "yep", "sure", "ok", "okay", "please", "go ahead", "ya", "iya"]
        is_affirmative = any(phrase in query_lower for phrase in affirmative_phrases)
        
        if (is_short_query and is_affirmative) or (is_continuation and not extracted_topic):
            history = self.memory.get_history(session_id)
            if history:
                for msg in reversed(history):
                    if msg["role"] == "assistant":
                        patterns = [
                            r"would you like to (?:know|learn|hear) more about (.+?)\?",
                            r"want (?:me )?to (?:tell|explain) (?:you )?(?:more )?about (.+?)\?",
                            r"interested in (?:learning|knowing) (?:more )?about (.+?)\?",
                            r"shall i (?:tell|explain) (?:you )?(?:more )?about (.+?)\?",
                        ]
                        for pattern in patterns:
                            match = re.search(pattern, msg["content"].lower())
                            if match:
                                match_topic = match.group(1).strip().rstrip('.,!?')
                                if len(match_topic) > 10:
                                    extracted_topic = match_topic
                                    retrieval_query = extracted_topic
                                    break
                        if extracted_topic: break
                
                if not extracted_topic:
                    for msg in reversed(history):
                        if msg["role"] == "user" and len(msg["content"].split()) > 3:
                            retrieval_query = msg["content"]
                            break
                            
        return retrieval_query, extracted_topic

    async def _perform_retrieval(self, query: str, mode: str, session_id: str) -> Dict[str, Any]:
        """Perform document retrieval and validation."""
        top_k = 4 if mode == "reduce_hallucination" else None
        retrieved_chunks = await self.retriever.retrieve(query, top_k=top_k)
        all_chunks_flat = self.retriever.flatten_results(retrieved_chunks)
        source_texts = [c['text'] for c in all_chunks_flat]
        
        if not source_texts:
            return {
                "answer": "Sorry, I couldn't find relevant information in the news database.",
                "analysis": {}, "confidence": 0.0, "model_comparison": {}, "session_id": session_id
            }
        
        similarities = [c['similarity'] for c in all_chunks_flat]
        max_similarity = max(similarities)
        threshold = 0.40 if mode == "reduce_hallucination" else 0.35
        
        if max_similarity < threshold:
            refusal_msg = "Insufficient source coverage." if mode == "reduce_hallucination" else "I cannot answer this based on the available news sources. The retrieved articles do not contain sufficiently relevant information. Please try asking about specific Indonesian political events or candidates covered in the database."
            return {
                "answer": refusal_msg,
                "analysis": {"retrieval_quality": "insufficient"},
                "confidence": 0.0, "model_comparison": {},
                "source_quality": {
                    "max_similarity": round(max_similarity, 3),
                    "avg_similarity": round(sum(similarities)/len(similarities), 3),
                    "total_sources": len(all_chunks_flat),
                    "refusal_reason": f"max_sim={max_similarity:.2f} < {threshold}",
                    "mode": mode
                },
                "session_id": session_id
            }

        if mode == "reduce_hallucination":
            for media in retrieved_chunks:
                retrieved_chunks[media] = retrieved_chunks[media][:2]
                
        return {
            "chunks": retrieved_chunks,
            "flat_chunks": all_chunks_flat,
            "source_texts": source_texts
        }

    def _perform_analysis(self, retrieved_chunks: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Perform framing analysis using Model A (Neural) and Model B (Statistical)."""
        texts_by_media = {
            media: [c['text'] for c in chunks]
            for media, chunks in retrieved_chunks.items()
        }
        
        # Model B: Statistical (TF-IDF)
        framing_statistical = self.framing_analyzer.analyze_media_framing(texts_by_media)
        
        # Model A: Neural (DistilBERT)
        framing_neural = {}
        for media, texts in texts_by_media.items():
            if texts:
                combined_text = " ".join(texts)[:512]
                prediction = self.bert_framing.predict_source_style(combined_text)
                framing_neural[media] = prediction.get(media, 0.0)
                
        return {
            "statistical": framing_statistical,
            "neural": framing_neural
        }

    def _perform_hallucination_check(self, response: str, source_texts: List[str]) -> Dict[str, Any]:
        """Perform hallucination detection with Model A and Model B."""
        # Model A: Trained Logistic Regression
        verification_result_a, unsupported_a, stats_a = self._analyze_hallucinations_aggregated(
            response, source_texts
        )
        
        # Model B: Keyword overlap
        hallucination_result_b = self.baseline_models["hallucination"].check_response(
            response, source_texts
        )
        
        # Clean up Model B unsupported claims
        unsupported_b = []
        for d in hallucination_result_b.get("details", []):
            if not d.get("is_supported", True):
                s_lower = d["sentence"].lower()
                if any(x in s_lower for x in ["retrieved sources do not contain", "sufficient information"]):
                    continue
                if d["sentence"].strip().startswith("##"):
                    continue
                unsupported_b.append({"sentence": d["sentence"], "confidence": d["confidence"]})
                
        return {
            "verification_result_a": verification_result_a,
            "unsupported_a": unsupported_a,
            "stats_a": stats_a,
            "hallucination_result_b": hallucination_result_b,
            "unsupported_b": unsupported_b
        }

    def _score_confidence(self, query: str, chunks: Dict, flat_chunks: List) -> Dict[str, Any]:
        """Score response confidence using Model A and Model B."""
        all_sims = [c['similarity'] for c in flat_chunks]
        score_a = self.confidence_scorer.predict(all_sims, len(chunks))
        score_b_data = self.baseline_models["confidence"].score(query, chunks)
        
        return {
            "score_a": score_a,
            "score_b_data": score_b_data
        }

    def _build_comparison_metadata(
        self, 
        hallucination_res: Dict, 
        framing_res: Dict, 
        confidence_res: Dict
    ) -> Dict[str, Any]:
        """Construct the competition/comparison metadata for the UI."""
        return {
            "hallucination": {
                "model_a": {
                    "name": "Logistic Regression (Fine-tuned)",
                    "supported_ratio": f"{hallucination_res['stats_a']['unsupported_count']}/{hallucination_res['stats_a']['total_checked']}",
                    "accuracy_estimate": f"{hallucination_res['stats_a']['support_rate']:.0%}",
                    "method": "Embedding Similarity Features"
                },
                "model_b": {
                    "name": "Keyword Overlap (Baseline)",
                    "supported_ratio": hallucination_res['hallucination_result_b']["supported_ratio"],
                    "accuracy_estimate": f"{hallucination_res['hallucination_result_b']['overall_score']:.0%}",
                    "method": "N-gram Token Matching"
                }
            },
            "framing": {
                "model_a": {
                    "name": "DistilBERT Classifier (Neural)",
                    "keywords": {
                        m: [f"Style Match: {score:.1%}"] 
                        for m, score in framing_res["neural"].items()
                    },
                    "method": "Fine-tuned Transformer"
                },
                "model_b": {
                    "name": "TF-IDF Analyzer (Statistical)",
                    "keywords": {
                        m: [k[0] for k in data.get("top_keywords", [])][:3] 
                        for m, data in framing_res["statistical"].items()
                    },
                    "method": "Keyword Frequency Analysis"
                }
            },
            "confidence": {
                "model_a": {
                    "name": "Random Forest Regressor",
                    "score": confidence_res["score_a"],
                    "score_percent": f"{int(confidence_res['score_a'] * 100)}%",
                    "method": "Trained on Retreival Features"
                },
                "model_b": {
                    "name": "Heuristic Scorer",
                    "score": confidence_res["score_b_data"]["confidence"],
                    "score_percent": confidence_res["score_b_data"]["confidence_percent"],
                    "method": "Rule-based Weighted Sum"
                }
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

