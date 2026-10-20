"""
Integration of RAG + Custom Models + LLM Generator
The 'Brain' of the chatbot with A/B Model Comparison.

Improved: Aggregated hallucination warnings instead of per-sentence spam.
"""
from typing import Dict, List, Any
import asyncio
import os
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
from chatbot.llm_client import LLMClient, LLMUnavailable
from chatbot.conversation_memory import conversation_memory
from chatbot.sentences import prepare_for_check, split_sentences
from chatbot.language import (
    INTENT_HELP,
    INTENT_QUESTION,
    claim_category,
    classify_intent,
    detect_help_topic,
    detect_language,
    disclaimer,
    help_reply,
    refusal,
    small_talk_reply,
)
from utils.trace import format_similarity_spread, trace_request

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
            self.hallucination_detector = HallucinationDetector(
                os.path.join(settings.MODELS_DIR, "hallucination_detector.pkl")
            )
            self.confidence_scorer = ConfidenceScorer(
                os.path.join(settings.MODELS_DIR, "confidence_scorer.pkl")
            )
            print("[INFO] Loaded custom evaluation models (Model A)")
        except Exception as e:
            print(f"[WARN] Could not load custom models ({e}); using untrained instance. "
                  "Run 'python scripts/train_models.py' to generate them.")
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
        with trace_request(query, session_id) as trace:
            trace.set(mode=mode)

            # 0. Answer in whatever language the user wrote in, and divert small talk
            #    before it reaches retrieval - a greeting has no sources to find.
            language = detect_language(query)
            intent = classify_intent(query)
            trace.set(lang=language, intent=intent)

            if intent != INTENT_QUESTION:
                trace.set(outcome="small_talk")
                return self._small_talk_response(query, intent, language, mode, session_id)

            # 1. Query Refinement & Context Extraction
            retrieval_query, extracted_topic = self._handle_query_refinement(
                query, session_id, mode, language
            )
            if isinstance(retrieval_query, dict): # Check for refusal (query too short)
                trace.set(outcome="refused", gate="too_short")
                self._remember(session_id, query, retrieval_query["answer"])
                return retrieval_query
            if extracted_topic:
                trace.set(refined="yes")

            # 2. Retrieval & Gating
            with trace.stage("retrieval"):
                retrieval_result = await self._perform_retrieval(
                    retrieval_query, mode, session_id, language, trace
                )
            if "answer" in retrieval_result: # Check for retrieval refusal
                trace.set(outcome="refused")
                self._remember(session_id, query, retrieval_result["answer"])
                return retrieval_result

            retrieved_chunks = retrieval_result["chunks"]
            all_chunks_flat = retrieval_result["flat_chunks"]
            source_texts = retrieval_result["source_texts"]

            # 3. Framing Analysis (A/B)
            with trace.stage("framing"):
                framing_results = self._perform_analysis(retrieved_chunks)

            # 4. LLM Generation
            conversation_history = self.memory.format_for_llm(session_id, num_turns=5)
            try:
                with trace.stage("generate"):
                    raw_response = await self.llm.generate_conversational_response(
                        user_query=query,
                        retrieved_chunks=retrieved_chunks,
                        conversation_history=conversation_history,
                        mode=mode,
                        focus_topic=extracted_topic,
                        language=language
                    )
            except LLMUnavailable as exc:
                # No completion means there is nothing to verify or score. Return
                # an error envelope instead of grading the error text as an answer,
                # and do NOT write it to memory - a failed turn must not become the
                # context a follow-up resolves against.
                trace.set(outcome="llm_unavailable", error=type(exc).__name__)
                return self._llm_error_response(exc, language, mode, session_id, retrieved_chunks)

            trace.set(answer_chars=len(raw_response or ""))

            # 5. Hallucination Detection (A/B)
            with trace.stage("verify"):
                hallucination_results = self._perform_hallucination_check(raw_response, source_texts)

            # 6. Confidence Scoring (A/B)
            confidence_results = self._score_confidence(query, retrieved_chunks, all_chunks_flat)

            stats_a = hallucination_results["stats_a"]
            trace.set(
                support=f"{stats_a['supported_count']}/{stats_a['total_checked']}",
                conf_a=confidence_results["score_a"],
                outcome="answered",
            )

            # 7. Final Response Assembly
            final_answer = self._build_final_answer(
                raw_response,
                hallucination_results["verification_result_a"],
                hallucination_results["unsupported_a"],
                language
            )

            # 8. Build Comparison Metadata
            model_comparison = self._build_comparison_metadata(
                hallucination_results,
                framing_results,
                confidence_results
            )

            # 9. Persist the turn so the NEXT request can resolve follow-ups against it.
            # Store the raw answer: the appended disclaimers are display-only noise in history.
            self._remember(session_id, query, raw_response)

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
                "language": language,
                "source_quality": {
                    "max_similarity": round(max([c['similarity'] for c in all_chunks_flat]) if all_chunks_flat else 0, 3),
                    "avg_similarity": round(sum([c['similarity'] for c in all_chunks_flat]) / len(all_chunks_flat) if all_chunks_flat else 0, 3),
                    "total_sources": len(all_chunks_flat),
                    "coverage_note": "Articles are stored as snapshots; original URLs may have changed."
                }
            }

    def _small_talk_response(
        self,
        query: str,
        intent: str,
        language: str,
        mode: str,
        session_id: str
    ) -> Dict[str, Any]:
        """
        Answer a greeting, a thanks, a farewell, or a question about the app
        itself, without touching the corpus.

        Returns the same envelope as a normal answer so the frontend needs no
        special case; the analysis panels are simply empty because no retrieval
        or verification took place.
        """
        if intent == INTENT_HELP:
            # Which panel or feature was asked about, falling back to an overview
            answer = help_reply(detect_help_topic(query) or "overview", language)
        else:
            answer = small_talk_reply(intent, language)

        self._remember(session_id, query, answer)

        return {
            "answer": answer,
            "raw_answer": answer,
            "intent": intent,
            "language": language,
            "framing_analysis": {},
            "verification_summary": {},
            "unsupported_claims": [],
            "unsupported_claims_b": [],
            "confidence_score": 0.0,
            "sources": {},
            "model_comparison": {},
            "mode": mode,
            "session_id": session_id,
        }

    def _llm_error_response(
        self,
        exc: "LLMUnavailable",
        language: str,
        mode: str,
        session_id: str,
        retrieved_chunks: Dict,
    ) -> Dict[str, Any]:
        """
        Envelope for "the model server did not answer".

        Retrieval succeeded, so the sources are still returned - they are real and
        worth showing. Everything downstream of generation is empty, which makes
        the sidebar render "Not evaluated" rather than inventing a score for text
        that is not an answer.
        """
        return {
            "answer": f"{refusal('llm_unavailable', language)}\n\n`{exc.user_message()}`",
            "raw_answer": "",
            "error": "llm_unavailable",
            "language": language,
            "framing_analysis": {},
            "verification_summary": {},
            "unsupported_claims": [],
            "unsupported_claims_b": [],
            "confidence_score": 0.0,
            "sources": retrieved_chunks,
            "model_comparison": {},
            "mode": mode,
            "session_id": session_id,
        }

    def _remember(self, session_id: str, query: str, answer: str) -> None:
        """
        Record one completed turn in conversation memory.

        Called at every exit point of process_query (including refusals) so that
        follow-up resolution in _handle_query_refinement has something to read.
        Recording happens AFTER refinement on purpose: the refinement step must
        only see previous turns, never the message currently being answered.
        """
        self.memory.add_message(session_id, "user", query)
        self.memory.add_message(session_id, "assistant", answer)

    def _handle_query_refinement(
        self, query: str, session_id: str, mode: str, language: str = "en"
    ) -> tuple:
        """Handle short queries, topic extraction, and refinement."""
        if mode == "reduce_hallucination" and len(query.strip().split()) < 3:
            return {
                "answer": refusal("too_short", language),
                "analysis": {"error": "query_too_short"},
                "confidence": 0.0,
                "model_comparison": {},
                "language": language,
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

    async def _perform_retrieval(
        self, query: str, mode: str, session_id: str, language: str = "en", trace=None
    ) -> Dict[str, Any]:
        """Perform document retrieval and validation."""
        top_k = 4 if mode == "reduce_hallucination" else None
        retrieved_chunks = await self.retriever.retrieve(query, top_k=top_k)
        all_chunks_flat = self.retriever.flatten_results(retrieved_chunks)
        source_texts = [c['text'] for c in all_chunks_flat]

        if not source_texts:
            if trace:
                trace.set(gate="no_sources", chunks=0)
            return {
                "answer": refusal("no_sources", language),
                "analysis": {}, "confidence": 0.0, "model_comparison": {},
                "language": language, "session_id": session_id
            }

        similarities = [c['similarity'] for c in all_chunks_flat]
        max_similarity = max(similarities)
        threshold = (
            settings.RETRIEVAL_THRESHOLD_STRICT
            if mode == "reduce_hallucination"
            else settings.RETRIEVAL_THRESHOLD
        )

        if trace:
            # sim is max/median/min. All three sliding down together is the
            # signature of an embedding mismatch, not of a difficult query -
            # that distinction is why the spread is logged and not just the max.
            trace.set(
                sim=format_similarity_spread(similarities),
                threshold=threshold,
                chunks=f"{len(all_chunks_flat)}/{len(retrieved_chunks)}src",
                gate="pass" if max_similarity >= threshold else "below_threshold",
            )

        if max_similarity < threshold:
            kind = "low_similarity_strict" if mode == "reduce_hallucination" else "low_similarity"
            return {
                "answer": refusal(kind, language),
                "analysis": {"retrieval_quality": "insufficient"},
                "confidence": 0.0, "model_comparison": {},
                "language": language,
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
            # Note: the ratio reported here is UNSUPPORTED/TOTAL sentences (lower is
            # better), and support_rate is the share that passed verification.
            "hallucination": {
                "model_a": {
                    "name": "Logistic Regression (Fine-tuned)",
                    "unsupported_ratio": f"{hallucination_res['stats_a']['unsupported_count']}/{hallucination_res['stats_a']['total_checked']}",
                    "support_rate": f"{hallucination_res['stats_a']['support_rate']:.0%}",
                    "method": "Embedding Similarity Features"
                },
                "model_b": {
                    "name": "Keyword Overlap (Baseline)",
                    "unsupported_ratio": hallucination_res['hallucination_result_b']["unsupported_ratio"],
                    "support_rate": f"{hallucination_res['hallucination_result_b']['overall_score']:.0%}",
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
        unsupported = []
        supported_count = 0
        total_checked = 0
        
        # Pre-compute source embeddings once
        source_embs = self.embedder.generate(source_texts, show_progress=False)
        
        for sentence in split_sentences(response_text):
            # Headings, refusals and short fragments are skipped (chatbot/sentences.py)
            clean_sentence = prepare_for_check(sentence)
            if clean_sentence is None:
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
        unsupported_claims: List[Dict],
        language: str = "en"
    ) -> str:
        """
        Build final answer with academic disclaimer and categorized claim warnings.
        NO per-sentence [Unverified] spam!
        Uses softer language and claim categories instead of full sentences.

        Disclaimers follow the answer's language so a reply in Indonesian does
        not end with an English footer.
        """
        final_answer = raw_response

        # For low verification, add categorized claims (not full sentences)
        if verification_summary["level"] == "low" and unsupported_claims:
            claim_categories = self._extract_claim_categories(unsupported_claims, language)

            if claim_categories:
                final_answer += disclaimer("claims_header", language)
                for category in claim_categories[:4]:  # Max 4 categories
                    final_answer += f"- {category}\n"
                final_answer += disclaimer("claims_footer", language)
        elif verification_summary["level"] == "moderate":
            final_answer += disclaimer("moderate", language)

        # Always add the global academic disclaimer at the end
        final_answer += disclaimer("academic", language)

        return final_answer

    # Keyword -> category key. Keys are language-independent; the display text
    # is looked up per language, so detection never has to be translated.
    CATEGORY_PATTERNS = [
        (["jokowi", "widodo", "president", "facilitat", "presiden"], "jokowi_role"),
        (["opposition", "rival", "opponent", "accept", "oposisi", "lawan"], "opposition"),
        (["campaign", "strateg", "victory", "success", "kampanye", "kemenangan"], "campaign"),
        (["cabinet", "minister", "appoint", "position", "kabinet", "menteri"], "cabinet"),
        (["coalition", "party", "political", "alliance", "koalisi", "partai"], "coalition"),
        (["economic", "invest", "market", "business", "ekonomi", "investasi", "pasar"], "economic"),
        (["diplomatic", "foreign", "international", "diplomatik", "luar negeri"], "diplomatic"),
        (["military", "defense", "security", "militer", "pertahanan", "keamanan"], "military"),
        (["social", "program", "meal", "welfare", "sosial", "kesejahteraan"], "social"),
        (["constitutional", "court", "legal", "law", "konstitusi", "mahkamah", "hukum"], "legal"),
    ]

    def _extract_claim_categories(
        self, unsupported_claims: List[Dict], language: str = "en"
    ) -> List[str]:
        """
        Extract high-level claim categories from unsupported sentences.
        Instead of showing full sentences, show thematic categories.
        """
        categories = []
        seen_keys = set()

        for claim in unsupported_claims:
            sentence_lower = claim['sentence'].lower()

            for keywords, key in self.CATEGORY_PATTERNS:
                if any(kw in sentence_lower for kw in keywords):
                    if key not in seen_keys:
                        categories.append(claim_category(key, language))
                        seen_keys.add(key)
                    break

        # If no patterns matched, add generic category
        if not categories and unsupported_claims:
            categories.append(claim_category("generic", language))

        return categories

