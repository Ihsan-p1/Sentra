"""
API Routes
Updated: Multi-turn chatbot support with session management
"""
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime
import secrets
import uuid

import sys
sys.path.append('..')
from config.settings import settings
from chatbot.engine import ChatbotEngine
from database.connection import db_manager
from pipeline.ingest import ArticleIngestor

# Marks articles that arrived through the API rather than from a corpus file.
# Without it these rows land with corpus_version NULL, which the startup
# consistency check correctly reports as a mixed-version corpus.
API_CORPUS_VERSION = "api-ingest"

router = APIRouter()
chatbot = None
ingestor = None

def init_components():
    """Initialize chatbot and ingestor components"""
    global chatbot, ingestor
    if chatbot is None:
        print("Initializing Chatbot Engine...")
        chatbot = ChatbotEngine()
    if ingestor is None:
        print("Initializing Article Ingestor...")
        ingestor = ArticleIngestor()
    print("Components initialized.")

class ChatRequest(BaseModel):
    message: str
    mode: Optional[str] = "default"
    session_id: Optional[str] = None  # For conversation continuity

class ClearChatRequest(BaseModel):
    session_id: str

class ArticleRequest(BaseModel):
    title: str
    content: str
    media_source: str
    url: Optional[str] = None
    author: Optional[str] = None
    category: Optional[str] = None
    # Was missing entirely, so every API-ingested article stored a NULL date
    # while corpus-ingested ones carry a real one.
    published_date: Optional[datetime] = None

@router.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Chat endpoint with multi-turn conversation support.
    Returns answer + analysis + confidence score + session_id.
    """
    if chatbot is None:
        raise HTTPException(status_code=503, detail="System is still initializing, please try again in a few seconds")
    
    try:
        # Generate or use existing session_id
        session_id = request.session_id or str(uuid.uuid4())
        
        response = await chatbot.process_query(
            query=request.message,
            session_id=session_id,
            mode=request.mode
        )
        
        # Ensure session_id is in response
        response["session_id"] = session_id

        return response
    except Exception as e:
        # The traceback goes to the server log; the client gets a generic message.
        # Echoing str(e) back leaked internal paths and connection strings.
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail="Failed to process the query. See server logs for details.",
        )

@router.post("/chat/clear")
async def clear_chat_endpoint(request: ClearChatRequest):
    """
    Clear conversation history for a session.
    Useful for starting a new conversation.
    """
    if chatbot is None:
        raise HTTPException(status_code=503, detail="System is still initializing")
    
    try:
        success = chatbot.memory.clear_session(request.session_id)
        return {
            "status": "cleared" if success else "session_not_found",
            "session_id": request.session_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def _require_ingest_key(provided: Optional[str]) -> None:
    """
    Gate write access to the corpus.

    Ingested articles become sources the chatbot cites, so an open endpoint lets
    anyone put words in its mouth. When INGEST_API_KEY is unset the endpoint stays
    open for local development, and startup logs a warning.
    """
    expected = settings.INGEST_API_KEY
    if not expected:
        return

    if not provided or not secrets.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


@router.post("/ingest")
async def ingest_endpoint(article: ArticleRequest, x_api_key: Optional[str] = Header(default=None)):
    """
    Ingest a new article into the system. Requires X-API-Key when INGEST_API_KEY is set.
    """
    _require_ingest_key(x_api_key)

    if ingestor is None:
        raise HTTPException(status_code=503, detail="System is still initializing")

    try:
        article_data = article.model_dump()
        article_id = await ingestor.ingest_article(
            article_data, corpus_version=API_CORPUS_VERSION
        )
        return {"status": "success", "article_id": article_id}
    except ValueError as e:
        # Unsupported media source is the caller's mistake, not a server fault.
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"[ERROR] Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail="Ingestion failed")

@router.get("/health")
async def health_check():
    """
    Real component status, not a fixed string.

    This previously returned `{"status": "ok", "models": "loaded"}` unconditionally
    - it reported healthy before initialisation had run, with the database down,
    and with Model A missing. A health check that cannot fail is not a health
    check, and it is the first endpoint anyone reviewing the project will call.
    """
    checks: Dict[str, Any] = {
        "components_initialized": chatbot is not None and ingestor is not None,
    }

    if chatbot is not None:
        # Model A is optional: the engine falls back to untrained instances and
        # warns. That is a degraded state and should be visible as one.
        checks["model_a_trained"] = bool(
            getattr(chatbot.hallucination_detector, "is_trained", False)
            and getattr(chatbot.confidence_scorer, "is_trained", False)
        )
        checks["framing_bert_loaded"] = bool(getattr(chatbot.bert_framing, "is_loaded", False))
        checks["sessions"] = chatbot.memory.stats()

    corpus: Dict[str, Any] = {}
    if ingestor is not None:
        try:
            versions = await ingestor.get_corpus_versions()
            chunk_count = await db_manager.fetchval(
                "SELECT COUNT(*) FROM article_chunks WHERE embedding IS NOT NULL"
            )
            checks["database_reachable"] = True
            corpus = {
                "versions": versions,
                "chunks": chunk_count,
                "embedding_model": settings.EMBEDDING_MODEL,
                # More than one version means a partial re-ingest left the corpus mixed.
                "consistent": len(versions) <= 1,
            }
        except Exception as e:
            checks["database_reachable"] = False
            corpus = {"error": str(e)}

    degraded = [name for name, value in checks.items() if value is False]
    if corpus.get("consistent") is False:
        degraded.append("corpus_consistent")

    return {
        "status": "ok" if not degraded else "degraded",
        "degraded": degraded,
        "checks": checks,
        "corpus": corpus,
        "llm": {"model": settings.LLM_MODEL, "base_url": settings.LLM_BASE_URL},
        "version": "2.0-chatbot",
    }

