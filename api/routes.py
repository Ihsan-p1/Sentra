"""
API Routes
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

import sys
sys.path.append('..')
from chatbot.engine import ChatbotEngine
from pipeline.ingest import ArticleIngestor

router = APIRouter()
chatbot = None
ingestor = None

def init_components():
    """Initialize chatbot and ingestor components"""
    global chatbot, ingestor
    if chatbot is None:
        print("🔄 Initializing Chatbot Engine...")
        chatbot = ChatbotEngine()
    if ingestor is None:
        print("🔄 Initializing Article Ingestor...")
        ingestor = ArticleIngestor()
    print("✅ Components initialized")

class ChatRequest(BaseModel):
    message: str
    mode: Optional[str] = "default"

class ArticleRequest(BaseModel):
    title: str
    content: str
    media_source: str
    url: Optional[str] = None
    author: Optional[str] = None
    category: Optional[str] = None

@router.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Chat endpoint.
    Returns answer + analysis + confidence score.
    """
    if chatbot is None:
        raise HTTPException(status_code=503, detail="System is still initializing, please try again in a few seconds")
    
    try:
        response = await chatbot.process_query(request.message, mode=request.mode)
        return response
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ingest")
async def ingest_endpoint(article: ArticleRequest):
    """
    Ingest a new article into the system.
    """
    if ingestor is None:
        raise HTTPException(status_code=503, detail="System is still initializing")

    try:
        article_data = article.dict()
        article_id = await ingestor.ingest_article(article_data)
        return {"status": "success", "article_id": article_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    return {"status": "ok", "models": "loaded"}
