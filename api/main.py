"""
FastAPI Main Application
Routes and server configuration.
"""
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

import sys
sys.path.append('..')
from config.settings import settings
from database.connection import db_manager, init_database
from api.routes import router, init_components

# This project logs with print(). When stdout is a pipe rather than a terminal -
# which it is whenever the server runs under a process manager, in Docker, or
# with output redirected to a file - Python block-buffers it, so log lines sit
# unwritten for as long as it takes to fill 8KB. Startup diagnostics and
# per-request traces then appear minutes late or, if the process is killed, never.
# Line buffering costs nothing here and makes the logs trustworthy.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting Sentra API...")
    if not settings.INGEST_API_KEY:
        print("[WARN] INGEST_API_KEY is not set - POST /api/ingest is unauthenticated. "
              "Set it in .env before exposing this server beyond localhost.")
    await init_database()
    init_components()  # Initialize models

    # Fail loudly on a stale corpus. A model swap without re-ingesting silently
    # degrades every retrieval, and nothing else in the stack notices.
    from api.routes import ingestor
    check = await ingestor.verify_embeddings_match_model()
    if not check["ok"]:
        print(f"[ERROR] Stored embeddings are stale: {check['detail']}")
        print(f"[ERROR] {check}")
    elif check["total_chunks"]:
        print(f"[INFO] Embedding check passed: {check['total_chunks']} chunks labelled "
              f"{check['model']}, {check['checked']} re-embedded and verified")

    # Which corpus the database actually holds. More than one version means a
    # partial re-ingest left the corpus in a mixed state.
    versions = await ingestor.get_corpus_versions()
    if len(versions) > 1:
        print(f"[WARN] Database holds articles from multiple corpus versions: {versions}. "
              "Run scripts/tools/refresh_database.py for a clean rebuild.")
    elif versions:
        version, count = next(iter(versions.items()))
        print(f"[INFO] Corpus: {version} ({count} articles)")

    yield
    # Shutdown
    await db_manager.disconnect()
    print("Shutting down Sentra API.")

app = FastAPI(
    title="Sentra - Media Framing Analysis",
    description="RAG-based chatbot with custom evaluation models",
    lifespan=lifespan
)

# CORS - restricted to the origins that actually serve the UI. Override via
# CORS_ALLOW_ORIGINS in .env when deploying elsewhere.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ALLOW_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)

# Routes
app.include_router(router, prefix="/api")

# Serve frontend
app.mount("/", StaticFiles(directory="web", html=True), name="web")
