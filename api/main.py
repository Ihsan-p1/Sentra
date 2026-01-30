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
from database.connection import db_manager, init_database
from api.routes import router, init_components

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting Sentra API...")
    await init_database()
    init_components()  # Initialize models
    yield
    # Shutdown
    await db_manager.disconnect()
    print("Shutting down Sentra API.")

app = FastAPI(
    title="Sentra - Media Framing Analysis",
    description="RAG-based chatbot with custom evaluation models",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(router, prefix="/api")

# Serve frontend
app.mount("/", StaticFiles(directory="web", html=True), name="web")
