"""
Main FastAPI Application — Entry point untuk backend chatbot pencarian file.

Jalankan dengan:
    cd backend
    uvicorn app.main:app --reload --port 8001
"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.database import init_db
from app.api.routes import router
from app.services.scheduler import auto_index_scheduler

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Inisialisasi FastAPI
app = FastAPI(
    title="Chatbot Pencarian File",
    description=(
        "Sistem chatbot pencarian file berbasis Sentence-BERT (SBERT) "
        "dan Retrieval-Augmented Generation (RAG)"
    ),
    version="1.0.0",
)

# CORS — agar Streamlit frontend bisa akses API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Daftarkan routes
app.include_router(router, prefix="/api")


@app.on_event("startup")
def startup_event():
    """Dijalankan saat server start: inisialisasi database."""
    logger.info("=" * 60)
    logger.info("  CHATBOT PENCARIAN FILE — Starting up...")
    logger.info("=" * 60)
    logger.info("Inisialisasi database...")
    init_db()
    logger.info("Database siap!")
    auto_index_scheduler.start()
    logger.info("Server berjalan di http://localhost:8001")
    logger.info("Dokumentasi API: http://localhost:8001/docs")
    logger.info("=" * 60)


@app.on_event("shutdown")
def shutdown_event():
    """Dijalankan saat server berhenti: hentikan scheduler background."""
    auto_index_scheduler.stop()


@app.get("/")
def root():
    """Health check / landing endpoint."""
    return {
        "message": "🤖 Chatbot Pencarian File API",
        "docs": "/docs",
        "version": "1.0.0",
        "endpoints": {
            "chat": "POST /api/chat",
            "search": "POST /api/search",
            "debug_lookup": "POST /api/debug/lookup",
            "index": "POST /api/index",
            "incremental_index": "POST /api/index/incremental",
            "stats": "GET /api/stats",
            "history": "GET /api/history",
        },
    }
