"""
API Routes — Semua endpoint FastAPI untuk sistem chatbot pencarian file.

Endpoints:
  POST /api/chat                — Chat + RAG (cari file & generate jawaban)
  POST /api/search              — Pencarian saja (tanpa RAG)
  POST /api/index               — Full indexing
  POST /api/index/incremental   — Incremental indexing
  GET  /api/stats               — Statistik database
  GET  /api/history             — Riwayat chat
"""

import os
import sys
import time
import json
import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.database import get_db
from app.config import settings
from app.models.file_model import File, FileChunk, ChatHistory
from app.services.indexer import IndexingService
from app.services.searcher import SearchService
from app.services.rag import RAGService
from app.services.gdrive_service import GoogleDriveService
from app.services.watch_config import (
    get_watch_config_payload,
    get_effective_watch_targets,
    update_watch_config,
    mark_local_index_started,
    mark_gdrive_index_started,
)
from app.api.schemas import (
    ChatRequest, ChatResponse,
    SearchRequest, SearchResponse,
    IndexRequest, IndexResponse,
    GoogleDriveIndexRequest,
    IndexWatchConfigRequest,
    IndexWatchConfigResponse,
    IndexProgressResponse,
    StatsResponse, FileInfo,
    IndexedDirectoryInfo,
    IndexedFileItem, IndexedFileListResponse,
    GoogleAuthUrlResponse, GoogleAuthStatusResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()

# =====================================================================
# LAZY-LOADED SERVICES (agar model SBERT tidak di-load saat import)
# =====================================================================
_indexer: IndexingService | None = None
_searcher: SearchService | None = None
_rag: RAGService | None = None
_gdrive: GoogleDriveService | None = None


def get_indexer() -> IndexingService:
    global _indexer
    if _indexer is None:
        _indexer = IndexingService()
    return _indexer


def get_searcher() -> SearchService:
    global _searcher
    if _searcher is None:
        _searcher = SearchService()
    return _searcher


def get_rag() -> RAGService:
    global _rag
    if _rag is None:
        _rag = RAGService()
    return _rag


def get_gdrive() -> GoogleDriveService:
    global _gdrive
    if _gdrive is None:
        _gdrive = GoogleDriveService()
    return _gdrive



def _merge_search_results(primary: List[dict], secondary: List[dict]) -> List[dict]:
    """Gabungkan hasil dua retrieval pass sambil menjaga chunk terbaik per file."""
    merged: dict[int, dict] = {}

    def add_results(results: List[dict]):
        for item in results or []:
            file_id = item.get("file_id")
            if file_id is None:
                continue

            existing = merged.get(file_id)
            if existing is None:
                merged[file_id] = dict(item)
                continue

            if float(item.get("max_similarity", 0.0)) > float(existing.get("max_similarity", 0.0)):
                existing["max_similarity"] = float(item.get("max_similarity", 0.0))
                if "rank_score" in item:
                    existing["rank_score"] = float(item.get("rank_score", 0.0))

            existing_chunks = existing.get("relevant_chunks", []) or []
            seen_chunk_ids = {chunk.get("chunk_id") for chunk in existing_chunks}
            for chunk in item.get("relevant_chunks", []) or []:
                chunk_id = chunk.get("chunk_id")
                if chunk_id in seen_chunk_ids:
                    continue
                existing_chunks.append(chunk)
                seen_chunk_ids.add(chunk_id)

            existing["relevant_chunks"] = sorted(
                existing_chunks,
                key=lambda c: c.get("similarity", 0.0),
                reverse=True,
            )

    add_results(primary)
    add_results(secondary)

    return sorted(
        merged.values(),
        key=lambda x: x.get("max_similarity", 0.0),
        reverse=True,
    )


def _tag_results_provenance(results: List[dict], provenance: str) -> None:
    """Tambahkan label provenance pada setiap chunk dalam hasil pencarian."""
    for item in results or []:
        chunks = item.get("relevant_chunks") or []
        for c in chunks:
            # Jangan overwrite jika sudah ada provenance
            if c.get("provenance"):
                continue
            c["provenance"] = provenance


def _virtual_directory_path(file_path: str | None, source: str | None) -> str | None:
    """Ambil path direktori untuk path virtual cloud seperti gdrive:/folder/file."""
    if not file_path:
        return None

    path = str(file_path).strip().replace("\\", "/")
    if not path:
        return None

    normalized_source = (source or "").strip().lower()
    if normalized_source == "gdrive" or path.lower().startswith("gdrive:/"):
        prefix = "gdrive:/"
        if path.lower().startswith(prefix):
            remainder = path[len(prefix):].lstrip("/")
            parts = [part for part in remainder.split("/") if part]
            directory = "/".join(parts[:-1])
            return f"gdrive:/{directory}" if directory else "gdrive:/"

    return os.path.normpath(os.path.dirname(path))


def _plan_query(searcher: SearchService, rag: RAGService, raw_query: str) -> dict:
    """Plan query dulu, lalu siapkan query retrieval yang lebih kuat."""
    planned = rag.plan_query_for_retrieval(raw_query)
    rewritten_query = str(planned.get("rewritten_query") or raw_query).strip()
    keywords = planned.get("keywords") or []

    if keywords:
        keyword_tail = " ".join(str(item) for item in keywords if str(item).strip())
        if keyword_tail and keyword_tail not in rewritten_query.lower():
            rewritten_query = f"{rewritten_query} {keyword_tail}".strip()

    lookup_by_raw = searcher.is_file_lookup_query(raw_query)
    lookup_by_rewrite = searcher.is_file_lookup_query(rewritten_query)
    lookup_by_intent = bool(planned.get("intent") == "file_lookup")
    is_lookup = lookup_by_raw or lookup_by_rewrite or lookup_by_intent

    retrieval_query = raw_query if is_lookup else (rewritten_query or raw_query)

    planned["retrieval_query"] = retrieval_query
    planned["is_lookup"] = is_lookup
    return planned


# =====================================================================
# CHAT ENDPOINT
# =====================================================================
@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, db: Session = Depends(get_db)):
    """Endpoint utama chatbot."""
    start_time = time.time()

    searcher = get_searcher()
    rag = get_rag()

    query_plan = _plan_query(searcher, rag, request.message)
    retrieval_query = str(query_plan.get("retrieval_query") or request.message)
    is_lookup = bool(query_plan.get("is_lookup"))
    candidate_top_k_files = max(int(settings.TOP_K_FILES or 5) * 3, 15) if is_lookup else None

    search_results = searcher.search(retrieval_query, db, top_k_files=candidate_top_k_files)
    primary_tag = "rewritten" if retrieval_query.strip() != request.message.strip() else "raw"
    _tag_results_provenance(search_results, primary_tag)
    if retrieval_query.strip() != request.message.strip():
        raw_results = searcher.search(request.message, db, top_k_files=candidate_top_k_files)
        _tag_results_provenance(raw_results, "raw")
        search_results = _merge_search_results(search_results, raw_results)

    top_k = int(settings.TOP_K_FILES or 5)
    top_results = search_results[:top_k]

    answer = rag.generate_response(
        request.message,
        top_results,
        file_lookup_mode=is_lookup,
    )

    grounded_results = rag.rerank_results_by_answer(answer, top_results)
    grounded_results = grounded_results[:top_k]

    response_time_ms = int((time.time() - start_time) * 1000)

    chat_record = ChatHistory(
        user_message=request.message,
        bot_response=answer,
        retrieved_files=json.dumps([
            {
                "file_name": r["file_name"],
                "file_path": r["file_path"],
                "similarity": r["max_similarity"],
            }
            for r in grounded_results
        ], ensure_ascii=False),
        response_time_ms=response_time_ms,
    )
    db.add(chat_record)
    db.commit()

    retrieved_files = [
        FileInfo(
            file_id=r["file_id"],
            file_name=r["file_name"],
            file_path=r["file_path"],
            source=r.get("source", "local"),
            web_view_link=r.get("web_view_link"),
            file_type=r["file_type"],
            file_size=r["file_size"],
            last_modified=r["last_modified"],
            max_similarity=r["max_similarity"],
            metadata_score=r.get("metadata_score"),
            metadata_exact_hits=r.get("metadata_exact_hits"),
            metadata_partial_hits=r.get("metadata_partial_hits"),
            metadata_fuzzy_hits=r.get("metadata_fuzzy_hits"),
            metadata_name_ratio=r.get("metadata_name_ratio"),
            metadata_path_ratio=r.get("metadata_path_ratio"),
            metadata_text_ratio=r.get("metadata_text_ratio"),
            metadata_overlap=r.get("metadata_overlap"),
            metadata_compact_match=r.get("metadata_compact_match"),
            metadata_phrase_match=r.get("metadata_phrase_match"),
            relevant_chunks=r["relevant_chunks"],
        )
        for r in grounded_results
    ]

    return ChatResponse(
        answer=answer,
        retrieved_files=retrieved_files,
        response_time_ms=response_time_ms,
    )


# =====================================================================
# SEARCH ENDPOINT (tanpa RAG)
# =====================================================================
@router.post("/search", response_model=SearchResponse)
def search(request: SearchRequest, db: Session = Depends(get_db)):
    """Pencarian semantik saja, tanpa generate jawaban."""
    start_time = time.time()

    searcher = get_searcher()
    rag = get_rag()
    query_plan = _plan_query(searcher, rag, request.query)
    retrieval_query = str(query_plan.get("retrieval_query") or request.query)
    is_lookup = bool(query_plan.get("is_lookup"))
    requested_top_k = int(request.top_k or settings.TOP_K_FILES or 5)
    candidate_top_k_files = max(requested_top_k * 3, 15) if is_lookup else requested_top_k

    results = searcher.search(retrieval_query, db, top_k_files=candidate_top_k_files)
    primary_tag = "rewritten" if retrieval_query.strip() != request.query.strip() else "raw"
    _tag_results_provenance(results, primary_tag)
    if retrieval_query.strip() != request.query.strip():
        raw_results = searcher.search(request.query, db, top_k_files=candidate_top_k_files)
        _tag_results_provenance(raw_results, "raw")
        results = _merge_search_results(results, raw_results)

    results = results[:requested_top_k]
    response_time_ms = int((time.time() - start_time) * 1000)

    retrieved_files = [
        FileInfo(
            file_id=r["file_id"],
            file_name=r["file_name"],
            file_path=r["file_path"],
            source=r.get("source", "local"),
            web_view_link=r.get("web_view_link"),
            file_type=r["file_type"],
            file_size=r["file_size"],
            last_modified=r["last_modified"],
            max_similarity=r["max_similarity"],
            metadata_score=r.get("metadata_score"),
            metadata_exact_hits=r.get("metadata_exact_hits"),
            metadata_partial_hits=r.get("metadata_partial_hits"),
            metadata_fuzzy_hits=r.get("metadata_fuzzy_hits"),
            metadata_name_ratio=r.get("metadata_name_ratio"),
            metadata_path_ratio=r.get("metadata_path_ratio"),
            metadata_text_ratio=r.get("metadata_text_ratio"),
            metadata_overlap=r.get("metadata_overlap"),
            metadata_compact_match=r.get("metadata_compact_match"),
            metadata_phrase_match=r.get("metadata_phrase_match"),
            relevant_chunks=r["relevant_chunks"],
        )
        for r in results
    ]

    return SearchResponse(
        results=retrieved_files,
        total_results=len(retrieved_files),
        response_time_ms=response_time_ms,
    )
# =====================================================================
# INDEXING ENDPOINTS (Background)
# =====================================================================
@router.post("/index", response_model=IndexResponse)
def index_files(request: IndexRequest, db: Session = Depends(get_db)):
    """Full indexing: mulai background task, hapus data lama → index ulang."""
    logger.info(f"Memulai full indexing untuk: {request.directories} (exclude: {request.exclude_dirs})")
    update_watch_config(
        db,
        local_directories=request.directories,
        exclude_directories=request.exclude_dirs or [],
    )
    indexer = get_indexer()
    started = indexer.start_background_index("full", request.directories, request.exclude_dirs)
    if started:
        mark_local_index_started(db)
    if not started:
        return IndexResponse(status="already_running", stats={})
    return IndexResponse(status="started", stats={})


@router.post("/index/incremental", response_model=IndexResponse)
def incremental_index(request: IndexRequest, db: Session = Depends(get_db)):
    """Incremental indexing: mulai background task untuk file baru/berubah."""
    logger.info(f"Memulai incremental indexing untuk: {request.directories} (exclude: {request.exclude_dirs})")
    update_watch_config(
        db,
        local_directories=request.directories,
        exclude_directories=request.exclude_dirs or [],
    )
    indexer = get_indexer()
    started = indexer.start_background_index("incremental", request.directories, request.exclude_dirs)
    if started:
        mark_local_index_started(db)
    if not started:
        return IndexResponse(status="already_running", stats={})
    return IndexResponse(status="started", stats={})


@router.post("/index/google-drive", response_model=IndexResponse)
def index_google_drive(request: GoogleDriveIndexRequest, db: Session = Depends(get_db)):
    """Mulai indexing Google Drive (full atau incremental) di background."""
    if not settings.SEARCH_INCLUDE_GDRIVE:
        raise HTTPException(
            status_code=403,
            detail="Indexing Google Drive sementara dinonaktifkan untuk testing baseline.",
        )

    if request.folder_id:
        update_watch_config(
            db,
            gdrive_folder_ids=[request.folder_id],
            gdrive_monitor_all=False,
        )

    indexer = get_indexer()
    mode = request.mode.lower().strip()
    if mode not in {"full", "incremental"}:
        raise HTTPException(status_code=400, detail="mode harus 'full' atau 'incremental'")

    started = indexer.start_background_google_index(mode=mode, folder_id=request.folder_id)
    if started:
        mark_gdrive_index_started(db)
    if not started:
        return IndexResponse(status="already_running", stats={})
    return IndexResponse(status="started", stats={})



@router.get("/index/watch-config", response_model=IndexWatchConfigResponse)
def get_index_watch_config(db: Session = Depends(get_db)):
    """Ambil konfigurasi watch indexing terakhir (persisten di backend)."""
    payload = get_watch_config_payload(db)
    return IndexWatchConfigResponse(**payload)


@router.post("/index/watch-config", response_model=IndexWatchConfigResponse)
def set_index_watch_config(request: IndexWatchConfigRequest, db: Session = Depends(get_db)):
    """Simpan konfigurasi watch indexing dari UI agar dipakai scheduler berkala."""
    payload = update_watch_config(
        db,
        local_directories=request.local_directories,
        exclude_directories=request.exclude_directories,
        gdrive_folder_ids=request.gdrive_folder_ids,
        gdrive_monitor_all=request.gdrive_monitor_all,
    )
    return IndexWatchConfigResponse(**payload)


@router.get("/index/status", response_model=IndexProgressResponse)
def index_status():
    """Polling progress indexing."""
    indexer = get_indexer()
    return IndexProgressResponse(**indexer.get_progress())


# =====================================================================
# GOOGLE OAUTH ENDPOINTS
# =====================================================================
@router.get("/auth/google/login", response_model=GoogleAuthUrlResponse)
def google_login_url():
    """Generate URL login Google OAuth untuk user."""
    gdrive = get_gdrive()
    try:
        auth_url = gdrive.get_authorization_url()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal membuat auth URL: {e}")
    return GoogleAuthUrlResponse(auth_url=auth_url)


@router.get("/auth/google/callback", response_class=HTMLResponse)
def google_callback(
    code: str = Query(...),
    error: str | None = Query(default=None),
    db: Session = Depends(get_db),
):
    """Callback OAuth dari Google setelah user approve consent."""
    if error:
        return HTMLResponse(
            content=f"<h3>OAuth dibatalkan atau gagal</h3><p>{error}</p>",
            status_code=400,
        )

    gdrive = get_gdrive()
    try:
        result = gdrive.exchange_code(code)

        auto_check_message = "Auto-check Google Drive menunggu jadwal scheduler berikutnya."
        if settings.AUTO_GDRIVE_INCREMENTAL_INDEX_ENABLED:
            watch_targets = get_effective_watch_targets(db)
            monitor_all = bool(watch_targets.get("gdrive_monitor_all"))
            folder_ids = watch_targets.get("gdrive_folder_ids") or []

            if monitor_all or folder_ids:
                indexer = get_indexer()
                started = False
                targets = [None] if monitor_all else folder_ids
                for folder_id in targets:
                    started = indexer.start_background_google_index(mode="incremental", folder_id=folder_id)
                    if started:
                        mark_gdrive_index_started(db)
                        break

                if started:
                    auto_check_message = "Auto-check Google Drive langsung dimulai setelah reconnect."
                else:
                    auto_check_message = "Auto-check Google Drive belum dimulai karena proses indexing lain sedang berjalan."
            else:
                auto_check_message = "Watch config Google Drive belum diset. Simpan konfigurasi watch dulu di UI."

        return HTMLResponse(
            content=(
                "<h3>Google Drive berhasil terhubung</h3>"
                "<p>Anda bisa kembali ke aplikasi Streamlit dan mulai indexing Google Drive.</p>"
                f"<p>{auto_check_message}</p>"
                f"<p>Token expiry: {result.get('expiry')}</p>"
            ),
            status_code=200,
        )
    except Exception as e:
        return HTMLResponse(
            content=f"<h3>Gagal memproses callback OAuth</h3><p>{e}</p>",
            status_code=500,
        )


@router.get("/auth/google/status", response_model=GoogleAuthStatusResponse)
def google_status():
    """Cek apakah token Google OAuth sudah tersedia dan valid."""
    gdrive = get_gdrive()
    connected = gdrive.is_connected()
    return GoogleAuthStatusResponse(
        connected=connected,
        token_file=gdrive.token_file if connected else None,
    )


# =====================================================================
# STATS ENDPOINT
# =====================================================================
@router.get("/stats", response_model=StatsResponse)
def get_stats(db: Session = Depends(get_db)):
    """Statistik jumlah file & chunk di database."""
    total_files = db.query(File).count()
    total_chunks = db.query(FileChunk).count()

    local_files = db.query(File).filter((File.source == "local") | (File.source.is_(None))).count()
    gdrive_files = db.query(File).filter(File.source == "gdrive").count()

    watch_config = get_watch_config_payload(db)
    local_directory_stats: dict[str, dict] = {}
    gdrive_directory_stats: dict[str, dict] = {}

    source_rows = (
        db.query(
            File.source,
            File.id,
            File.file_path,
            File.updated_at,
            func.count(FileChunk.id).label("chunk_count"),
        )
        .outerjoin(FileChunk, FileChunk.file_id == File.id)
        .group_by(File.id)
        .all()
    )

    for source, _file_id, file_path, updated_at, chunk_count in source_rows:
        if not file_path:
            continue

        normalized_source = (source or "local").strip().lower() or "local"
        if normalized_source == "local":
            directory = os.path.normpath(os.path.dirname(str(file_path)))
            target_stats = local_directory_stats
        elif normalized_source == "gdrive":
            directory = _virtual_directory_path(str(file_path), normalized_source)
            target_stats = gdrive_directory_stats
        else:
            continue

        if not directory:
            continue

        bucket = target_stats.get(directory)
        if bucket is None:
            bucket = {
                "directory_path": directory,
                "total_files": 0,
                "embedded_files": 0,
                "metadata_only_files": 0,
                "last_indexed": None,
            }
            target_stats[directory] = bucket

        bucket["total_files"] += 1
        if int(chunk_count or 0) > 0:
            bucket["embedded_files"] += 1
        else:
            bucket["metadata_only_files"] += 1

        if updated_at is not None:
            current_last = bucket.get("last_indexed")
            if current_last is None or updated_at > current_last:
                bucket["last_indexed"] = updated_at

    local_dirs = watch_config.get("local_directories") or []
    if local_dirs:
        seen_local_dirs = set()
        normalized_local_dirs = []
        for directory in local_dirs:
            normalized = os.path.normpath(str(directory).strip())
            if not normalized:
                continue
            dedupe_key = normalized.lower()
            if dedupe_key in seen_local_dirs:
                continue
            seen_local_dirs.add(dedupe_key)
            normalized_local_dirs.append(normalized)
        local_dirs = normalized_local_dirs
    else:
        local_paths = db.query(File.file_path).filter((File.source == "local") | (File.source.is_(None))).all()
        local_dirs = []
        seen_local_dirs = set()
        for (fp,) in local_paths:
            if not fp:
                continue
            directory = os.path.normpath(os.path.dirname(fp))
            dedupe_key = directory.lower()
            if dedupe_key in seen_local_dirs:
                continue
            seen_local_dirs.add(dedupe_key)
            local_dirs.append(directory)

    gdrive_paths = db.query(File.file_path).filter(File.source == "gdrive").all()
    gdrive_roots = sorted(list({(fp.split(":", 1)[0] + ":") for (fp,) in gdrive_paths if fp and ":" in fp}))
    gdrive_file_names = [
        name for (name,) in db.query(File.file_name).filter(File.source == "gdrive").order_by(File.updated_at.desc()).all()
    ]

    directories = local_dirs + gdrive_roots

    indexed_local_directory_details = [
        IndexedDirectoryInfo(
            directory_path=item["directory_path"],
            total_files=int(item["total_files"]),
            embedded_files=int(item["embedded_files"]),
            metadata_only_files=int(item["metadata_only_files"]),
            last_indexed=str(item["last_indexed"]) if item.get("last_indexed") else None,
        )
        for item in sorted(
            local_directory_stats.values(),
            key=lambda row: (-int(row.get("total_files", 0)), str(row.get("directory_path", "")).lower()),
        )
    ]

    indexed_gdrive_directory_details = [
        IndexedDirectoryInfo(
            directory_path=item["directory_path"],
            total_files=int(item["total_files"]),
            embedded_files=int(item["embedded_files"]),
            metadata_only_files=int(item["metadata_only_files"]),
            last_indexed=str(item["last_indexed"]) if item.get("last_indexed") else None,
        )
        for item in sorted(
            gdrive_directory_stats.values(),
            key=lambda row: (-int(row.get("total_files", 0)), str(row.get("directory_path", "")).lower()),
        )
    ]

    return StatsResponse(
        total_files=total_files,
        total_chunks=total_chunks,
        indexed_directories=directories,
        local_files=local_files,
        gdrive_files=gdrive_files,
        indexed_local_directories=local_dirs,
        indexed_gdrive_roots=gdrive_roots,
        indexed_gdrive_files=gdrive_file_names,
        indexed_gdrive_directory_details=indexed_gdrive_directory_details,
        indexed_local_directory_details=indexed_local_directory_details,
    )


@router.get("/index/files", response_model=IndexedFileListResponse)
def list_indexed_files(
    source: str | None = Query(default=None, description="Filter source: local, gdrive"),
    limit: int = Query(default=200, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    """Lihat daftar file yang sudah ter-index beserta metadata dan jumlah chunk."""
    query = (
        db.query(
            File,
            func.count(FileChunk.id).label("chunk_count"),
        )
        .outerjoin(FileChunk, FileChunk.file_id == File.id)
        .group_by(File.id)
        .order_by(File.updated_at.desc(), File.id.desc())
    )

    normalized_source = (source or "").strip().lower()
    if normalized_source:
        query = query.filter(File.source == normalized_source)

    total = query.count()
    rows = query.offset(offset).limit(limit).all()

    items = [
        IndexedFileItem(
            file_id=file.id,
            file_name=file.file_name,
            file_path=file.file_path,
            source=file.source or "local",
            cloud_file_id=file.cloud_file_id,
            web_view_link=file.web_view_link,
            file_type=file.file_type,
            file_size=file.file_size,
            last_modified=str(file.last_modified),
            content_hash=file.content_hash,
            chunk_count=int(chunk_count or 0),
            created_at=str(file.created_at) if file.created_at else None,
            updated_at=str(file.updated_at) if file.updated_at else None,
        )
        for file, chunk_count in rows
    ]

    return IndexedFileListResponse(total=total, items=items)


# =====================================================================
# CHAT HISTORY
# =====================================================================
@router.get("/history")
def get_chat_history(limit: int = 50, db: Session = Depends(get_db)):
    """Ambil riwayat chat terakhir."""
    history = (
        db.query(ChatHistory)
        .order_by(ChatHistory.created_at.desc())
        .limit(limit)
        .all()
    )

    return [
        {
            "id": h.id,
            "user_message": h.user_message,
            "bot_response": h.bot_response,
            "retrieved_files": h.retrieved_files,
            "response_time_ms": h.response_time_ms,
            "created_at": str(h.created_at),
        }
        for h in reversed(history)
    ]


# =====================================================================
# BACKEND DIAGNOSTICS — untuk verifikasi backend yang aktif
# =====================================================================
@router.get("/debug/backend-info")
def get_backend_info():
    """
    Endpoint diagnostik untuk memastikan backend yang aktif benar.
    Berisi: PID, executable path, CWD, config runtime, timestamp.
    """
    import os
    import platform
    from datetime import datetime
    
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "backend": {
            "pid": os.getpid(),
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "cwd": os.getcwd(),
            "platform": platform.system(),
        },
        "configuration": {
            "supported_extensions": settings.SUPPORTED_EXTENSIONS,
            "crawl_directories": settings.CRAWL_DIRECTORIES,
            "database_url": settings.DATABASE_URL.replace(settings.DATABASE_URL.split("@")[0].split("://")[1], "***"),
            "embedding_model": settings.SBERT_MODEL,
            "ollama_model": settings.OLLAMA_MODEL,
            "chunk_size": settings.CHUNK_SIZE,
            "chunk_overlap": settings.CHUNK_OVERLAP,
            "top_k_chunks": settings.TOP_K_CHUNKS,
            "top_k_files": settings.TOP_K_FILES,
        },
        "message": "Gunakan PID ini untuk verifikasi: kill PID jika perlu restart backend."
    }
