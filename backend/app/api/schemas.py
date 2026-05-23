"""
Pydantic Schemas — Request & Response models untuk FastAPI endpoints.
"""

from pydantic import BaseModel
from typing import List, Optional


# =====================================================================
# REQUEST SCHEMAS
# =====================================================================

class ChatRequest(BaseModel):
    """Request body untuk endpoint /chat."""
    message: str


class IndexRequest(BaseModel):
    """Request body untuk endpoint /index."""
    directories: List[str]
    exclude_dirs: Optional[List[str]] = []


class GoogleDriveIndexRequest(BaseModel):
    """Request body untuk endpoint indexing Google Drive."""
    folder_id: Optional[str] = None
    mode: str = "full"  # full | incremental


class IndexWatchConfigRequest(BaseModel):
    """Request body untuk simpan konfigurasi watch indexing."""
    local_directories: Optional[List[str]] = None
    exclude_directories: Optional[List[str]] = None
    gdrive_folder_ids: Optional[List[str]] = None
    gdrive_monitor_all: Optional[bool] = None


class SearchRequest(BaseModel):
    """Request body untuk endpoint /search."""
    query: str
    top_k: Optional[int] = 5


class DebugLookupRequest(BaseModel):
    """Request body untuk endpoint debug lookup."""
    query: str
    top_k: Optional[int] = 5


# =====================================================================
# RESPONSE SCHEMAS
# =====================================================================

class ChunkInfo(BaseModel):
    """Informasi satu chunk yang relevan."""
    chunk_id: int
    chunk_index: int
    chunk_text: str
    similarity: float


class FileInfo(BaseModel):
    """Informasi satu file hasil pencarian."""
    file_id: int
    file_name: str
    file_path: str
    source: str = "local"
    web_view_link: Optional[str] = None
    file_type: str
    file_size: int
    last_modified: str
    max_similarity: float
    metadata_score: Optional[float] = None
    metadata_exact_hits: Optional[float] = None
    metadata_partial_hits: Optional[float] = None
    metadata_fuzzy_hits: Optional[float] = None
    metadata_name_ratio: Optional[float] = None
    metadata_path_ratio: Optional[float] = None
    metadata_text_ratio: Optional[float] = None
    metadata_overlap: Optional[float] = None
    metadata_compact_match: Optional[float] = None
    metadata_phrase_match: Optional[float] = None
    relevant_chunks: List[dict] = []


class ChatResponse(BaseModel):
    """Response dari endpoint /chat."""
    answer: str
    retrieved_files: List[FileInfo]
    response_time_ms: int


class SearchResponse(BaseModel):
    """Response dari endpoint /search."""
    results: List[FileInfo]
    total_results: int
    response_time_ms: int


class DebugLookupResponse(BaseModel):
    """Response debug untuk melihat hasil retrieval mentah dan hasil planner."""
    query: str
    retrieval_query: str
    is_lookup: bool
    planner_intent: Optional[str] = None
    planner_keywords: List[str] = []
    raw_results: List[FileInfo] = []
    planned_results: List[FileInfo] = []
    merged_results: List[FileInfo] = []
    response_time_ms: int = 0


class IndexResponse(BaseModel):
    """Response dari endpoint /index."""
    status: str
    stats: dict


class IndexedFileItem(BaseModel):
    """Detail metadata untuk satu file yang sudah ter-index."""
    file_id: int
    file_name: str
    file_path: str
    source: str = "local"
    cloud_file_id: Optional[str] = None
    web_view_link: Optional[str] = None
    file_type: str
    file_size: int
    last_modified: str
    content_hash: Optional[str] = None
    chunk_count: int = 0
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class IndexedFileListResponse(BaseModel):
    """Response berisi daftar file yang sudah ter-index."""
    total: int
    items: List[IndexedFileItem]


class IndexProgressResponse(BaseModel):
    """Response dari endpoint /index/status — polling progress."""
    running: bool
    mode: Optional[str] = None
    phase: str = "idle"
    total_files: int = 0
    processed_files: int = 0
    current_file: str = ""
    stats: dict = {}
    error: Optional[str] = None


class IndexWatchConfigResponse(BaseModel):
    """Response konfigurasi watch indexing yang tersimpan di backend."""
    local_directories: List[str] = []
    exclude_directories: List[str] = []
    gdrive_folder_ids: List[str] = []
    gdrive_monitor_all: bool = False
    updated_at: Optional[str] = None
    last_local_index_started_at: Optional[str] = None
    last_gdrive_index_started_at: Optional[str] = None
    last_local_check_at: Optional[str] = None
    last_local_check_has_changes: Optional[bool] = None
    last_local_check_stats: dict = {}
    last_gdrive_check_at: Optional[str] = None
    last_gdrive_check_has_changes: Optional[bool] = None
    last_gdrive_check_stats: dict = {}


class IndexedDirectoryInfo(BaseModel):
    """Ringkasan status index per direktori local."""
    directory_path: str
    total_files: int = 0
    embedded_files: int = 0
    metadata_only_files: int = 0
    last_indexed: Optional[str] = None


class StatsResponse(BaseModel):
    """Response dari endpoint /stats."""
    total_files: int
    total_chunks: int
    indexed_directories: List[str]
    local_files: int = 0
    gdrive_files: int = 0
    indexed_local_directories: List[str] = []
    indexed_gdrive_roots: List[str] = []
    indexed_gdrive_files: List[str] = []
    indexed_gdrive_directory_details: List[IndexedDirectoryInfo] = []
    indexed_local_directory_details: List[IndexedDirectoryInfo] = []


class GoogleAuthUrlResponse(BaseModel):
    """Response untuk endpoint URL login Google OAuth."""
    auth_url: str


class GoogleAuthStatusResponse(BaseModel):
    """Response untuk status koneksi Google OAuth."""
    connected: bool
    token_file: Optional[str] = None


