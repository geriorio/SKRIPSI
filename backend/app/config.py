import os
import json
from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    """Konfigurasi utama sistem chatbot pencarian file."""

    # --- Database PostgreSQL ---
    DATABASE_URL: str = "postgresql://skripsi_user:skripsi_password@localhost:5432/skripsi_chatbot"

    # --- Model SBERT ---
    SBERT_MODEL: str = "distiluse-base-multilingual-cased-v2"
    EMBEDDING_DIMENSION: int = 512  # dimensi output distiluse-base-multilingual-cased-v2

    # --- Ollama LLM (untuk RAG) ---
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen2.5:7b"

    # --- Google Drive OAuth ---
    GOOGLE_OAUTH_CLIENT_FILE: str = ""
    GOOGLE_OAUTH_TOKEN_FILE: str = ""
    GOOGLE_OAUTH_REDIRECT_URI: str = "http://localhost:8001/api/auth/google/callback"
    GOOGLE_OAUTH_SCOPES: List[str] = [
        "https://www.googleapis.com/auth/drive.readonly"
    ]

    # --- Crawler ---
    CRAWL_DIRECTORIES: List[str] = []
    EXCLUDE_DIRECTORIES: List[str] = []
    AUTO_INCREMENTAL_INDEX_ENABLED: bool = True
    AUTO_INCREMENTAL_INDEX_INTERVAL_MINUTES: int = 60
    AUTO_GDRIVE_INCREMENTAL_INDEX_ENABLED: bool = False
    AUTO_GDRIVE_FOLDER_IDS: List[str] = []
    SUPPORTED_EXTENSIONS: List[str] = [
        ".pdf", ".docx", ".xlsx", ".csv", ".ppt", ".pptx", ".txt"
    ]
    MAX_TXT_FILE_SIZE: int = 512_000  # 500KB — .txt lebih besar dari ini biasanya bukan dokumen kerja

    # Folder yang otomatis di-skip (non-dokumen)
    AUTO_EXCLUDE_DIRS: List[str] = [
        "node_modules", ".git", "__pycache__", "venv", ".venv",
        ".idea", ".vscode", "dist", "build", "bin", "obj",
        "packages", "vendor", ".svn", ".hg",
        "dotnet", "runtime", "sdk",
        "$recycle.bin", "recycle.bin",  # Windows Recycle Bin
    ]

    # Nama file yang otomatis di-skip (bukan dokumen kerja)
    SKIP_FILE_PATTERNS: List[str] = [
        "license", "licence", "notice", "thirdpartynotices",
        "changelog", "readme", "authors", "contributors",
        "copying", "patents", "credits",
    ]

    # Hint path opsional dari user (default kosong; tidak hardcode vendor/folder tertentu).
    INDEX_EXCLUDE_PATH_HINTS: List[str] = []

    # Deteksi otomatis folder "noise" saat indexing (tanpa hardcode path spesifik).
    INDEX_AUTO_NOISE_FILTER_ENABLED: bool = False
    INDEX_AUTO_NOISE_SAMPLE_LIMIT: int = 200
    INDEX_AUTO_NOISE_MAX_SCANNED_DIRS: int = 80
    INDEX_AUTO_NOISE_MIN_SAMPLED_FILES: int = 24
    INDEX_AUTO_NOISE_MIN_SUPPORTED_FILES: int = 1
    INDEX_AUTO_NOISE_MAX_SUPPORTED_RATIO: float = 0.08
    INDEX_AUTO_NOISE_MIN_TECH_RATIO: float = 0.75
    INDEX_AUTO_NOISE_EXTENSION_HINTS: List[str] = [
        ".dll", ".exe", ".so", ".dylib", ".jar", ".aar", ".class", ".o", ".obj",
        ".a", ".lib", ".pdb", ".pyc", ".pyo", ".whl", ".egg", ".lock", ".tmp",
        ".cache", ".bin", ".dat", ".pak", ".rmeta", ".rlib", ".wasm", ".tsbuildinfo",
    ]

    # --- Chunking ---
    CHUNK_SIZE: int = 500       # jumlah karakter per chunk
    CHUNK_OVERLAP: int = 50     # overlap antar chunk
    EMBEDDING_BATCH_SIZE: int = 8  # default lebih aman untuk file besar (tetap bisa embed, lebih stabil)
    INDEX_FILE_RETRY_ATTEMPTS: int = 2
    INDEX_GRANULAR_PROGRESS_ENABLED: bool = True

    # --- Search ---
    TOP_K_CHUNKS: int = 20      # jumlah chunk teratas yang diambil
    TOP_K_FILES: int = 5        # jumlah file teratas yang dikembalikan
    SEARCH_INCLUDE_GDRIVE: bool = True  # libatkan dokumen Google Drive saat retrieval
    SEARCH_FILE_LOOKUP_TOP_K_CHUNKS: int = 40
    SEARCH_FILE_LOOKUP_METADATA_BOOST: float = 0.35
    SEARCH_FILE_LOOKUP_EXACT_PHRASE_BOOST: float = 0.20
    SEARCH_FILE_LOOKUP_RARE_TOKEN_BOOST: float = 0.10
    SEARCH_FILE_LOOKUP_MIN_TOKEN_RATIO: float = 0.50
    SEARCH_FILE_LOOKUP_LOW_MATCH_PENALTY: float = 0.15
    SEARCH_FILE_LOOKUP_GDRIVE_PENALTY: float = 0.04
    SEARCH_EXCLUDE_PATH_KEYWORDS: List[str] = [
        "software", "licenses", "common7", "msbuild", "vc", "dotnet", "runtime"
    ]

    # --- Cloud indexing ---
    CLOUD_TEMP_DIR: str = ""

    class Config:
        env_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
        extra = "ignore"


settings = Settings()
