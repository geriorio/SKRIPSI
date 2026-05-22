"""
SQLAlchemy models untuk sistem chatbot pencarian file.

Tabel:
  - files        : metadata setiap file yang telah di-crawl
  - file_chunks  : potongan teks + embedding vector dari setiap file
  - chat_history : riwayat percakapan pengguna dengan chatbot
"""

from sqlalchemy import (
    Column, Integer, String, Text, BigInteger, Boolean,
    DateTime, ForeignKey
)
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
from datetime import datetime

from app.database import Base
from app.config import settings


class File(Base):
    """Menyimpan metadata file yang telah di-index."""
    __tablename__ = "files"

    id = Column(Integer, primary_key=True, index=True)
    file_name = Column(String(500), nullable=False)
    file_path = Column(String(1000), unique=True, nullable=False)
    source = Column(String(50), nullable=False, default="local", index=True)  # local | gdrive | onedrive
    cloud_file_id = Column(String(255), nullable=True, index=True)
    web_view_link = Column(String(1200), nullable=True)
    file_type = Column(String(20), nullable=False)
    file_size = Column(BigInteger, nullable=False)              # dalam byte
    last_modified = Column(DateTime, nullable=False)
    content_text = Column(Text, nullable=True)                  # teks hasil ekstraksi
    content_hash = Column(String(64), nullable=True)            # SHA-256 untuk deteksi perubahan
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relasi ke chunks
    chunks = relationship(
        "FileChunk", back_populates="file", cascade="all, delete-orphan"
    )


class FileChunk(Base):
    """Menyimpan potongan teks (chunk) beserta embedding vector."""
    __tablename__ = "file_chunks"

    id = Column(Integer, primary_key=True, index=True)
    file_id = Column(
        Integer,
        ForeignKey("files.id", ondelete="CASCADE"),
        nullable=False, index=True
    )
    chunk_index = Column(Integer, nullable=False)               # urutan chunk dalam file
    chunk_text = Column(Text, nullable=False)
    embedding = Column(Vector(settings.EMBEDDING_DIMENSION))    # vector(512)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relasi ke file
    file = relationship("File", back_populates="chunks")


class ChatHistory(Base):
    """Menyimpan riwayat percakapan untuk logging dan evaluasi."""
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, index=True)
    user_message = Column(Text, nullable=False)
    bot_response = Column(Text, nullable=False)
    retrieved_files = Column(Text, nullable=True)               # JSON string
    response_time_ms = Column(Integer, nullable=True)           # waktu respon (ms)
    created_at = Column(DateTime, default=datetime.utcnow)


class IndexWatchConfig(Base):
    """Konfigurasi direktori/folder yang dipantau scheduler auto-index."""
    __tablename__ = "index_watch_config"

    id = Column(Integer, primary_key=True, default=1)

    # Local watch settings
    local_directories = Column(Text, nullable=True)             # JSON array string
    exclude_directories = Column(Text, nullable=True)           # JSON array string

    # Google Drive watch settings
    gdrive_folder_ids = Column(Text, nullable=True)             # JSON array string
    gdrive_monitor_all = Column(Boolean, nullable=False, default=False)

    # OneDrive watch settings
    onedrive_folder_ids = Column(Text, nullable=True)           # JSON array string
    onedrive_monitor_all = Column(Boolean, nullable=False, default=False)

    # Audit timestamps
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_local_index_started_at = Column(DateTime, nullable=True)
    last_gdrive_index_started_at = Column(DateTime, nullable=True)
    last_onedrive_index_started_at = Column(DateTime, nullable=True)
    last_local_check_at = Column(DateTime, nullable=True)
    last_local_check_has_changes = Column(Boolean, nullable=True)
    last_local_check_stats = Column(Text, nullable=True)        # JSON string
    last_gdrive_check_at = Column(DateTime, nullable=True)
    last_gdrive_check_has_changes = Column(Boolean, nullable=True)
    last_gdrive_check_stats = Column(Text, nullable=True)       # JSON string
    last_onedrive_check_at = Column(DateTime, nullable=True)
    last_onedrive_check_has_changes = Column(Boolean, nullable=True)
    last_onedrive_check_stats = Column(Text, nullable=True)     # JSON string
