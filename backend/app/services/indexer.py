"""
Indexing Service — Orkestrasi proses crawling → ekstraksi → chunking → embedding → penyimpanan.

Dua mode:
  - full_index()         : Hapus data lama, index ulang semua file.
  - incremental_index()  : Hanya proses file baru/berubah, hapus file yang sudah dihapus.

Background indexing:
  - start_background_index() : Jalankan indexing di thread terpisah.
  - get_progress()           : Polling status progress.
"""

import hashlib
import logging
import threading
from typing import List, Dict, Optional, Callable

from sqlalchemy.orm import Session

from app.models.file_model import File, FileChunk
from app.services.crawler import FileCrawler
from app.services.extractor import TextExtractor
from app.services.embedder import EmbeddingService
from app.services.gdrive_service import GoogleDriveService
from app.database import SessionLocal
from app.config import settings
from app.services.watch_config import (
    mark_local_check_result,
    mark_gdrive_check_result,
)

logger = logging.getLogger(__name__)


class IndexingService:
    """Mengelola proses indexing file ke database (sinkron & background)."""

    def __init__(self):
        self.crawler = FileCrawler()
        self.extractor = TextExtractor()
        self.embedder = EmbeddingService()
        self.gdrive = GoogleDriveService()

        # Progress tracking
        self._progress: Dict = {
            "running": False,
            "mode": None,
            "phase": "idle",
            "total_files": 0,
            "processed_files": 0,
            "current_file": "",
            "stats": {},
            "error": None,
        }
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # PROGRESS
    # ------------------------------------------------------------------
    def get_progress(self) -> Dict:
        """Ambil status progress indexing saat ini."""
        with self._lock:
            return dict(self._progress)

    def _update_progress(self, **kwargs):
        with self._lock:
            self._progress.update(kwargs)

    def _set_phase(self, phase: str):
        if bool(settings.INDEX_GRANULAR_PROGRESS_ENABLED):
            self._update_progress(phase=phase)

    def _run_with_retries(
        self,
        operation: Callable[[], bool],
        *,
        file_label: str,
        stage_label: str,
    ) -> bool:
        attempts = max(1, int(settings.INDEX_FILE_RETRY_ATTEMPTS or 1))
        last_error: Optional[Exception] = None

        for attempt in range(1, attempts + 1):
            try:
                return operation()
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Gagal pada %s untuk %s (attempt %s/%s): %s",
                    stage_label,
                    file_label,
                    attempt,
                    attempts,
                    exc,
                )

        if last_error:
            raise last_error
        return False

    # ------------------------------------------------------------------
    # BACKGROUND INDEX
    # ------------------------------------------------------------------
    def start_background_index(self, mode: str, directories: List[str], exclude_dirs: List[str] = None) -> bool:
        """Mulai indexing di background thread. Return False jika sudah running."""
        with self._lock:
            if self._progress["running"]:
                return False
            self._progress.update({
                "running": True,
                "mode": mode,
                "phase": "starting",
                "total_files": 0,
                "processed_files": 0,
                "current_file": "",
                "stats": {},
                "error": None,
            })

        # Apply exclude_dirs to crawler
        if exclude_dirs:
            self.crawler.exclude_dirs = [d.lower() for d in exclude_dirs]

        thread = threading.Thread(
            target=self._run_index_background,
            args=(mode, directories),
            daemon=True,
        )
        thread.start()
        return True

    def _run_index_background(self, mode: str, directories: List[str]):
        """Thread target: jalankan indexing dengan session sendiri."""
        db: Session = SessionLocal()
        try:
            if mode == "full":
                self._full_index_internal(db, directories)
            else:
                self._incremental_index_internal(db, directories)
        except Exception as e:
            logger.error(f"Background indexing error: {e}")
            self._update_progress(phase="error", error=str(e), running=False)
        finally:
            db.close()

    def start_background_google_index(self, mode: str, folder_id: Optional[str] = None) -> bool:
        """Mulai indexing Google Drive di background thread."""
        with self._lock:
            if self._progress["running"]:
                return False
            self._progress.update({
                "running": True,
                "mode": f"google_{mode}",
                "phase": "starting",
                "total_files": 0,
                "processed_files": 0,
                "current_file": "",
                "stats": {},
                "error": None,
            })

        thread = threading.Thread(
            target=self._run_google_index_background,
            args=(mode, folder_id),
            daemon=True,
        )
        thread.start()
        return True

    def _run_google_index_background(self, mode: str, folder_id: Optional[str]):
        db: Session = SessionLocal()
        try:
            if mode == "full":
                self._google_full_index_internal(db, folder_id)
            else:
                self._google_incremental_index_internal(db, folder_id)
        except Exception as e:
            logger.error(f"Background Google indexing error: {e}")
            self._update_progress(phase="error", error=str(e), running=False)
        finally:
            db.close()

    # ------------------------------------------------------------------
    # FULL INDEX
    # ------------------------------------------------------------------
    def full_index(self, db: Session, directories: List[str] | None = None) -> Dict:
        """Index ulang semua file (sinkron wrapper)."""
        if directories:
            self.crawler.directories = directories
        return self._full_index_internal(db, directories)

    def _full_index_internal(self, db: Session, directories: Optional[List[str]] = None) -> Dict:
        """Internal full indexing dengan progress tracking."""
        if directories:
            self.crawler.directories = directories

        stats = {"crawled": 0, "indexed": 0, "skipped": 0, "errors": 0}

        # Hapus hanya data local agar data cloud tidak ikut terhapus
        self._update_progress(phase="cleaning")
        local_file_ids = [f.id for f in db.query(File.id).filter(File.source == "local").all()]
        if local_file_ids:
            db.query(FileChunk).filter(FileChunk.file_id.in_(local_file_ids)).delete(synchronize_session=False)
        db.query(File).filter(File.source == "local").delete(synchronize_session=False)
        db.commit()
        logger.info("Data lama dihapus. Memulai full indexing...")

        # Crawl
        self._update_progress(phase="crawling")
        crawled_files = self.crawler.crawl()
        stats["crawled"] = len(crawled_files)
        self._update_progress(phase="indexing", total_files=len(crawled_files), processed_files=0)

        # Proses setiap file
        for i, file_info in enumerate(crawled_files):
            self._update_progress(processed_files=i, current_file=file_info["file_name"])
            try:
                result = self._run_with_retries(
                    lambda: self._process_file(file_info, db),
                    file_label=file_info["file_name"],
                    stage_label="local_index",
                )
                if result:
                    stats["indexed"] += 1
                else:
                    stats["skipped"] += 1
            except Exception as e:
                db.rollback()
                logger.error(f"Error indexing {file_info['file_path']}: {e}")
                stats["errors"] += 1

            # Commit setiap 5 file agar progress tersimpan
            if (i + 1) % 5 == 0:
                db.commit()

        db.commit()
        self._update_progress(phase="done", processed_files=len(crawled_files), stats=stats, running=False)
        logger.info(f"Full indexing selesai: {stats}")
        return stats

    # ------------------------------------------------------------------
    # INCREMENTAL INDEX
    # ------------------------------------------------------------------
    def incremental_index(
        self, db: Session, directories: List[str] | None = None
    ) -> Dict:
        """Incremental index (sinkron wrapper)."""
        if directories:
            self.crawler.directories = directories
        return self._incremental_index_internal(db, directories)

    def _incremental_index_internal(
        self, db: Session, directories: Optional[List[str]] = None
    ) -> Dict:
        """Internal incremental indexing dengan progress tracking."""
        if directories:
            self.crawler.directories = directories

        stats = {
            "crawled": 0, "new": 0, "updated": 0,
            "deleted": 0, "unchanged": 0, "errors": 0,
        }

        # 1. Crawl
        self._update_progress(phase="crawling")
        crawled_files = self.crawler.crawl()
        stats["crawled"] = len(crawled_files)

        # 2. Ambil file yang sudah ada di DB — tapi batasi pada direktori yang sedang dicrawl
        all_local_files = db.query(File).filter(File.source == "local").all()
        target_dirs = []
        if directories:
            import os as _os
            target_dirs = [_os.path.abspath(d).replace("\\", "/").lower() for d in directories]

        if target_dirs:
            existing_files = {
                f.file_path: f
                for f in all_local_files
                if any(str(f.file_path).replace("\\", "/").lower().startswith(td) for td in target_dirs)
            }
        else:
            existing_files = {f.file_path: f for f in all_local_files}
        crawled_paths = set()

        self._update_progress(phase="indexing", total_files=len(crawled_files), processed_files=0)

        # 3. Proses setiap file yang di-crawl
        for i, file_info in enumerate(crawled_files):
            file_path = file_info["file_path"]
            crawled_paths.add(file_path)

            self._update_progress(processed_files=i, current_file=file_info["file_name"])

            try:
                existing = existing_files.get(file_path)

                if existing is None:
                    result = self._run_with_retries(
                        lambda: self._process_file(file_info, db),
                        file_label=file_info["file_name"],
                        stage_label="local_incremental_new",
                    )
                    stats["new"] += 1 if result else 0
                elif existing.last_modified < file_info["last_modified"]:
                    # Sebelum hapus + reindex, coba bandingkan content hash.
                    try:
                        content = self.extractor.extract(file_path, file_info["file_type"])
                        content_hash = hashlib.sha256(content.encode()).hexdigest() if content else None
                    except Exception:
                        content = None
                        content_hash = None

                    # Jika hash sama, file tidak berubah secara substantif — skip reindex.
                    if content_hash and existing.content_hash and content_hash == existing.content_hash:
                        # Perbarui last_modified agar tidak terus dianggap berubah.
                        existing.last_modified = file_info["last_modified"]
                        db.add(existing)
                        stats["unchanged"] += 1
                    else:
                        db.delete(existing)
                        # Remove from existing_files mapping to avoid later double-delete
                        try:
                            existing_files.pop(file_path, None)
                        except Exception:
                            pass
                        db.flush()
                        self._run_with_retries(
                            lambda: self._process_file(file_info, db),
                            file_label=file_info["file_name"],
                            stage_label="local_incremental_update",
                        )
                        stats["updated"] += 1
                else:
                    stats["unchanged"] += 1

            except Exception as e:
                db.rollback()
                logger.error(f"Error indexing {file_path}: {e}")
                stats["errors"] += 1

            if (i + 1) % 5 == 0:
                db.commit()

        # 4. Hapus file yang sudah tidak ada di disk — hanya untuk direktori target yang dicrawl
        if crawled_paths:
            for path, file_obj in list(existing_files.items()):
                if path not in crawled_paths:
                    db.delete(file_obj)
                    stats["deleted"] += 1
        else:
            # Jika crawling tidak menemukan file apa pun, jangan hapus apa-apa (safety guard).
            logger.warning("Crawling returned 0 files; skipping delete phase for local incremental indexing.")

        db.commit()
        mark_local_check_result(db, stats)
        self._update_progress(phase="done", processed_files=len(crawled_files), stats=stats, running=False)
        logger.info(f"Incremental indexing selesai: {stats}")
        return stats

    # ------------------------------------------------------------------
    # GOOGLE DRIVE INDEX
    # ------------------------------------------------------------------
    def _google_full_index_internal(self, db: Session, folder_id: Optional[str] = None) -> Dict:
        """Full indexing untuk Google Drive: reset source gdrive lalu index ulang."""
        stats = {"crawled": 0, "indexed": 0, "skipped": 0, "errors": 0}

        self._update_progress(phase="cleaning")
        gdrive_ids = [f.id for f in db.query(File.id).filter(File.source == "gdrive").all()]
        if gdrive_ids:
            db.query(FileChunk).filter(FileChunk.file_id.in_(gdrive_ids)).delete(synchronize_session=False)
        db.query(File).filter(File.source == "gdrive").delete(synchronize_session=False)
        db.commit()

        self._update_progress(phase="crawling")
        crawled_files = self.gdrive.list_supported_files(folder_id=folder_id)
        stats["crawled"] = len(crawled_files)
        self._update_progress(phase="indexing", total_files=len(crawled_files), processed_files=0)

        for i, file_info in enumerate(crawled_files):
            self._update_progress(processed_files=i, current_file=file_info["file_name"])
            try:
                def _index_gdrive_one() -> bool:
                    self._set_phase("downloading")
                    file_bytes = self.gdrive.download_file_bytes(
                        file_info["file_id"],
                        file_info.get("source_mime_type"),
                    )
                    return self._process_cloud_file(file_info, file_bytes, db)

                result = self._run_with_retries(
                    _index_gdrive_one,
                    file_label=file_info["file_name"],
                    stage_label="gdrive_full",
                )
                if result:
                    stats["indexed"] += 1
                else:
                    stats["skipped"] += 1
            except Exception as e:
                db.rollback()
                logger.error(f"Error indexing Google Drive file {file_info.get('file_name')}: {e}")
                stats["errors"] += 1

            if (i + 1) % 5 == 0:
                db.commit()

        db.commit()
        self._update_progress(phase="done", processed_files=len(crawled_files), stats=stats, running=False)
        logger.info(f"Google full indexing selesai: {stats}")
        return stats

    def _google_incremental_index_internal(self, db: Session, folder_id: Optional[str] = None) -> Dict:
        """Incremental indexing untuk Google Drive berdasarkan modified time."""
        stats = {
            "crawled": 0, "new": 0, "updated": 0,
            "deleted": 0, "unchanged": 0, "errors": 0,
        }

        self._update_progress(phase="crawling")
        crawled_files = self.gdrive.list_supported_files(folder_id=folder_id)
        stats["crawled"] = len(crawled_files)

        existing_files = {
            (f.cloud_file_id or ""): f
            for f in db.query(File).filter(File.source == "gdrive").all()
        }
        crawled_ids = set()

        self._update_progress(phase="indexing", total_files=len(crawled_files), processed_files=0)

        for i, file_info in enumerate(crawled_files):
            cloud_id = file_info["file_id"]
            crawled_ids.add(cloud_id)
            self._update_progress(processed_files=i, current_file=file_info["file_name"])

            try:
                existing = existing_files.get(cloud_id)

                if existing is None:
                    def _index_new_cloud() -> bool:
                        self._set_phase("downloading")
                        file_bytes = self.gdrive.download_file_bytes(
                            cloud_id,
                            file_info.get("source_mime_type"),
                        )
                        return self._process_cloud_file(file_info, file_bytes, db)

                    result = self._run_with_retries(
                        _index_new_cloud,
                        file_label=file_info["file_name"],
                        stage_label="gdrive_incremental_new",
                    )
                    stats["new"] += 1 if result else 0
                elif existing.last_modified < file_info["last_modified"]:
                    # Download dulu dan bandingkan hash konten dengan yang tersimpan.
                    try:
                        self._set_phase("downloading")
                        file_bytes = self.gdrive.download_file_bytes(
                            cloud_id,
                            file_info.get("source_mime_type"),
                        )
                        content = self.extractor.extract_from_bytes(file_bytes, file_info.get("file_type"))
                        content_hash = hashlib.sha256(content.encode()).hexdigest() if content else None
                    except Exception as e:
                        logger.warning(f"Gagal download/ekstrak untuk perbandingan hash: {e}")
                        file_bytes = None
                        content_hash = None

                    if content_hash and existing.content_hash and content_hash == existing.content_hash:
                        # Konten sama — tidak perlu reindex. Perbarui last_modified.
                        existing.last_modified = file_info["last_modified"]
                        db.add(existing)
                        stats["unchanged"] += 1
                    else:
                        # Konten berubah (atau gagal periksa) — hapus dan index ulang.
                        db.delete(existing)
                        # Remove from existing_files mapping to avoid later double-delete
                        try:
                            existing_files.pop(cloud_id, None)
                        except Exception:
                            pass
                        db.flush()
                        def _index_updated_cloud() -> bool:
                            if file_bytes is None:
                                self._set_phase("downloading")
                                fb = self.gdrive.download_file_bytes(
                                    cloud_id,
                                    file_info.get("source_mime_type"),
                                )
                                return self._process_cloud_file(file_info, fb, db)
                            else:
                                return self._process_cloud_file(file_info, file_bytes, db)

                        self._run_with_retries(
                            _index_updated_cloud,
                            file_label=file_info["file_name"],
                            stage_label="gdrive_incremental_update",
                        )
                        stats["updated"] += 1
                else:
                    stats["unchanged"] += 1

            except Exception as e:
                db.rollback()
                logger.error(f"Error indexing Google Drive file {file_info.get('file_name')}: {e}")
                stats["errors"] += 1

            if (i + 1) % 5 == 0:
                db.commit()

        if not crawled_ids and existing_files:
            logger.warning(
                "Google Drive crawling mengembalikan 0 file padahal DB punya %d entri gdrive. "
                "Melewati delete phase sebagai safety guard.",
                len(existing_files),
            )
        else:
            for cloud_id, file_obj in existing_files.items():
                if cloud_id and cloud_id not in crawled_ids:
                    db.delete(file_obj)
                    stats["deleted"] += 1

        db.commit()
        mark_gdrive_check_result(db, stats)
        self._update_progress(phase="done", processed_files=len(crawled_files), stats=stats, running=False)
        logger.info(f"Google incremental indexing selesai: {stats}")
        return stats

    # ------------------------------------------------------------------
    # PRIVATE
    # ------------------------------------------------------------------
    def _process_file(self, file_info: Dict, db: Session) -> bool:
        """
        Proses satu file: extract → chunk → embed → simpan.
        Returns True jika berhasil di-index, False jika dilewati.
        """
        file_path = file_info["file_path"]
        file_type = file_info["file_type"]

        # 1. Ekstraksi teks
        self._set_phase("extracting")
        content = self.extractor.extract(file_path, file_type)

        # 2. Hitung hash konten
        content_hash = (
            hashlib.sha256(content.encode()).hexdigest() if content else None
        )

        # 3. Simpan metadata file
        file_record = File(
            file_name=file_info["file_name"],
            file_path=file_info["file_path"],
            source="local",
            file_type=file_info["file_type"],
            file_size=file_info["file_size"],
            last_modified=file_info["last_modified"],
            content_text=content,
            content_hash=content_hash,
        )
        db.add(file_record)
        db.flush()  # dapatkan ID

        # 4. Chunk + embed (jika ada konten teks)
        if content:
            self._set_phase("chunking")
            chunks = self.embedder.chunk_text(content)

            if chunks:
                self._set_phase("embedding")
                self._save_chunks_with_embeddings(file_record.id, chunks, db)

            logger.info(
                f"✓ Indexed: {file_path} "
                f"({len(chunks)} chunks)"
            )
            return True
        else:
            logger.warning(f"✗ Tidak ada teks: {file_path}")
            return False

    def _process_cloud_file(self, file_info: Dict, file_bytes: bytes, db: Session) -> bool:
        """Proses satu file cloud: extract bytes → chunk → embed → simpan."""
        file_path = file_info["file_path"]
        file_type = file_info["file_type"]

        self._set_phase("extracting")
        content = self.extractor.extract_from_bytes(file_bytes, file_type)
        content_hash = hashlib.sha256(content.encode()).hexdigest() if content else None

        file_record = File(
            file_name=file_info["file_name"],
            file_path=file_path,
            source=file_info.get("source", "gdrive"),
            cloud_file_id=file_info.get("file_id"),
            web_view_link=file_info.get("web_view_link"),
            file_type=file_type,
            file_size=file_info["file_size"],
            last_modified=file_info["last_modified"],
            content_text=content,
            content_hash=content_hash,
        )
        db.add(file_record)
        db.flush()

        if content:
            self._set_phase("chunking")
            chunks = self.embedder.chunk_text(content)
            if chunks:
                self._set_phase("embedding")
                self._save_chunks_with_embeddings(file_record.id, chunks, db)

            logger.info(f"✓ Indexed cloud file: {file_info['file_name']} ({len(chunks)} chunks)")
            return True

        logger.warning(f"✗ Tidak ada teks (cloud): {file_info['file_name']}")
        return False

    def _save_chunks_with_embeddings(self, file_id: int, chunks: List[str], db: Session):
        """Simpan chunk + embedding secara bertahap agar aman untuk dokumen besar."""
        batch_size = max(1, int(settings.EMBEDDING_BATCH_SIZE or 16))

        for start in range(0, len(chunks), batch_size):
            chunk_batch = chunks[start:start + batch_size]
            embeddings = self.embedder.embed_texts(chunk_batch, batch_size=batch_size)

            for offset, (chunk_text, embedding) in enumerate(zip(chunk_batch, embeddings)):
                db.add(
                    FileChunk(
                        file_id=file_id,
                        chunk_index=start + offset,
                        chunk_text=chunk_text,
                        embedding=embedding,
                    )
                )

            # Flush per batch agar penggunaan memori lebih terkontrol.
            db.flush()
