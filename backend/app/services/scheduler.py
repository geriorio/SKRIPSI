"""
Scheduler ringan untuk menjalankan incremental indexing secara berkala.

Scheduler ini memakai singleton IndexingService yang sama dengan endpoint API,
sehingga indexing manual dan otomatis tidak berjalan bersamaan.
"""

import logging
import threading
from typing import Optional

from app.config import settings
from app.api.routes import get_indexer
from app.database import SessionLocal
from app.services.watch_config import (
    get_effective_watch_targets,
    mark_local_index_started,
    mark_gdrive_index_started,
)

logger = logging.getLogger(__name__)


class AutoIncrementalIndexScheduler:
    """Menjalankan incremental indexing otomatis dalam background thread."""

    def __init__(self):
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

    def start(self) -> bool:
        """Mulai scheduler. Return False jika fitur dimatikan atau sudah berjalan."""
        if not settings.AUTO_INCREMENTAL_INDEX_ENABLED:
            logger.info("Auto incremental indexing dinonaktifkan.")
            return False

        with self._lock:
            if self._thread and self._thread.is_alive():
                return False

            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
            return True

    def stop(self):
        """Hentikan scheduler."""
        self._stop_event.set()

    def _run(self):
        interval_minutes = max(1, int(settings.AUTO_INCREMENTAL_INDEX_INTERVAL_MINUTES or 60))
        interval_seconds = interval_minutes * 60
        logger.info(
            "Auto incremental indexing aktif: cek setiap %s menit.",
            interval_minutes,
        )

        # Jalankan sekali 30 detik setelah startup, agar tidak perlu tunggu interval penuh.
        if not self._stop_event.wait(30):
            try:
                self.run_once()
            except Exception as exc:
                logger.error("Auto indexing scheduler error (startup run): %s", exc)

        while not self._stop_event.wait(interval_seconds):
            try:
                self.run_once()
            except Exception as exc:
                logger.error("Auto indexing scheduler error: %s", exc)

    def run_once(self) -> bool:
        """Jalankan satu siklus incremental indexing untuk local dan Google Drive."""
        ran_anything = False

        if self.run_local_once():
            ran_anything = True

        if self.run_google_drive_once():
            ran_anything = True

        return ran_anything

    def run_local_once(self) -> bool:
        """Jalankan incremental indexing lokal bila ada direktori yang dipantau."""
        db = SessionLocal()
        try:
            watch_targets = get_effective_watch_targets(db)
        finally:
            db.close()

        directories = watch_targets.get("local_directories") or settings.CRAWL_DIRECTORIES
        exclude_dirs = watch_targets.get("exclude_directories") or settings.EXCLUDE_DIRECTORIES
        if not directories:
            logger.info("Auto incremental indexing dilewati karena CRAWL_DIRECTORIES masih kosong.")
            return False

        indexer = get_indexer()
        progress = indexer.get_progress()
        if progress.get("running"):
            logger.info("Auto incremental indexing dilewati karena proses indexing lain masih berjalan.")
            return False

        started = indexer.start_background_index(
            mode="incremental",
            directories=directories,
            exclude_dirs=exclude_dirs or None,
        )

        if started:
            logger.info("Auto incremental indexing dimulai untuk direktori: %s", directories)
            db = SessionLocal()
            try:
                mark_local_index_started(db)
            finally:
                db.close()
        else:
            logger.info("Auto incremental indexing tidak dimulai karena proses lain masih aktif.")

        return started

    def run_google_drive_once(self) -> bool:
        """Jalankan incremental indexing Google Drive bila konfigurasi folder tersedia."""
        if not settings.AUTO_GDRIVE_INCREMENTAL_INDEX_ENABLED:
            logger.info("Auto incremental indexing Google Drive dinonaktifkan.")
            return False

        db = SessionLocal()
        try:
            watch_targets = get_effective_watch_targets(db)
        finally:
            db.close()

        monitor_all = bool(watch_targets.get("gdrive_monitor_all"))
        folder_ids = watch_targets.get("gdrive_folder_ids") or settings.AUTO_GDRIVE_FOLDER_IDS or []

        if not monitor_all and not folder_ids:
            logger.info("Auto incremental indexing Google Drive dilewati karena folder watch belum diset.")
            return False

        indexer = get_indexer()
        if not indexer.gdrive.is_connected():
            logger.info("Auto incremental indexing Google Drive dilewati karena OAuth belum terhubung.")
            return False

        progress = indexer.get_progress()
        if progress.get("running"):
            logger.info("Auto incremental indexing Google Drive dilewati karena proses indexing lain masih berjalan.")
            return False

        started_any = False
        targets = [None] if monitor_all else folder_ids
        for folder_id in targets:
            started = indexer.start_background_google_index(mode="incremental", folder_id=folder_id)
            if started:
                logger.info("Auto incremental indexing Google Drive dimulai untuk folder_id=%s", folder_id)
                started_any = True
                db = SessionLocal()
                try:
                    mark_gdrive_index_started(db)
                finally:
                    db.close()
                break
            logger.info("Auto incremental indexing Google Drive tidak dimulai karena proses lain masih aktif.")

        return started_any


auto_index_scheduler = AutoIncrementalIndexScheduler()