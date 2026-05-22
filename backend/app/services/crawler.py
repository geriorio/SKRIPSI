"""
File Crawler — Modul penelusuran otomatis (traversal) terhadap folder & subfolder.

Proses:
  1. Menerima daftar direktori target.
  2. Melakukan os.walk (rekursif) ke setiap subfolder terdalam.
  3. Mengumpulkan metadata: nama, path, tipe, ukuran, tanggal modifikasi.

Smart filtering (otomatis):
  - Auto-exclude folder umum (node_modules, .git, venv, dotnet, runtime, dll)
  - Skip file non-dokumen berdasarkan nama (LICENSE, NOTICE, ThirdPartyNotices, dll)
"""

import os
import logging
from datetime import datetime
from typing import List, Dict

from app.config import settings

logger = logging.getLogger(__name__)


class FileCrawler:
    """Melakukan traversal rekursif untuk menemukan file yang didukung."""

    def __init__(
        self,
        directories: List[str] | None = None,
        extensions: List[str] | None = None,
        exclude_dirs: List[str] | None = None,
    ):
        self.directories = directories or settings.CRAWL_DIRECTORIES
        self.extensions = [e.lower() for e in (extensions or settings.SUPPORTED_EXTENSIONS)]

        # Gabungkan user exclude + auto exclude
        user_exclude = [d.lower() for d in (exclude_dirs or settings.EXCLUDE_DIRECTORIES)]
        auto_exclude = [d.lower() for d in settings.AUTO_EXCLUDE_DIRS]
        self.exclude_dirs = list(set(user_exclude + auto_exclude))

        # Pattern nama file yang di-skip
        self.skip_patterns = [p.lower() for p in settings.SKIP_FILE_PATTERNS]
        self.exclude_path_hints = [
            h.lower().replace("\\", "/").strip("/")
            for h in settings.INDEX_EXCLUDE_PATH_HINTS
            if h and h.strip()
        ]
    def _normalize_path(self, path: str) -> str:
        return (path or "").lower().replace("\\", "/")

    def _is_excluded_by_hint(self, path: str) -> bool:
        normalized = self._normalize_path(path)
        return any(hint in normalized for hint in self.exclude_path_hints)

    # ------------------------------------------------------------------
    # PUBLIC
    # ------------------------------------------------------------------
    def crawl(self) -> List[Dict]:
        """Crawl semua direktori dan kembalikan list metadata file."""
        all_files: List[Dict] = []

        for directory in self.directories:
            if not os.path.exists(directory):
                logger.warning(f"Direktori tidak ditemukan: {directory}")
                continue

            if self._is_excluded_by_hint(directory):
                logger.info(f"Direktori di-skip karena match INDEX_EXCLUDE_PATH_HINTS: {directory}")
                continue

            logger.info(f"Mulai crawling: {directory}")
            found = self._traverse(directory)
            all_files.extend(found)
            logger.info(f"  → Ditemukan {len(found)} file di {directory}")

        logger.info(f"Total file ditemukan: {len(all_files)}")
        return all_files

    # ------------------------------------------------------------------
    # PRIVATE
    # ------------------------------------------------------------------
    def _traverse(self, directory: str) -> List[Dict]:
        """Rekursif traverse menggunakan os.walk."""
        files_found: List[Dict] = []

        try:
            for root, _dirs, files in os.walk(directory):
                # Skip excluded directories
                exclude_dirs_set = set(self.exclude_dirs)
                filtered_dirs = []
                for d in _dirs:
                    subdir_path = os.path.join(root, d)
                    if d.lower() in exclude_dirs_set:
                        continue
                    if subdir_path.lower() in exclude_dirs_set:
                        continue
                    if self._is_excluded_by_hint(subdir_path):
                        continue
                    filtered_dirs.append(d)
                _dirs[:] = filtered_dirs

                for file_name in files:
                    file_path = os.path.join(root, file_name)
                    name_no_ext, ext = os.path.splitext(file_name)
                    ext_lower = ext.lower()

                    if self._is_excluded_by_hint(file_path):
                        logger.debug(f"Skip (path hint): {file_path}")
                        continue

                    # Filter 1: ekstensi
                    if ext_lower not in self.extensions:
                        continue

                    # Filter 2: nama file non-dokumen (LICENSE, NOTICE, dll)
                    if name_no_ext.lower() in self.skip_patterns:
                        logger.debug(f"Skip (nama non-dokumen): {file_path}")
                        continue

                    try:
                        stat = os.stat(file_path)

                        files_found.append({
                            "file_name": file_name,
                            "file_path": os.path.abspath(file_path),
                            "file_type": ext_lower,
                            "file_size": stat.st_size,
                            "last_modified": datetime.fromtimestamp(stat.st_mtime),
                        })
                    except (OSError, PermissionError) as e:
                        logger.warning(f"Tidak bisa akses file {file_path}: {e}")

        except PermissionError as e:
            logger.warning(f"Tidak bisa akses direktori {directory}: {e}")

        return files_found
