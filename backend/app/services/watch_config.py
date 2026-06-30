"""Helper untuk menyimpan dan mengambil konfigurasi watch indexing."""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from app.models.file_model import File, IndexWatchConfig


def _loads_list(value: Optional[str]) -> List[str]:
    # Ubah string JSON dari database menjadi list Python, return list kosong kalau gagal
    if not value:
        return []
    try:
        data = json.loads(value)
        if isinstance(data, list):
            # Buang item kosong, pastikan semua berupa string
            return [str(item) for item in data if str(item).strip()]
    except Exception:
        return []
    return []


def _dumps_list(values: Optional[List[str]]) -> str:
    # Ubah list Python menjadi string JSON untuk disimpan ke database
    # Sekaligus deduplikasi dan buang item kosong
    cleaned = []
    seen = set()
    for value in values or []:
        item = str(value).strip()
        if item and item not in seen:
            seen.add(item)
            cleaned.append(item)
    return json.dumps(cleaned, ensure_ascii=False)


def get_or_create_watch_config(db: Session) -> IndexWatchConfig:
    # Selalu pakai row dengan id=1 — hanya ada satu konfigurasi watch di seluruh sistem
    config = db.query(IndexWatchConfig).filter(IndexWatchConfig.id == 1).first()
    if config is None:
        # Belum ada konfigurasi, buat dengan nilai default kosong
        config = IndexWatchConfig(
            id=1,
            local_directories="[]",
            exclude_directories="[]",
            gdrive_folder_ids="[]",
            gdrive_monitor_all=False,
        )
        db.add(config)
        db.commit()
        db.refresh(config)
    return config


def get_watch_config_payload(db: Session) -> Dict[str, Any]:
    config = get_or_create_watch_config(db)
    local_check_stats = _loads_stats(config.last_local_check_stats)
    gdrive_check_stats = _loads_stats(config.last_gdrive_check_stats)
    return {
        "local_directories": _loads_list(config.local_directories),
        "exclude_directories": _loads_list(config.exclude_directories),
        "gdrive_folder_ids": _loads_list(config.gdrive_folder_ids),
        "gdrive_monitor_all": bool(config.gdrive_monitor_all),
        "updated_at": config.updated_at.isoformat() if config.updated_at else None,
        "last_local_index_started_at": (
            config.last_local_index_started_at.isoformat()
            if config.last_local_index_started_at else None
        ),
        "last_gdrive_index_started_at": (
            config.last_gdrive_index_started_at.isoformat()
            if config.last_gdrive_index_started_at else None
        ),
        "last_local_check_at": (
            config.last_local_check_at.isoformat()
            if config.last_local_check_at else None
        ),
        "last_local_check_has_changes": config.last_local_check_has_changes,
        "last_local_check_stats": local_check_stats,
        "last_gdrive_check_at": (
            config.last_gdrive_check_at.isoformat()
            if config.last_gdrive_check_at else None
        ),
        "last_gdrive_check_has_changes": config.last_gdrive_check_has_changes,
        "last_gdrive_check_stats": gdrive_check_stats,
    }


def _dumps_stats(stats: Dict[str, Any]) -> str:
    return json.dumps(stats or {}, ensure_ascii=False)


def _loads_stats(value: Optional[str]) -> Dict[str, Any]:
    if not value:
        return {}
    try:
        data = json.loads(value)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def update_watch_config(
    db: Session,
    local_directories: Optional[List[str]] = None,
    exclude_directories: Optional[List[str]] = None,
    gdrive_folder_ids: Optional[List[str]] = None,
    gdrive_monitor_all: Optional[bool] = None,
) -> Dict[str, Any]:
    # Dipanggil dari UI saat user menyimpan konfigurasi folder yang dipantau
    config = get_or_create_watch_config(db)

    # Hanya update field yang dikirim, yang tidak dikirim tidak berubah
    if local_directories is not None:
        config.local_directories = _dumps_list(local_directories)
    if exclude_directories is not None:
        config.exclude_directories = _dumps_list(exclude_directories)
    if gdrive_folder_ids is not None:
        config.gdrive_folder_ids = _dumps_list(gdrive_folder_ids)
    if gdrive_monitor_all is not None:
        config.gdrive_monitor_all = bool(gdrive_monitor_all)

    # Catat waktu terakhir konfigurasi diubah
    config.updated_at = datetime.utcnow()
    db.add(config)
    db.commit()
    db.refresh(config)
    return get_watch_config_payload(db)


def mark_local_index_started(db: Session):
    # Dipanggil scheduler setelah berhasil memulai indexing lokal
    # Mencatat timestamp agar UI bisa tampilkan kapan terakhir indexing dijalankan
    config = get_or_create_watch_config(db)
    config.last_local_index_started_at = datetime.utcnow()
    db.add(config)
    db.commit()


def mark_gdrive_index_started(db: Session):
    # Dipanggil scheduler setelah berhasil memulai indexing Google Drive
    config = get_or_create_watch_config(db)
    config.last_gdrive_index_started_at = datetime.utcnow()
    db.add(config)
    db.commit()


def mark_local_check_result(db: Session, stats: Dict[str, Any]):
    # Simpan hasil pengecekan perubahan file lokal ke database
    config = get_or_create_watch_config(db)
    # Hitung total perubahan: file baru + file diupdate + file dihapus
    changes = int(stats.get("new", 0)) + int(stats.get("updated", 0)) + int(stats.get("deleted", 0))
    config.last_local_check_at = datetime.utcnow()
    # True kalau ada perubahan, False kalau tidak ada — ditampilkan di UI
    config.last_local_check_has_changes = changes > 0
    config.last_local_check_stats = _dumps_stats(stats)
    db.add(config)
    db.commit()


def mark_gdrive_check_result(db: Session, stats: Dict[str, Any]):
    # Simpan hasil pengecekan perubahan file Google Drive ke database
    config = get_or_create_watch_config(db)
    changes = int(stats.get("new", 0)) + int(stats.get("updated", 0)) + int(stats.get("deleted", 0))
    config.last_gdrive_check_at = datetime.utcnow()
    config.last_gdrive_check_has_changes = changes > 0
    config.last_gdrive_check_stats = _dumps_stats(stats)
    db.add(config)
    db.commit()


def get_effective_watch_targets(db: Session) -> Dict[str, Any]:
    """
    Ambil target watch yang efektif.

    Prioritas:
    1) Konfigurasi watch yang disimpan user
    2) Fallback otomatis dari data file yang sudah ter-index
    """
    cfg = get_watch_config_payload(db)

    local_dirs = cfg.get("local_directories") or []
    exclude_dirs = cfg.get("exclude_directories") or []
    gdrive_folder_ids = cfg.get("gdrive_folder_ids") or []
    gdrive_monitor_all = bool(cfg.get("gdrive_monitor_all"))

    # Fallback local: kalau user belum set direktori di UI,
    # ambil otomatis dari direktori file yang sudah ter-index di database
    if not local_dirs:
        local_paths = (
            db.query(File.file_path)
            .filter((File.source == "local") | (File.source.is_(None)))
            .all()
        )
        derived_dirs = []
        seen = set()
        for (path,) in local_paths:
            if not path:
                continue
            # Ambil direktori dari path file, bukan nama filenya
            directory = os.path.dirname(path)
            if directory and directory not in seen:
                seen.add(directory)
                derived_dirs.append(directory)
        local_dirs = derived_dirs

    # Fallback GDrive: kalau belum ada folder yang diset tapi sudah ada file GDrive di DB,
    # aktifkan monitor_all agar semua file GDrive yang sudah ter-index tetap terpantau
    if not gdrive_monitor_all and not gdrive_folder_ids:
        gdrive_count = db.query(File.id).filter(File.source == "gdrive").count()
        if gdrive_count > 0:
            gdrive_monitor_all = True

    return {
        "local_directories": local_dirs,
        "exclude_directories": exclude_dirs,
        "gdrive_folder_ids": gdrive_folder_ids,
        "gdrive_monitor_all": gdrive_monitor_all,
        "source": "watch_config_or_indexed_files",
    }
