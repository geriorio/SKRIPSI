from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base
from app.config import settings

engine = create_engine(settings.DATABASE_URL, echo=False, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """Dependency untuk mendapatkan database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Inisialisasi database: buat ekstensi pgvector dan semua tabel."""
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()

    # Import semua model agar terdaftar di Base.metadata
    from app.models import file_model  # noqa: F401

    Base.metadata.create_all(bind=engine)

    # Backward-compatible schema updates for existing deployments.
    with engine.connect() as conn:
        conn.execute(
            text(
                """
                ALTER TABLE files
                ADD COLUMN IF NOT EXISTS source VARCHAR(50) NOT NULL DEFAULT 'local'
                """
            )
        )
        conn.execute(
            text(
                """
                ALTER TABLE files
                ADD COLUMN IF NOT EXISTS cloud_file_id VARCHAR(255)
                """
            )
        )
        conn.execute(
            text(
                """
                ALTER TABLE files
                ADD COLUMN IF NOT EXISTS web_view_link VARCHAR(1200)
                """
            )
        )
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_files_source ON files (source)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_files_cloud_file_id ON files (cloud_file_id)"))

        conn.execute(
            text(
                """
                ALTER TABLE index_watch_config
                ADD COLUMN IF NOT EXISTS last_local_check_at TIMESTAMP
                """
            )
        )
        conn.execute(
            text(
                """
                ALTER TABLE index_watch_config
                ADD COLUMN IF NOT EXISTS last_local_check_has_changes BOOLEAN
                """
            )
        )
        conn.execute(
            text(
                """
                ALTER TABLE index_watch_config
                ADD COLUMN IF NOT EXISTS last_local_check_stats TEXT
                """
            )
        )
        conn.execute(
            text(
                """
                ALTER TABLE index_watch_config
                ADD COLUMN IF NOT EXISTS last_gdrive_check_at TIMESTAMP
                """
            )
        )
        conn.execute(
            text(
                """
                ALTER TABLE index_watch_config
                ADD COLUMN IF NOT EXISTS last_gdrive_check_has_changes BOOLEAN
                """
            )
        )
        conn.execute(
            text(
                """
                ALTER TABLE index_watch_config
                ADD COLUMN IF NOT EXISTS last_gdrive_check_stats TEXT
                """
            )
        )
        conn.execute(
            text(
                """
                ALTER TABLE index_watch_config
                ADD COLUMN IF NOT EXISTS onedrive_folder_ids TEXT
                """
            )
        )
        conn.execute(
            text(
                """
                ALTER TABLE index_watch_config
                ADD COLUMN IF NOT EXISTS onedrive_monitor_all BOOLEAN NOT NULL DEFAULT FALSE
                """
            )
        )
        conn.execute(
            text(
                """
                ALTER TABLE index_watch_config
                ADD COLUMN IF NOT EXISTS last_onedrive_index_started_at TIMESTAMP
                """
            )
        )
        conn.execute(
            text(
                """
                ALTER TABLE index_watch_config
                ADD COLUMN IF NOT EXISTS last_onedrive_check_at TIMESTAMP
                """
            )
        )
        conn.execute(
            text(
                """
                ALTER TABLE index_watch_config
                ADD COLUMN IF NOT EXISTS last_onedrive_check_has_changes BOOLEAN
                """
            )
        )
        conn.execute(
            text(
                """
                ALTER TABLE index_watch_config
                ADD COLUMN IF NOT EXISTS last_onedrive_check_stats TEXT
                """
            )
        )
        conn.commit()
