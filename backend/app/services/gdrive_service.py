"""
Google Drive OAuth + crawling service.

Service ini mengelola:
- pembuatan URL login OAuth
- pertukaran authorization code menjadi token
- load/refresh token
- listing file Google Drive
- download konten file
"""

import io
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from app.config import settings

logger = logging.getLogger(__name__)


class GoogleDriveService:
    """Integrasi OAuth dan API Google Drive untuk indexing."""

    SUPPORTED_MIME_TO_EXT = {
        "application/pdf": ".pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
        "text/csv": ".csv",
        "application/vnd.ms-powerpoint": ".ppt",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
        "text/plain": ".txt",
        # Google native files (harus diexport dulu sebelum diekstrak)
        "application/vnd.google-apps.document": ".docx",
        "application/vnd.google-apps.spreadsheet": ".xlsx",
        "application/vnd.google-apps.presentation": ".pptx",
    }

    GOOGLE_EXPORT_MIME = {
        "application/vnd.google-apps.document": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.google-apps.spreadsheet": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.google-apps.presentation": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    }

    def __init__(self):
        self.client_file = settings.GOOGLE_OAUTH_CLIENT_FILE
        self.token_file = settings.GOOGLE_OAUTH_TOKEN_FILE
        self.redirect_uri = settings.GOOGLE_OAUTH_REDIRECT_URI
        self.scopes = settings.GOOGLE_OAUTH_SCOPES

    def _ensure_paths(self):
        if not self.client_file:
            raise ValueError("GOOGLE_OAUTH_CLIENT_FILE belum di-set")
        if not Path(self.client_file).exists():
            raise FileNotFoundError(f"Client OAuth file tidak ditemukan: {self.client_file}")
        if not self.token_file:
            raise ValueError("GOOGLE_OAUTH_TOKEN_FILE belum di-set")
        Path(self.token_file).parent.mkdir(parents=True, exist_ok=True)

    def get_authorization_url(self) -> str:
        """Buat URL OAuth consent untuk login Google akun user."""
        self._ensure_paths()

        flow = Flow.from_client_secrets_file(
            self.client_file,
            scopes=self.scopes,
            redirect_uri=self.redirect_uri,
        )

        auth_url, _state = flow.authorization_url(
            access_type="offline",
            include_granted_scopes="true",
            prompt="consent",
        )
        return auth_url

    def exchange_code(self, code: str) -> Dict:
        """Tukar authorization code dari callback menjadi access/refresh token."""
        self._ensure_paths()

        flow = Flow.from_client_secrets_file(
            self.client_file,
            scopes=self.scopes,
            redirect_uri=self.redirect_uri,
        )
        flow.fetch_token(code=code)
        creds = flow.credentials

        self._save_credentials(creds)

        return {
            "connected": True,
            "expiry": str(creds.expiry) if creds.expiry else None,
            "scopes": list(creds.scopes or []),
        }

    def _save_credentials(self, creds: Credentials):
        with open(self.token_file, "w", encoding="utf-8") as f:
            f.write(creds.to_json())

    def _load_credentials(self) -> Optional[Credentials]:
        self._ensure_paths()
        token_path = Path(self.token_file)
        if not token_path.exists():
            return None

        try:
            data = json.loads(token_path.read_text(encoding="utf-8"))
            creds = Credentials.from_authorized_user_info(data, self.scopes)
        except Exception as e:
            logger.error(f"Gagal membaca token Google OAuth: {e}")
            return None

        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            self._save_credentials(creds)

        return creds if creds and creds.valid else None

    def is_connected(self) -> bool:
        try:
            creds = self._load_credentials()
            return creds is not None
        except Exception:
            return False

    def build_drive_client(self):
        """Bangun client Google Drive API dengan token user ter-login."""
        creds = self._load_credentials()
        if creds is None:
            raise PermissionError("Belum login Google OAuth. Jalankan /api/auth/google/login dulu.")
        return build("drive", "v3", credentials=creds)

    def list_supported_files(self, folder_id: Optional[str] = None) -> List[Dict]:
        """List file Google Drive sesuai format yang didukung extractor."""
        service = self.build_drive_client()
        mime_list = list(self.SUPPORTED_MIME_TO_EXT.keys())
        mime_filter = " or ".join([f"mimeType='{m}'" for m in mime_list])
        folder_path_cache: Dict[str, str] = {}
        files: List[Dict] = []

        target_folder_id = self._normalize_folder_id(folder_id)
        if target_folder_id:
            # Traversal rekursif: ambil file pada folder target + seluruh subfolder.
            folder_ids = self._collect_folder_tree_ids(service, target_folder_id)
            for current_folder_id in folder_ids:
                files.extend(
                    self._list_supported_files_by_parent(
                        service=service,
                        parent_id=current_folder_id,
                        mime_filter=mime_filter,
                        folder_path_cache=folder_path_cache,
                    )
                )
        else:
            # Tanpa folder_id: ambil semua file yang bisa diakses akun (sesuai filter MIME).
            files.extend(
                self._list_supported_files_by_parent(
                    service=service,
                    parent_id=None,
                    mime_filter=mime_filter,
                    folder_path_cache=folder_path_cache,
                )
            )

        return files

    def _normalize_folder_id(self, folder_id: Optional[str]) -> Optional[str]:
        """Normalisasi input folder id agar lebih user-friendly."""
        if not folder_id:
            return None

        value = folder_id.strip()
        if not value:
            return None

        if value.lower() in {"my drive", "mydrive"}:
            return "root"

        return value

    def _collect_folder_tree_ids(self, service, root_folder_id: str) -> List[str]:
        """Kumpulkan id folder root + seluruh descendant folder (BFS)."""
        visited = set()
        queue = [root_folder_id]
        ordered_ids: List[str] = []

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue

            visited.add(current)
            ordered_ids.append(current)

            subfolders = self._list_subfolders(service, current)
            for sub in subfolders:
                sub_id = sub.get("id")
                if sub_id and sub_id not in visited:
                    queue.append(sub_id)

        return ordered_ids

    def _list_subfolders(self, service, parent_id: str) -> List[Dict]:
        """Ambil subfolder langsung dari suatu parent folder."""
        query = (
            "trashed=false and "
            "mimeType='application/vnd.google-apps.folder' and "
            f"'{parent_id}' in parents"
        )

        page_token = None
        items: List[Dict] = []
        while True:
            resp = service.files().list(
                q=query,
                spaces="drive",
                fields="nextPageToken, files(id, name)",
                pageToken=page_token,
                pageSize=200,
            ).execute()

            items.extend(resp.get("files", []))
            page_token = resp.get("nextPageToken")
            if not page_token:
                break

        return items

    def _list_supported_files_by_parent(
        self,
        service,
        parent_id: Optional[str],
        mime_filter: str,
        folder_path_cache: Dict[str, str],
    ) -> List[Dict]:
        """List file yang didukung pada parent tertentu atau seluruh drive jika parent_id None."""
        query_parts = ["trashed=false", f"({mime_filter})"]
        if parent_id:
            query_parts.append(f"'{parent_id}' in parents")
        query = " and ".join(query_parts)

        page_token = None
        collected: List[Dict] = []
        while True:
            resp = service.files().list(
                q=query,
                spaces="drive",
                fields="nextPageToken, files(id, name, mimeType, size, modifiedTime, webViewLink, parents)",
                pageToken=page_token,
                pageSize=200,
            ).execute()

            for f in resp.get("files", []):
                ext = self.SUPPORTED_MIME_TO_EXT.get(f.get("mimeType"))
                if not ext:
                    continue

                modified = self._parse_google_datetime(f.get("modifiedTime"))
                name = f.get("name", "unknown")
                file_id = f.get("id")
                actual_parent_id = (f.get("parents") or [None])[0]

                folder_path = self._build_folder_path(service, actual_parent_id, folder_path_cache)
                virtual_path = f"gdrive:/{folder_path}/{name}" if folder_path else f"gdrive:/{name}"

                collected.append({
                    "file_id": file_id,
                    "file_name": name,
                    "file_path": virtual_path,
                    "file_type": ext,
                    "source_mime_type": f.get("mimeType"),
                    "file_size": int(f.get("size", 0) or 0),
                    "last_modified": modified,
                    "web_view_link": f.get("webViewLink"),
                    "source": "gdrive",
                })

            page_token = resp.get("nextPageToken")
            if not page_token:
                break

        return collected

    def _build_folder_path(
        self,
        service,
        folder_id: Optional[str],
        cache: Dict[str, str],
    ) -> str:
        """Bangun path folder dari parent ID secara rekursif dengan cache."""
        if not folder_id:
            return ""

        if folder_id in cache:
            return cache[folder_id]

        try:
            # Ambil metadata folder parent (nama + parent berikutnya).
            meta = service.files().get(
                fileId=folder_id,
                fields="id, name, parents",
            ).execute()
            name = (meta.get("name") or "").strip()
            parent_ids = meta.get("parents") or []
            parent_path = self._build_folder_path(service, parent_ids[0] if parent_ids else None, cache)

            full_path = f"{parent_path}/{name}" if parent_path and name else name
            cache[folder_id] = full_path
            return full_path
        except Exception as e:
            logger.warning(f"Gagal resolve folder path untuk parent {folder_id}: {e}")
            cache[folder_id] = ""
            return ""

    def download_file_bytes(self, file_id: str, source_mime_type: Optional[str] = None) -> bytes:
        """Download/Export konten file Drive sebagai bytes untuk diekstrak."""
        service = self.build_drive_client()
        export_mime = self.GOOGLE_EXPORT_MIME.get((source_mime_type or "").strip())

        if export_mime:
            # Google Docs/Sheets/Slides native tidak bisa get_media langsung, harus export.
            req = service.files().export_media(fileId=file_id, mimeType=export_mime)
        else:
            req = service.files().get_media(fileId=file_id)

        stream = io.BytesIO()
        downloader = MediaIoBaseDownload(stream, req)

        done = False
        while not done:
            _status, done = downloader.next_chunk()

        return stream.getvalue()

    @staticmethod
    def _parse_google_datetime(value: Optional[str]) -> datetime:
        if not value:
            return datetime.utcnow()
        # Google format: 2026-03-15T08:33:10.123Z
        return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
