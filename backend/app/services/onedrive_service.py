"""
Microsoft OneDrive OAuth + crawling service.

Service ini mengelola:
- pembuatan URL login OAuth Microsoft
- pertukaran authorization code menjadi token
- refresh token access otomatis
- listing file OneDrive (rekursif)
- download konten file
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import requests
from msal import ConfidentialClientApplication

from app.config import settings

logger = logging.getLogger(__name__)


class OneDriveService:
	"""Integrasi OAuth Microsoft Graph untuk indexing OneDrive."""

	GRAPH_BASE_URL = "https://graph.microsoft.com/v1.0"

	def __init__(self):
		self.client_id = settings.ONEDRIVE_CLIENT_ID
		self.client_secret = settings.ONEDRIVE_CLIENT_SECRET
		self.tenant_id = settings.ONEDRIVE_TENANT_ID or "common"
		self.redirect_uri = settings.ONEDRIVE_REDIRECT_URI
		self.token_file = settings.ONEDRIVE_TOKEN_FILE
		self.scopes = settings.ONEDRIVE_SCOPES
		self.supported_extensions = {ext.lower() for ext in settings.SUPPORTED_EXTENSIONS}

	@property
	def authority(self) -> str:
		return f"https://login.microsoftonline.com/{self.tenant_id}"

	@property
	def token_endpoint(self) -> str:
		return f"{self.authority}/oauth2/v2.0/token"

	def _ensure_paths(self):
		if not self.client_id:
			raise ValueError("ONEDRIVE_CLIENT_ID belum di-set")
		if not self.client_secret:
			raise ValueError("ONEDRIVE_CLIENT_SECRET belum di-set")
		if not self.redirect_uri:
			raise ValueError("ONEDRIVE_REDIRECT_URI belum di-set")
		if not self.token_file:
			raise ValueError("ONEDRIVE_TOKEN_FILE belum di-set")
		Path(self.token_file).parent.mkdir(parents=True, exist_ok=True)

	def _build_app(self) -> ConfidentialClientApplication:
		self._ensure_paths()
		return ConfidentialClientApplication(
			client_id=self.client_id,
			client_credential=self.client_secret,
			authority=self.authority,
		)

	def get_authorization_url(self) -> str:
		"""Buat URL OAuth consent Microsoft untuk login user."""
		app = self._build_app()
		return app.get_authorization_request_url(
			scopes=self.scopes,
			redirect_uri=self.redirect_uri,
			prompt="select_account",
		)

	def exchange_code(self, code: str) -> Dict:
		"""Tukar authorization code callback menjadi access/refresh token."""
		app = self._build_app()
		result = app.acquire_token_by_authorization_code(
			code=code,
			scopes=self.scopes,
			redirect_uri=self.redirect_uri,
		)

		if not result or not result.get("access_token"):
			error_desc = (result or {}).get("error_description") or (result or {}).get("error") or "unknown"
			raise RuntimeError(f"Gagal exchange code OneDrive: {error_desc}")

		self._save_token(result)
		return {
			"connected": True,
			"expiry": self._expiry_to_iso(result.get("expires_in")),
			"scopes": self.scopes,
		}

	def _save_token(self, token_payload: Dict):
		data = {
			"access_token": token_payload.get("access_token"),
			"refresh_token": token_payload.get("refresh_token"),
			"expires_at": int(datetime.utcnow().timestamp()) + int(token_payload.get("expires_in", 0) or 0),
			"scope": token_payload.get("scope", " ".join(self.scopes)),
		}
		Path(self.token_file).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

	def _load_token(self) -> Optional[Dict]:
		self._ensure_paths()
		token_path = Path(self.token_file)
		if not token_path.exists():
			return None
		try:
			data = json.loads(token_path.read_text(encoding="utf-8"))
			return data if isinstance(data, dict) else None
		except Exception as exc:
			logger.error("Gagal membaca token OneDrive: %s", exc)
			return None

	def _refresh_token(self, refresh_token: str) -> Dict:
		payload = {
			"client_id": self.client_id,
			"client_secret": self.client_secret,
			"grant_type": "refresh_token",
			"refresh_token": refresh_token,
			"redirect_uri": self.redirect_uri,
			"scope": " ".join(self.scopes),
		}
		resp = requests.post(self.token_endpoint, data=payload, timeout=30)
		resp.raise_for_status()
		token_data = resp.json()
		if not token_data.get("access_token"):
			raise RuntimeError("Refresh token OneDrive gagal: access_token kosong")
		self._save_token(token_data)
		return token_data

	def get_access_token(self) -> str:
		token_data = self._load_token()
		if not token_data:
			raise PermissionError("Belum login OneDrive OAuth. Jalankan /api/auth/onedrive/login dulu.")

		expires_at = int(token_data.get("expires_at", 0) or 0)
		now_ts = int(datetime.utcnow().timestamp())

		if token_data.get("access_token") and expires_at > (now_ts + 60):
			return token_data["access_token"]

		refresh_token = token_data.get("refresh_token")
		if not refresh_token:
			raise PermissionError("Token OneDrive kedaluwarsa dan refresh_token tidak tersedia. Login ulang diperlukan.")

		refreshed = self._refresh_token(refresh_token)
		return refreshed["access_token"]

	def is_connected(self) -> bool:
		try:
			token = self.get_access_token()
			return bool(token)
		except Exception:
			return False

	def _graph_headers(self) -> Dict[str, str]:
		token = self.get_access_token()
		return {
			"Authorization": f"Bearer {token}",
		}

	def _graph_get(self, url: str, params: Optional[Dict] = None) -> Dict:
		resp = requests.get(url, headers=self._graph_headers(), params=params, timeout=60)
		resp.raise_for_status()
		return resp.json()

	def _list_children(self, folder_id: Optional[str]) -> List[Dict]:
		if folder_id and folder_id != "root":
			url = f"{self.GRAPH_BASE_URL}/me/drive/items/{folder_id}/children"
		else:
			url = f"{self.GRAPH_BASE_URL}/me/drive/root/children"

		params = {
			"$select": "id,name,size,lastModifiedDateTime,file,folder,webUrl",
			"$top": 200,
		}

		items: List[Dict] = []
		next_url = url
		next_params = params

		while next_url:
			resp = self._graph_get(next_url, params=next_params)
			items.extend(resp.get("value", []))
			next_url = resp.get("@odata.nextLink")
			next_params = None

		return items

	def _normalize_folder_id(self, folder_id: Optional[str]) -> Optional[str]:
		if not folder_id:
			return None
		raw = folder_id.strip()
		if not raw:
			return None
		if raw.lower() in {"root", "my drive", "mydrive"}:
			return "root"
		return raw

	def list_supported_files(self, folder_id: Optional[str] = None) -> List[Dict]:
		"""List file OneDrive sesuai ekstensi yang didukung extractor."""
		normalized_folder_id = self._normalize_folder_id(folder_id) or "root"
		root_path = ""
		if normalized_folder_id != "root":
			meta = self._graph_get(
				f"{self.GRAPH_BASE_URL}/me/drive/items/{normalized_folder_id}",
				params={"$select": "name"},
			)
			root_path = str(meta.get("name") or "").strip()

		files: List[Dict] = []
		queue: List[tuple[str, str]] = [(normalized_folder_id, root_path)]

		while queue:
			current_folder_id, current_path = queue.pop(0)
			children = self._list_children(current_folder_id)

			for item in children:
				item_id = str(item.get("id") or "").strip()
				name = str(item.get("name") or "").strip()
				if not item_id or not name:
					continue

				if item.get("folder") is not None:
					next_path = f"{current_path}/{name}" if current_path else name
					queue.append((item_id, next_path))
					continue

				file_ext = Path(name).suffix.lower()
				if file_ext not in self.supported_extensions:
					continue

				logical_path = f"{current_path}/{name}" if current_path else name
				virtual_path = f"onedrive:/{logical_path}" if logical_path else f"onedrive:/{name}"

				files.append(
					{
						"file_id": item_id,
						"file_name": name,
						"file_path": virtual_path,
						"file_type": file_ext,
						"file_size": int(item.get("size", 0) or 0),
						"last_modified": self._parse_onedrive_datetime(item.get("lastModifiedDateTime")),
						"web_view_link": item.get("webUrl"),
						"source": "onedrive",
					}
				)

		return files

	def download_file_bytes(self, file_id: str) -> bytes:
		"""Download konten file OneDrive sebagai bytes untuk diekstrak."""
		url = f"{self.GRAPH_BASE_URL}/me/drive/items/{file_id}/content"
		resp = requests.get(url, headers=self._graph_headers(), timeout=120)
		resp.raise_for_status()
		return resp.content

	@staticmethod
	def _parse_onedrive_datetime(value: Optional[str]) -> datetime:
		if not value:
			return datetime.utcnow()
		return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)

	@staticmethod
	def _expiry_to_iso(expires_in: Optional[int]) -> Optional[str]:
		if not expires_in:
			return None
		try:
			future = datetime.utcnow().timestamp() + int(expires_in)
			return datetime.utcfromtimestamp(future).isoformat()
		except Exception:
			return None
