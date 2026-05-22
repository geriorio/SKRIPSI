"""
Embedding Service — Menggunakan Sentence-BERT (SBERT) untuk membuat
representasi vektor (embedding) dari teks.

Model: distiluse-base-multilingual-cased-v2 (512 dimensi, 50+ bahasa)

Fitur:
  - Singleton pattern agar model hanya di-load sekali.
  - embed_text()   : embedding satu teks.
  - embed_texts()  : batch embedding beberapa teks.
  - chunk_text()   : memecah teks panjang menjadi chunk yang lebih kecil.
"""

import logging
from typing import List

from sentence_transformers import SentenceTransformer
from app.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Singleton service untuk SBERT embedding dan text chunking."""

    _instance = None
    _model: SentenceTransformer | None = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if EmbeddingService._model is None:
            logger.info(f"Memuat model SBERT: {settings.SBERT_MODEL} ...")
            EmbeddingService._model = SentenceTransformer(settings.SBERT_MODEL)
            logger.info("Model SBERT berhasil dimuat.")

    # ------------------------------------------------------------------
    # EMBEDDING
    # ------------------------------------------------------------------
    def embed_text(self, text: str) -> List[float]:
        """Buat embedding untuk satu teks."""
        embedding = self._model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Batch embedding untuk beberapa teks sekaligus."""
        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True,
            batch_size=batch_size,
        )
        return embeddings.tolist()

    # ------------------------------------------------------------------
    # CHUNKING
    # ------------------------------------------------------------------
    def chunk_text(
        self,
        text: str,
        chunk_size: int | None = None,
        overlap: int | None = None,
    ) -> List[str]:
        """
        Pecah teks panjang menjadi chunk-chunk yang lebih kecil.
        Respek boundary [Page N] dan [Slide N] agar setiap chunk punya marker.
        """
        import re
        
        chunk_size = chunk_size or settings.CHUNK_SIZE
        overlap = overlap or settings.CHUNK_OVERLAP

        if not text or len(text.strip()) == 0:
            return []

        text = text.strip()

        # Jika teks lebih pendek dari chunk_size, kembalikan langsung
        if len(text) <= chunk_size:
            return [text]

        # Find all [Page N] dan [Slide N] markers dan posisinya
        marker_pattern = r'\[(?:Page|Slide) \d+\]'
        marker_matches = list(re.finditer(marker_pattern, text))
        
        if not marker_matches:
            # Jika tidak ada marker, gunakan chunking standar
            chunks: List[str] = []
            start = 0
            while start < len(text):
                end = start + chunk_size
                if end < len(text):
                    for separator in [". ", ".\n", "\n\n", "\n", " "]:
                        last_sep = text[start:end].rfind(separator)
                        if last_sep > chunk_size * 0.5:
                            end = start + last_sep + len(separator)
                            break
                chunk = text[start:end].strip()
                if chunk:
                    chunks.append(chunk)
                start = end - overlap
                if start >= len(text):
                    break
            return chunks

        # Pisahkan text ke sections berdasarkan marker positions
        chunks: List[str] = []

        for i, match in enumerate(marker_matches):
            marker = match.group()

            # Content dari marker ini sampai marker berikutnya (atau akhir text)
            content_start = match.end()
            if i + 1 < len(marker_matches):
                content_end = marker_matches[i + 1].start()
            else:
                content_end = len(text)

            content = text[content_start:content_end].strip()
            if not content:
                continue

            # Ambil title/heading slide: garis pertama non-empty atau kata pertama sampai 100 char
            lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
            title = lines[0] if lines else ""
            # normalize title (singkatkan jika terlalu panjang)
            if title and len(title) > 120:
                title = title[:120].rsplit(" ", 1)[0] + "..."

            # Chunk section ini per chunk_size, dengan marker + title prepended ke tiap sub-chunk
            pos = 0
            while pos < len(content):
                end = min(pos + chunk_size, len(content))

                # Coba potong di batas kalimat
                if end < len(content):
                    for separator in [". ", ".\n", "\n\n", "\n", " "]:
                        last_sep = content[pos:end].rfind(separator)
                        if last_sep > chunk_size * 0.5:
                            end = pos + last_sep + len(separator)
                            break

                chunk_content = content[pos:end].strip()
                if chunk_content:
                    # Jika chunk_content sudah diawali oleh title, jangan duplikasi
                    preview = chunk_content[: len(title) + 5] if title else ""
                    if title and preview.startswith(title):
                        final_chunk = f"{marker} {chunk_content}"
                    elif title:
                        final_chunk = f"{marker} {title} — {chunk_content}"
                    else:
                        final_chunk = f"{marker} {chunk_content}"

                    chunks.append(final_chunk)

                pos = end

        return chunks if chunks else [text]
