"""
Search Service — Hybrid retrieval: Metadata + BM25 + SBERT.

Alur:
  Mode lookup  : metadata search + BM25 + SBERT → score fusion (0.35/0.35/0.30)
  Mode semantic: BM25 + SBERT → score fusion (0.40/0.60)

Query preprocessing: strip kata non-konten sebelum BM25 dan SBERT embedding.
BM25 index dibangun dari semua chunk di DB, di-cache in-memory.
"""

import logging
import re
import threading
from difflib import SequenceMatcher
from types import SimpleNamespace
from typing import List, Dict, Set, Optional, Tuple

from sqlalchemy.orm import Session
from sqlalchemy import text

from app.config import settings
from app.models.file_model import File
from app.services.embedder import EmbeddingService

logger = logging.getLogger(__name__)


class SearchService:
    """Hybrid search: Metadata + BM25 + SBERT."""

    def __init__(self):
        self.embedder = EmbeddingService()
        # BM25 cache: (BM25Okapi, chunk_data_list, chunk_count)
        self._bm25_cache: Optional[Tuple] = None
        self._bm25_lock = threading.Lock()

    @staticmethod
    def _tokenize(text: str) -> List[str]: #jadi list kata
        return re.findall(r"\w+", (text or "").lower())

    @staticmethod
    def _normalize_compact(text: str) -> str: #hapus karakter dan angka
        return re.sub(r"[^a-z0-9]+", "", (text or "").lower())

    def _compute_metadata_debug(
        self,
        query: str,
        file_name: str | None,
        file_path: str | None,
    ) -> Dict[str, float]:
        q = (query or "").strip().lower()
        q_compact = self._normalize_compact(q)
        tokens = [t for t in self._tokenize(q) if len(t) >= 2]
        query_terms = [t for t in tokens if t not in {
            "dimana", "mana", "yang", "itu", "ini", "ada", "di", "ke", "dari",
            "untuk", "dan", "atau", "file", "dokumen", "folder", "lokasi", "letak",
            "tolong", "cari", "carikan", "saya", "punya",
        }]
        if not query_terms:
            query_terms = tokens

        name = (file_name or "").lower()
        path = (file_path or "").lower()
        name_compact = self._normalize_compact(name)
        path_compact = self._normalize_compact(path)
        metadata_text = f"{name} {path}"
        metadata_tokens = [t for t in self._tokenize(metadata_text) if len(t) >= 3]
        metadata_compact = f"{name_compact} {path_compact}"

        exact_term_hits = 0
        partial_term_hits = 0
        fuzzy_term_hits = 0
        for term in query_terms:
            if term in metadata_text:
                exact_term_hits += 1
                continue

            if any(term in meta_term or meta_term in term for meta_term in metadata_tokens):
                partial_term_hits += 1
                continue

            best_ratio = 0.0
            for meta_term in metadata_tokens:
                ratio = SequenceMatcher(None, term, meta_term).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
            if best_ratio >= 0.82:
                fuzzy_term_hits += 1

        score = 0
        score += exact_term_hits * 12
        score += partial_term_hits * 9
        score += fuzzy_term_hits * 6

        name_ratio = SequenceMatcher(None, q_compact, name_compact).ratio() if q_compact and name_compact else 0.0
        path_ratio = SequenceMatcher(None, q_compact, path_compact).ratio() if q_compact and path_compact else 0.0
        metadata_ratio = SequenceMatcher(None, q_compact, self._normalize_compact(metadata_text)).ratio() if q_compact else 0.0
        score += int(max(name_ratio * 35, path_ratio * 25, metadata_ratio * 20))

        overlap = 0
        if query_terms:
            overlap = sum(1 for term in query_terms if term in metadata_text or any(term in meta_term or meta_term in term for meta_term in metadata_tokens))
            score += int((overlap / max(len(query_terms), 1)) * 30)

        compact_match = False
        if q_compact and q_compact in metadata_compact:
            compact_match = True
            score += 20

        phrase_match = False
        if len(query_terms) >= 2:
            query_phrase = " ".join(query_terms)
            if query_phrase in metadata_text:
                phrase_match = True
                score += 12

        return {
            "metadata_score": float(score),
            "metadata_exact_hits": float(exact_term_hits),
            "metadata_partial_hits": float(partial_term_hits),
            "metadata_fuzzy_hits": float(fuzzy_term_hits),
            "metadata_name_ratio": float(name_ratio),
            "metadata_path_ratio": float(path_ratio),
            "metadata_text_ratio": float(metadata_ratio),
            "metadata_overlap": float(overlap),
            "metadata_compact_match": 1.0 if compact_match else 0.0,
            "metadata_phrase_match": 1.0 if phrase_match else 0.0,
        }

    def _is_file_lookup_query(self, query: str) -> bool:
        q = (query or "").lower()

        # Lapisan 1: compound phrase yang secara eksplisit bermakna "cari/temukan file"
        strong_phrases = [
            # "di mana/dimana" + objek file
            "dimana file", "dimana dokumen", "dimana folder",
            "di mana file", "di mana dokumen", "di mana folder",
            "mana file",
            # perintah cari
            "cari file", "cari dokumen", "temukan file",
            # metadata file
            "lokasi file", "letak file",
            "nama file", "nama dokumen",
            # pertanyaan "yang mana"
            "file mana", "dokumen mana",
            "di folder mana",
            # keberadaan file
            "ada file", "ada dokumen",
            "file bernama", "dokumen bernama",
        ]
        if any(phrase in q for phrase in strong_phrases):
            return True

        # Lapisan 2: ekstensi file adalah signal kuat
        if any(ext in q for ext in [".pdf", ".docx", ".xlsx", ".csv", ".pptx", ".txt"]):
            return True

        # Lapisan 3: kata lokasi ("dimana"/"mana") hanya trigger jika ada
        # kata konteks-file juga — mencegah false positive seperti
        # "Di mana kantor pusat?" atau "Bagaimana lokasi cabang di laporan?"
        tokens = set(self._tokenize(q))
        has_location = bool(tokens.intersection({"dimana", "mana"}))
        has_file_context = bool(tokens.intersection({"file", "dokumen", "folder"}))
        return has_location and has_file_context

    def _metadata_exact_candidates(
        self,
        db: Session,
        query: str,
        top_k_files: int,
        gdrive_requested: bool,
    ) -> List[tuple[int, int]]:
        """Cari kandidat soft match di metadata nama/path file."""
        # Query terlalu pendek tidak bisa diandalkan untuk mencocokkan nama file
        q = (query or "").strip().lower()
        if len(q) < 3:
            return []

        # q_compact: query tanpa spasi dan karakter non-alfanumerik, untuk mencocokkan nama file yang pakai strip/titik/dll
        q_compact = self._normalize_compact(q)
        # Pecah query jadi token kata, minimal 2 huruf
        tokens = [t for t in self._tokenize(q) if len(t) >= 2]
        # Buang kata-kata umum yang tidak mencerminkan nama file spesifik
        query_terms = [t for t in tokens if t not in {
            "dimana", "mana", "yang", "itu", "ini", "ada", "di", "ke", "dari",
            "untuk", "dan", "atau", "file", "dokumen", "folder", "lokasi", "letak",
            "tolong", "cari", "carikan", "saya", "punya",
        }]
        if not query_terms:
            query_terms = tokens

        # Ambil semua file dari database, akan dibandingkan satu per satu dengan query
        base_query = db.query(File.id, File.file_name, File.file_path, File.source)

        scored: List[tuple[int, int]] = []
        for file_id, file_name, file_path, _source in base_query.all():
            name = (file_name or "").lower()
            path = (file_path or "").lower()
            # Versi compact dari nama dan path file untuk fuzzy matching tanpa pemisah
            name_compact = self._normalize_compact(name)
            path_compact = self._normalize_compact(path)
            # Gabungkan nama dan path jadi satu teks untuk pencarian sekaligus
            metadata_text = f"{name} {path}"
            metadata_tokens = [t for t in self._tokenize(metadata_text) if len(t) >= 3]
            metadata_compact = f"{name_compact} {path_compact}"

            score = 0

            # OLD EXACT FLOW (referensi):
            # if q == name or q == path:
            #     score += 120
            # if q in name:
            #     score += 90
            # if q in path:
            #     score += 70
            # if q_compact and q_compact in name_compact:
            #     score += 85
            # if q_compact and q_compact in path_compact:
            #     score += 65
            # token_hits_name = sum(1 for t in tokens if t in name)
            # token_hits_path = sum(1 for t in tokens if t in path)
            # score += (token_hits_name * 6) + (token_hits_path * 3)

            # Setiap kata dari query dicek ke nama/path file dengan 3 tingkat kecocokan:
            # exact = kata persis ada di metadata, partial = sebagian cocok, fuzzy = mirip meski typo
            exact_term_hits = 0
            partial_term_hits = 0
            fuzzy_term_hits = 0
            for term in query_terms:
                # Tingkat 1: kata query persis ada di nama atau path file
                if term in metadata_text:
                    exact_term_hits += 1
                    continue

                # Tingkat 2: kata query adalah bagian dari kata di metadata, atau sebaliknya
                if any(term in meta_term or meta_term in term for meta_term in metadata_tokens):
                    partial_term_hits += 1
                    continue

                # Tingkat 3: cek kemiripan karakter, toleransi typo (threshold 0.82)
                best_ratio = 0.0
                for meta_term in metadata_tokens:
                    ratio = SequenceMatcher(None, term, meta_term).ratio()
                    if ratio > best_ratio:
                        best_ratio = ratio
                if best_ratio >= 0.82:
                    fuzzy_term_hits += 1

            # Bobot: exact lebih tinggi karena paling akurat
            score += exact_term_hits * 12
            score += partial_term_hits * 9
            score += fuzzy_term_hits * 6

            # Tambahan skor dari kemiripan keseluruhan query vs nama/path file (SequenceMatcher 0-1)
            name_ratio = SequenceMatcher(None, q_compact, name_compact).ratio() if q_compact and name_compact else 0.0
            path_ratio = SequenceMatcher(None, q_compact, path_compact).ratio() if q_compact and path_compact else 0.0
            metadata_ratio = SequenceMatcher(None, q_compact, self._normalize_compact(metadata_text)).ratio() if q_compact else 0.0
            score += int(max(name_ratio * 35, path_ratio * 25, metadata_ratio * 20))

            # (presentase kecocokan) Tambahan skor dari proporsi kata query yang cocok ke metadata
            if query_terms:
                overlap = sum(1 for term in query_terms if term in metadata_text or any(term in meta_term or meta_term in term for meta_term in metadata_tokens))
                score += int((overlap / max(len(query_terms), 1)) * 30)

            # (cek misal ada yang terpisah _) Bonus kalau seluruh query compact ada di dalam metadata compact
            if q_compact and q_compact in metadata_compact:
                score += 20

            # Bonus kalau beberapa kata query muncul berurutan di metadata
            if len(query_terms) >= 2:
                query_phrase = " ".join(query_terms)
                if query_phrase in metadata_text:
                    score += 12

            # Hanya simpan file yang punya skor lebih dari 0
            if score > 0:
                scored.append((file_id, score))

        # Urutkan dari skor tertinggi, ambil maksimal 3x top_k_files atau minimal 12
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[: max(top_k_files * 3, 12)]

    def _extract_focus_terms(self, query: str) -> List[str]:
        q = (query or "").lower()
        tail = q
        for marker in ["file", "dokumen"]:
            if marker in q:
                tail = q.split(marker, 1)[1]
                break

        stopwords: Set[str] = {
            "dimana", "mana", "yang", "itu", "ini", "ada", "di", "ke", "dari",
            "untuk", "dan", "atau", "file", "dokumen", "folder", "lokasi", "letak",
            "tolong", "cari", "carikan", "saya", "punya",
        }

        terms: List[str] = []
        seen: Set[str] = set()
        for token in self._tokenize(tail):
            if len(token) < 2 or token in stopwords:
                continue
            if token in seen:
                continue
            seen.add(token)
            terms.append(token)
        return terms

    @staticmethod
    def _mentions_gdrive(query: str) -> bool:
        q = (query or "").lower()
        return any(k in q for k in ["google drive", "gdrive", "drive"])

    def _metadata_search_candidates(
        self,
        db: Session,
        query: str,
        top_k_files: int,
        focus_terms: List[str],
        gdrive_requested: bool,
    ) -> List[tuple[int, int]]:
        """Cari kandidat file dari metadata nama/path agar file spesifik tidak tenggelam."""
        # focus_terms adalah kata kunci inti dari query, sudah dibuang stopword-nya
        # Kalau kosong, tidak ada yang bisa dicocokkan ke nama/path file
        if not focus_terms:
            return []

        # Ambil semua file dari database
        base_query = db.query(File.id, File.file_name, File.file_path, File.source)

        scored_candidates: List[tuple[int, int]] = []
        for file_id, file_name, file_path, source in base_query.all():
            # Gabungkan nama file dan path jadi satu teks untuk pencocokan
            metadata_text = f"{file_name or ''} {file_path or ''}".lower()
            # Hitung berapa kata dari focus_terms yang muncul di metadata file ini
            matched = sum(1 for term in focus_terms if term in metadata_text)
            # File yang tidak mengandung satu pun kata dari focus_terms dilewati
            if matched == 0:
                continue

            name_text = (file_name or "").lower()
            path_text = (file_path or "").lower()
            # Bonus jika kata kunci ada di nama file (lebih spesifik dari path)
            if any(term in name_text for term in focus_terms):
                matched += 2
            # Bonus lebih kecil jika ada di path saja
            if any(term in path_text for term in focus_terms):
                matched += 1

            scored_candidates.append((file_id, matched))

        # Urutkan dari yang paling banyak cocoknya, ambil maksimal 3x top_k_files atau minimal 12
        scored_candidates.sort(key=lambda item: item[1], reverse=True)
        return scored_candidates[: max(top_k_files * 3, 12)]

    def _combine_metadata_candidates(
        self,
        exact_candidates: List[tuple[int, int]],
        keyword_candidates: List[tuple[int, int]],
    ) -> List[tuple[int, int]]:
        """Gabungkan kandidat exact + keyword metadata sambil menjaga skor tertinggi per file."""
        # combined adalah dict: kunci = file_id, nilai = skor tertinggi yang pernah didapat file itu
        combined: Dict[int, int] = {}
        # Masukkan semua file dari exact_candidates
        for file_id, score in exact_candidates or []:
            # Kalau file belum ada di combined, get() return 0 lalu diisi skor ini
            # Kalau sudah ada, ambil yang lebih tinggi antara skor lama dan skor baru
            # kalau beririsan ambil skor tertinggi, kalau tidak ada, masukkan
            combined[file_id] = max(combined.get(file_id, 0), int(score))
        # Masukkan semua file dari keyword_candidates dengan cara yang sama
        for file_id, score in keyword_candidates or []:
            # File yang muncul di keduanya hanya menyimpan skor tertinggi, tidak dijumlah
            combined[file_id] = max(combined.get(file_id, 0), int(score))

        # Urutkan hasil gabungan dari skor tertinggi ke terendah
        ordered = sorted(combined.items(), key=lambda item: item[1], reverse=True)
        return ordered

    def _build_metadata_lookup_rows(
        self,
        db: Session,
        candidate_items: List[tuple[int, int]],
        query: str,
        query_embedding: str,
    ) -> List[SimpleNamespace]:
        """Bangun rows hasil lookup berbasis metadata, dengan chunk jika tersedia."""
        rows: List[SimpleNamespace] = []
        seen: Set[int] = set()

        for file_id, candidate_score in candidate_items:
            if file_id in seen:
                continue
            seen.add(file_id)

            file_meta = db.query(File).filter(File.id == file_id).first()
            if not file_meta:
                continue

            debug_meta = self._compute_metadata_debug(query, file_meta.file_name, file_meta.file_path)

            extra_chunks = self._fetch_top_chunks_for_file(
                db=db,
                file_id=file_id,
                query_embedding=query_embedding,
                limit=2,
            )
            if extra_chunks:
                for chunk in extra_chunks:
                    raw_similarity = float(chunk["similarity"])
                    boosted_similarity = max(raw_similarity, min(0.99, 0.50 + (0.01 * candidate_score)))
                    rows.append(
                        SimpleNamespace(
                            chunk_id=chunk["chunk_id"],
                            chunk_text=chunk["chunk_text"],
                            chunk_index=chunk["chunk_index"],
                            file_id=file_meta.id,
                            source=file_meta.source,
                            file_name=file_meta.file_name,
                            file_path=file_meta.file_path,
                            file_type=file_meta.file_type,
                            file_size=file_meta.file_size,
                            last_modified=file_meta.last_modified,
                            similarity=boosted_similarity,
                            raw_similarity=raw_similarity,
                            boosted_similarity=boosted_similarity,
                            metadata_score=float(candidate_score),
                            metadata_exact_hits=debug_meta["metadata_exact_hits"],
                            metadata_partial_hits=debug_meta["metadata_partial_hits"],
                            metadata_fuzzy_hits=debug_meta["metadata_fuzzy_hits"],
                            metadata_name_ratio=debug_meta["metadata_name_ratio"],
                            metadata_path_ratio=debug_meta["metadata_path_ratio"],
                            metadata_text_ratio=debug_meta["metadata_text_ratio"],
                            metadata_overlap=debug_meta["metadata_overlap"],
                            metadata_compact_match=debug_meta["metadata_compact_match"],
                            metadata_phrase_match=debug_meta["metadata_phrase_match"],
                        )
                    )
                continue

            # Metadata-only fallback jika file ada tapi konten belum menghasilkan chunk.
            rows.append(
                SimpleNamespace(
                    chunk_id=None,
                    chunk_text="",
                    chunk_index=-1,
                    file_id=file_meta.id,
                    source=file_meta.source,
                    file_name=file_meta.file_name,
                    file_path=file_meta.file_path,
                    file_type=file_meta.file_type,
                    file_size=file_meta.file_size,
                    last_modified=file_meta.last_modified,
                    similarity=min(0.99, 0.60 + (0.01 * candidate_score)),
                    raw_similarity=None,
                    boosted_similarity=min(0.99, 0.60 + (0.01 * candidate_score)),
                    metadata_score=float(candidate_score),
                    metadata_exact_hits=debug_meta["metadata_exact_hits"],
                    metadata_partial_hits=debug_meta["metadata_partial_hits"],
                    metadata_fuzzy_hits=debug_meta["metadata_fuzzy_hits"],
                    metadata_name_ratio=debug_meta["metadata_name_ratio"],
                    metadata_path_ratio=debug_meta["metadata_path_ratio"],
                    metadata_text_ratio=debug_meta["metadata_text_ratio"],
                    metadata_overlap=debug_meta["metadata_overlap"],
                    metadata_compact_match=debug_meta["metadata_compact_match"],
                    metadata_phrase_match=debug_meta["metadata_phrase_match"],
                )
            )

        return rows

    def _fetch_top_chunks_for_file(
        self,
        db: Session,
        file_id: int,
        query_embedding: str,
        limit: int = 2,
    ) -> List[Dict]:
        rows = db.execute(
            text(
                """
                SELECT
                    fc.id AS chunk_id,
                    fc.chunk_text,
                    fc.chunk_index,
                    fc.file_id AS file_id,
                    1 - (fc.embedding <=> CAST(:query_embedding AS vector)) AS similarity
                FROM file_chunks fc
                WHERE fc.file_id = :file_id
                  AND fc.embedding IS NOT NULL
                ORDER BY fc.embedding <=> CAST(:query_embedding AS vector)
                LIMIT :limit
                """
            ),
            {
                "file_id": file_id,
                "query_embedding": query_embedding,
                "limit": limit,
            },
        ).fetchall()

        return [
            {
                "chunk_id": row.chunk_id,
                "chunk_index": row.chunk_index,
                "chunk_text": row.chunk_text,
                "similarity": float(row.similarity),
            }
            for row in rows
        ]

    def is_file_lookup_query(self, query: str) -> bool:
        return self._is_file_lookup_query(query)

    # ------------------------------------------------------------------
    # QUERY PREPROCESSING
    # ------------------------------------------------------------------
    def _strip_query_noise(self, query: str) -> str:
        """Hapus kata non-konten dari query sebelum embedding dan BM25."""
        noise: Set[str] = {
            "dimana", "mana", "yang", "itu", "ini", "ada", "di", "ke", "dari",
            "untuk", "dan", "atau", "file", "dokumen", "folder", "lokasi", "letak",
            "tolong", "cari", "carikan", "saya", "punya", "tentang", "menjelaskan",
            "mengenai", "berisi", "membahas", "jelaskan", "beritahu", "tunjukkan",
            "apakah", "apa", "bagaimana", "gimana", "adalah", "sebuah", "suatu",
        }
        tokens = [t for t in self._tokenize(query) if t not in noise and len(t) >= 2]
        return " ".join(tokens) if tokens else query

    # ------------------------------------------------------------------
    # BM25
    # ------------------------------------------------------------------
    def _get_bm25_index(self, db: Session) -> Tuple:
        """Bangun atau ambil BM25 index dari cache. Rebuild jika jumlah chunk berubah."""
        from rank_bm25 import BM25Okapi

        count = db.execute(
            text("SELECT COUNT(*) FROM file_chunks WHERE chunk_text IS NOT NULL AND LENGTH(TRIM(chunk_text)) > 10")
        ).scalar() or 0

        with self._bm25_lock:
            if self._bm25_cache and self._bm25_cache[2] == count:
                return self._bm25_cache

            rows = db.execute(text("""
                SELECT fc.id, fc.chunk_text, fc.chunk_index,
                       f.id AS file_id, f.file_name, f.file_path,
                       f.file_type, f.file_size, f.last_modified, f.source
                FROM file_chunks fc
                JOIN files f ON fc.file_id = f.id
                WHERE fc.chunk_text IS NOT NULL AND LENGTH(TRIM(fc.chunk_text)) > 10
            """)).fetchall()

            chunk_data = [
                {
                    "id": r.id, "chunk_text": r.chunk_text, "chunk_index": r.chunk_index,
                    "file_id": r.file_id, "file_name": r.file_name, "file_path": r.file_path,
                    "file_type": r.file_type, "file_size": r.file_size,
                    "last_modified": r.last_modified, "source": r.source,
                }
                for r in rows
            ]
            corpus = [self._tokenize(c["chunk_text"]) for c in chunk_data]
            bm25 = BM25Okapi(corpus)
            self._bm25_cache = (bm25, chunk_data, count)
            logger.info("BM25 index dibangun: %d chunk", len(chunk_data))
            return self._bm25_cache

    def _retrieve_bm25_chunks(self, query: str, db: Session, top_k: int) -> List[Dict]:
        """Retrieve top-K chunk via BM25, kembalikan dengan skor ternormalisasi 0-1."""
        # Query kosong tidak bisa dihitung skor BM25-nya, langsung return kosong
        if not query.strip():
            return []
        try:
            # Ambil BM25 index dari cache (atau bangun ulang kalau belum ada)
            # Hasilnya: bm25 = objek BM25Okapi, chunk_data = list dict tiap chunk
            bm25, chunk_data, _ = self._get_bm25_index(db)
        except Exception as exc:
            logger.error("BM25 index error: %s", exc)
            return []

        # Pecah query jadi token kata, BM25 bekerja di level kata bukan kalimat
        tokens = self._tokenize(query)
        if not tokens:
            return []

        # Hitung skor BM25 untuk setiap chunk terhadap token query
        # Hasilnya array angka, satu angka per chunk, urutan sesuai chunk_data
        scores = bm25.get_scores(tokens)
        # Ambil skor tertinggi sebagai pembagi normalisasi
        max_score = float(max(scores)) if len(scores) > 0 else 0.0
        # Kalau semua skor 0, artinya tidak ada chunk yang relevan sama sekali
        if max_score <= 0:
            return []

        # Urutkan indeks chunk dari skor tertinggi ke terendah, ambil top_k teratas
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = []
        for idx in top_indices:
            # Berhenti kalau skor sudah 0, chunk sisanya pasti lebih rendah
            if scores[idx] <= 0:
                break
            c = chunk_data[idx]
            # Simpan skor asli (bm25_score) dan skor ternormalisasi 0-1 (bm25_score_norm)
            # Normalisasi: skor chunk dibagi skor tertinggi, agar skala sama dengan SBERT
            results.append({**c, "bm25_score": float(scores[idx]), "bm25_score_norm": float(scores[idx] / max_score)})
        return results

    # ------------------------------------------------------------------
    # SBERT
    # ------------------------------------------------------------------
    def _retrieve_sbert_chunks(self, query: str, db: Session, top_k: int) -> List:
        """Retrieve top-K chunk via pgvector cosine similarity."""
        # Ubah teks query menjadi vektor 512 dimensi menggunakan model SBERT
        query_embedding = self.embedder.embed_text(query)

        # Daftar kata yang kalau ada di path file maka file itu dilewati, misal "software", "dotnet"
        exclude_keywords = [k.strip().lower() for k in settings.SEARCH_EXCLUDE_PATH_KEYWORDS if k and k.strip()]
        exclude_clauses = ""

        # Siapkan parameter untuk query SQL, termasuk vektor query dan jumlah hasil yang diminta
        params = {
            "query_embedding": str(query_embedding),
            "top_k": top_k,
            "include_gdrive": bool(settings.SEARCH_INCLUDE_GDRIVE),
        }

        # Bangun klausa SQL tambahan untuk setiap kata yang ingin dikecualikan dari path
        for i, kw in enumerate(exclude_keywords):
            key = f"exclude_kw_{i}"
            exclude_clauses += f"\n  AND LOWER(f.file_path) NOT LIKE :{key}"
            params[key] = f"%{kw}%"

        # Query ke PostgreSQL menggunakan pgvector
        # "fc.embedding <=> query_embedding" menghitung jarak kosinus antar dua vektor
        # "1 - jarak" dibalik jadi similarity: semakin kecil jaraknya, semakin tinggi skornya
        # Hasilnya diurutkan dari chunk yang paling mirip maknanya dengan query
        sql = text(f"""
            SELECT fc.id AS chunk_id, fc.chunk_text, fc.chunk_index,
                   f.id AS file_id, f.source, f.file_name, f.file_path,
                   f.file_type, f.file_size, f.last_modified,
                   1 - (fc.embedding <=> CAST(:query_embedding AS vector)) AS similarity
            FROM file_chunks fc
            JOIN files f ON fc.file_id = f.id
            WHERE fc.embedding IS NOT NULL
              AND (:include_gdrive OR f.source <> 'gdrive')
              {exclude_clauses}
            ORDER BY fc.embedding <=> CAST(:query_embedding AS vector)
            LIMIT :top_k
        """)
        return db.execute(sql, params).fetchall()

    # ------------------------------------------------------------------
    # METADATA
    # ------------------------------------------------------------------
    def _get_metadata_candidates(self, query: str, db: Session, top_k_files: int) -> Dict[int, float]:
        """Ambil kandidat file dari metadata, normalisasi skor ke 0-1."""
        # Ekstrak kata kunci inti dari query, buang kata umum seperti "cari", "file", "dimana"
        focus_terms = self._extract_focus_terms(query)

        # (tidak dipakai) Cek apakah user menyebut Google Drive dalam querynya
        gdrive_requested = self._mentions_gdrive(query)

        # Cari kandidat file yang nama atau path-nya cocok dengan query secara fuzzy
        exact_cands = self._metadata_exact_candidates(db, query, top_k_files, gdrive_requested)

        # Cari kandidat file berdasarkan kata kunci inti yang diekstrak tadi
        kw_cands = self._metadata_search_candidates(db, query, top_k_files, focus_terms, gdrive_requested)

        # Gabungkan kedua hasil, ambil skor tertinggi per file kalau file muncul di keduanya
        combined = self._combine_metadata_candidates(exact_cands, kw_cands)
        if not combined:
            return {}

        # Normalisasi skor ke rentang 0-1 agar bisa digabung dengan skor BM25 dan SBERT
        max_score = max(s for _, s in combined)
        return {fid: float(s) / max(float(max_score), 1.0) for fid, s in combined}

    # ------------------------------------------------------------------
    # HYBRID SEARCH
    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        db: Session,
        top_k_chunks: int | None = None,
        top_k_files: int | None = None,
    ) -> List[Dict]:
        """
        Hybrid search: Metadata + BM25 + SBERT dengan score fusion.

        Mode lookup  : metadata(0.35) + BM25(0.35) + SBERT(0.30)
        Mode semantic: BM25(0.40) + SBERT(0.60)
        """
        # Kalau tidak dikirim dari luar, pakai nilai default dari config
        top_k_chunks = top_k_chunks or settings.TOP_K_CHUNKS
        top_k_files = top_k_files or settings.TOP_K_FILES

        # Cek ulang di sini karena search() bisa dipanggil langsung tanpa lewat _plan_query
        file_lookup_mode = self._is_file_lookup_query(query)

        # Bersihkan query
        # Buang kata tidak penting seperti "tolong", "cari", "dimana" sebelum dikirim ke BM25 dan SBERT
        clean_query = self._strip_query_noise(query)
        embed_query = clean_query or query

        # Ambil kandidat dari 3 sumber
        # Ambil dua kali lipat chunk agar ada ruang untuk penggabungan per file
        retrieve_k = top_k_chunks * 2
        # SBERT: ubah query jadi vektor, cari chunk yang vektornya paling mirip di pgvector
        sbert_rows = self._retrieve_sbert_chunks(embed_query, db, retrieve_k)
        # BM25: cari chunk yang kata-katanya paling cocok dengan query
        bm25_results = self._retrieve_bm25_chunks(clean_query, db, retrieve_k)
        # Metadata: cari file yang nama atau path-nya mengandung kata dari query (lookup saja)
        meta_candidates: Dict[int, float] = {}
        if file_lookup_mode:
            meta_candidates = self._get_metadata_candidates(query, db, top_k_files)

        # Bangun struktur data per file dari hasil SBERT 
        # files_dict memakai file_id sebagai kunci agar semua chunk dari file yang sama terkumpul
        files_dict: Dict[int, Dict] = {}
        for row in sbert_rows:
            fid = int(row.file_id)
            if fid not in files_dict:
                files_dict[fid] = {
                    "file_id": fid, "source": row.source,
                    "file_name": row.file_name, "file_path": row.file_path,
                    "file_type": row.file_type, "file_size": row.file_size,
                    "last_modified": str(row.last_modified),
                    "_sbert_sims": [], "_bm25_sims": [], "meta_score": 0.0,
                    "relevant_chunks": [],
                }
            # Kumpulkan semua skor SBERT dari berbagai chunk milik file ini
            files_dict[fid]["_sbert_sims"].append(float(row.similarity))
            if (row.chunk_text or "").strip():
                files_dict[fid]["relevant_chunks"].append({
                    "chunk_id": row.chunk_id, "chunk_index": row.chunk_index,
                    "chunk_text": row.chunk_text,
                    "similarity": float(row.similarity),
                    "raw_similarity": float(row.similarity),
                    "boosted_similarity": float(row.similarity),
                })

        # Masukkan skor BM25 ke files_dict 
        # File yang belum ada di files_dict (hanya masuk via BM25) tetap dibuat entry-nya
        for chunk in bm25_results:
            fid = int(chunk["file_id"])
            if fid not in files_dict:
                files_dict[fid] = {
                    "file_id": fid, "source": chunk.get("source", "local"),
                    "file_name": chunk["file_name"], "file_path": chunk["file_path"],
                    "file_type": chunk.get("file_type", ""),
                    "file_size": chunk.get("file_size", 0),
                    "last_modified": str(chunk.get("last_modified", "")),
                    "_sbert_sims": [], "_bm25_sims": [], "meta_score": 0.0,
                    "relevant_chunks": [],
                }
            # bm25_score_norm adalah skor BM25 yang sudah dinormalisasi ke rentang 0-1
            files_dict[fid]["_bm25_sims"].append(chunk["bm25_score_norm"])

        # Masukkan skor metadata ke files_dict (hanya mode lookup) 
        # File yang hanya ditemukan via metadata tapi belum ada di files_dict tetap diambil dari DB
        for fid, meta_score in meta_candidates.items():
            if fid not in files_dict:
                file_meta = db.query(File).filter(File.id == fid).first()
                if not file_meta:
                    continue
                files_dict[fid] = {
                    "file_id": fid, "source": file_meta.source or "local",
                    "file_name": file_meta.file_name, "file_path": file_meta.file_path,
                    "file_type": file_meta.file_type or "",
                    "file_size": file_meta.file_size or 0,
                    "last_modified": str(file_meta.last_modified or ""),
                    "_sbert_sims": [], "_bm25_sims": [], "meta_score": 0.0,
                    "relevant_chunks": [],
                }
            files_dict[fid]["meta_score"] = float(meta_score)

        # Ambil chunk untuk file yang belum punya chunk sama sekali
        # Terjadi kalau file masuk via BM25 atau metadata tapi tidak ada di hasil SBERT
        # Ubah query jadi vektor 512 dimensi, dipakai untuk cari chunk paling relevan per file
        query_embedding = self.embedder.embed_text(embed_query)
        for fid, fdata in files_dict.items():
            # Hanya proses file yang belum punya chunk sama sekali
            if not fdata["relevant_chunks"]:
                # Query ke DB: ambil 3 chunk dari file ini yang paling mirip vektornya dengan query
                chunks = self._fetch_top_chunks_for_file(db, fid, str(query_embedding), limit=3)
                # Simpan chunk ke relevant_chunks, tambahkan field raw dan boosted similarity
                fdata["relevant_chunks"] = [
                    {**c, "raw_similarity": c["similarity"], "boosted_similarity": c["similarity"]}
                    for c in chunks
                ]
                # Isi juga _sbert_sims agar score fusion di langkah 7 punya data SBERT untuk file ini
                fdata["_sbert_sims"] = [c["similarity"] for c in fdata["relevant_chunks"]]

        # Hitung skor akhir per file (score fusion)
        for fdata in files_dict.values():
            # Ambil rata-rata dari 3 skor SBERT tertinggi milik file ini
            sbert_top3 = sorted(fdata["_sbert_sims"], reverse=True)[:3]
            sbert_agg = sum(sbert_top3) / len(sbert_top3) if sbert_top3 else 0.0
            sbert_max = max(fdata["_sbert_sims"]) if fdata["_sbert_sims"] else 0.0

            # Ambil rata-rata dari 3 skor BM25 tertinggi milik file ini
            bm25_top3 = sorted(fdata["_bm25_sims"], reverse=True)[:3]
            bm25_agg = sum(bm25_top3) / len(bm25_top3) if bm25_top3 else 0.0

            meta_score = fdata["meta_score"]

            # Gabungkan skor dengan bobot sesuai mode
            if file_lookup_mode:
                rank_score = (
                    float(settings.HYBRID_LOOKUP_META_WEIGHT) * meta_score
                    + float(settings.HYBRID_LOOKUP_BM25_WEIGHT) * bm25_agg
                    + float(settings.HYBRID_LOOKUP_SBERT_WEIGHT) * sbert_agg
                )
            else:
                rank_score = (
                    float(settings.HYBRID_SEMANTIC_BM25_WEIGHT) * bm25_agg
                    + float(settings.HYBRID_SEMANTIC_SBERT_WEIGHT) * sbert_agg
                )

            fdata["rank_score"] = float(rank_score)
            # max_similarity dipakai UI untuk tampilkan skor relevansi
            fdata["max_similarity"] = float(rank_score)
            fdata["avg_top3"] = float(sbert_agg)
            fdata["max_sim"] = float(sbert_max)
            fdata["bm25_score"] = float(bm25_agg)

            # Urutkan chunk tiap file dari yang paling relevan
            fdata["relevant_chunks"] = sorted(
                fdata["relevant_chunks"], key=lambda c: c["similarity"], reverse=True
            )
            # Hapus field sementara yang hanya dipakai selama kalkulasi, tidak perlu dikirim ke UI
            del fdata["_sbert_sims"]
            del fdata["_bm25_sims"]

        # Urutkan semua file by skor dan ambil top-K
        return sorted(files_dict.values(), key=lambda x: x["rank_score"], reverse=True)[:top_k_files]

    def search_raw_chunks(
        self,
        query: str,
        db: Session,
        top_k_chunks: int | None = None,
    ) -> List[Dict]:
        """Ambil top-K chunk paling mirip (semantic), tanpa agregasi per file."""
        top_k_chunks = top_k_chunks or settings.TOP_K_CHUNKS
        query_embedding = self.embedder.embed_text(query)

        exclude_keywords = [
            k.strip().lower()
            for k in settings.SEARCH_EXCLUDE_PATH_KEYWORDS
            if k and k.strip()
        ]
        exclude_clauses = ""
        params = {
            "query_embedding": str(query_embedding),
            "top_k": top_k_chunks,
            "include_gdrive": bool(settings.SEARCH_INCLUDE_GDRIVE),
        }
        for i, keyword in enumerate(exclude_keywords):
            key = f"exclude_kw_{i}"
            exclude_clauses += f"\n              AND LOWER(f.file_path) NOT LIKE :{key}"
            params[key] = f"%{keyword}%"

        sql = text(f"""
            SELECT
                fc.id          AS chunk_id,
                fc.chunk_text,
                fc.chunk_index,
                f.id            AS file_id,
                f.file_name,
                f.file_path,
                f.file_type,
                1 - (fc.embedding <=> CAST(:query_embedding AS vector)) AS similarity
            FROM file_chunks fc
            JOIN files f ON fc.file_id = f.id
            WHERE fc.embedding IS NOT NULL
              AND (:include_gdrive OR f.source <> 'gdrive')
              {exclude_clauses}
            ORDER BY fc.embedding <=> CAST(:query_embedding AS vector)
            LIMIT :top_k
        """)

        rows = db.execute(sql, params).fetchall()
        return [
            {
                "chunk_id": row.chunk_id,
                "chunk_index": row.chunk_index,
                "chunk_text": row.chunk_text,
                "similarity": float(row.similarity),
                "file_id": row.file_id,
                "file_name": row.file_name,
                "file_path": row.file_path,
                "file_type": row.file_type,
            }
            for row in rows
        ]
