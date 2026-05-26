"""
RAG Service — Retrieval-Augmented Generation.

Menggunakan LLM (via Ollama) untuk menghasilkan jawaban kontekstual
berdasarkan dokumen yang ditemukan oleh SearchService.

Alur:
  1. Bangun konteks dari chunk-chunk relevan.
  2. Buat prompt yang menginstruksikan LLM menjawab berdasarkan konteks.
  3. Panggil Ollama API → dapatkan jawaban.
"""

import logging
import requests
import json
import re
from typing import List, Dict

from app.config import settings

logger = logging.getLogger(__name__)


class RAGService:
    """Retrieval-Augmented Generation menggunakan Ollama LLM."""

    def __init__(self):
        self.base_url = settings.OLLAMA_BASE_URL
        self.model = settings.OLLAMA_MODEL

    def plan_query_for_retrieval(self, query: str) -> Dict[str, object]:
        """Rencanakan query sebelum retrieval: rewrite, keyword extraction, dan intent."""
        raw_query = (query or "").strip()
        if not raw_query:
            return {
                "original_query": "",
                "rewritten_query": "",
                "keywords": [],
                "intent": "general",
                "used_llm": False,
            }

        rewrite_prompt = f"""
Kamu adalah query planner untuk sistem retrieval dokumen.
Ubah query user menjadi versi yang lebih retrieval-friendly.

Aturan:
- Pertahankan angka, rasio, tanggal, nama, dan istilah teknis apa adanya.
- Jangan menambah fakta baru.
- Jika query menanyakan isi dokumen, ringkas ke kata kunci inti.
- Jika query mencari file, fokus ke nama file, folder, atau istilah unik.

Kembalikan JSON valid TANPA markdown, TANPA kode blok, dengan key:
- rewritten_query (string)
- keywords (array string)
- intent (file_lookup|content_qa|general)

Query user: {raw_query}
""".strip()

        llm_payload = None
        try:
            raw = self._call_ollama(rewrite_prompt)
            llm_payload = self._parse_json_response(raw)
        except Exception as exc:
            logger.info("Query planning via LLM gagal, pakai fallback heuristik: %s", exc)

        if not llm_payload:
            return self._fallback_plan_query(raw_query)

        rewritten_query = str(llm_payload.get("rewritten_query") or raw_query).strip()
        keywords = llm_payload.get("keywords") or []
        if not isinstance(keywords, list):
            keywords = []
        keywords = [str(item).strip() for item in keywords if str(item).strip()]

        intent = str(llm_payload.get("intent") or "general").strip().lower()
        if intent not in {"file_lookup", "content_qa", "general"}:
            intent = "general"

        if not rewritten_query:
            rewritten_query = raw_query

        return {
            "original_query": raw_query,
            "rewritten_query": rewritten_query,
            "keywords": keywords,
            "intent": intent,
            "used_llm": True,
        }

    # ------------------------------------------------------------------
    # PUBLIC
    # ------------------------------------------------------------------
    def generate_response(
        self,
        query: str,
        search_results: List[Dict],
        file_lookup_mode: bool = False,
    ) -> str:
        """Hasilkan jawaban berdasarkan query + konteks dokumen."""
        if not search_results:
            return (
                "Maaf, saya tidak menemukan dokumen yang relevan dengan "
                "pertanyaan Anda. Silakan coba kata kunci lain atau pastikan "
                "file sudah di-index."
            )

        # Jika mode cari file aktif tetapi belum ada chunk teks terbaca,
        # berikan fallback berbasis metadata agar user tetap mendapat lokasi file.
        if file_lookup_mode:
            has_text_context = any(
                (chunk.get("chunk_text") or "").strip()
                for result in search_results
                for chunk in (result.get("relevant_chunks") or [])
            )
            if not has_text_context:
                return self._metadata_only_file_lookup_response(search_results)

        context = self._build_context(
            search_results,
            max_results=None,
            max_chunks=5,
        )
        prompt = self._build_prompt(query, context)

        try:
            answer = self._call_ollama(prompt)
            return self._strip_trailing_not_found(answer)
        except requests.ConnectionError:
            logger.error("Tidak dapat terhubung ke Ollama. Pastikan Ollama berjalan.")
            return (
                "⚠️ Tidak dapat terhubung ke Ollama LLM. "
                "Pastikan Ollama sudah dijalankan (`ollama serve`).\n\n"
                "Berikut file yang ditemukan berdasarkan pencarian:\n"
                + self._fallback_response(search_results)
            )
        except Exception as e:
            logger.error(f"Error saat memanggil Ollama: {e}")
            return (
                f"⚠️ Terjadi kesalahan saat menghasilkan jawaban: {e}\n\n"
                "Berikut file yang ditemukan berdasarkan pencarian:\n"
                + self._fallback_response(search_results)
            )

    def rerank_results_by_answer(self, answer: str, search_results: List[Dict]) -> List[Dict]:
        """
        Urutkan ulang hasil retrieval berdasarkan seberapa kuat dokumen mendukung jawaban final.

        Skor akhir = kombinasi lexical-overlap (jawaban vs isi chunk) + retrieval score awal.
        Pendekatan ini menjaga jawaban tetap grounded pada dokumen, sambil membuat top-1
        lebih konsisten dengan isi jawaban yang benar-benar dihasilkan model.
        """
        if not search_results:
            return []

        answer_terms = self._tokenize(answer)
        if not answer_terms:
            return search_results

        reranked: List[Dict] = []
        for result in search_results:
            doc_text = " ".join(
                chunk.get("chunk_text", "")
                for chunk in result.get("relevant_chunks", [])
            )
            doc_terms = self._tokenize(doc_text)

            overlap = len(answer_terms & doc_terms)
            coverage = overlap / max(len(answer_terms), 1)

            retrieval_score = float(result.get("max_similarity", 0.0))

            # 70% dukungan terhadap jawaban + 30% retrieval score awal.
            grounded_score = (0.70 * coverage) + (0.30 * retrieval_score)

            updated = dict(result)
            updated["grounded_score"] = float(grounded_score)
            # Pertahankan kompatibilitas UI lama: field max_similarity dipakai sebagai skor utama.
            updated["max_similarity"] = float(grounded_score)
            reranked.append(updated)

        reranked.sort(key=lambda x: x.get("grounded_score", 0.0), reverse=True)
        return reranked

    # ------------------------------------------------------------------
    # PRIVATE
    # ------------------------------------------------------------------
    def _build_context(
        self,
        search_results: List[Dict],
        max_results: int | None = None,
        max_chunks: int | None = None,
    ) -> str:
        """Gabungkan chunk-chunk dari file-file relevan sebagai konteks."""
        context_parts = []
        selected_results = search_results[:max_results] if max_results else search_results
        for i, result in enumerate(selected_results, 1):
            header = (
                f"[Dokumen {i}] {result['file_name']} "
                f"(Lokasi: {result['file_path']})"
            )
            chunks = result.get("relevant_chunks", [])
            if max_chunks:
                chunks = chunks[:max_chunks]
            chunks_text = "\n".join(chunk["chunk_text"] for chunk in chunks)
            context_parts.append(f"{header}\n{chunks_text}")

        return "\n\n---\n\n".join(context_parts)

    def _build_prompt(self, query: str, context: str) -> str:
        """Bangun prompt untuk LLM."""
        return f"""Kamu adalah asisten chatbot pencarian file.
Tugasmu adalah membantu pengguna menemukan file dan memahami isi dokumen mereka.
PENTING: Selalu jawab dalam Bahasa Indonesia. Jika isi dokumen berbahasa Inggris, terjemahkan dan jelaskan dalam Bahasa Indonesia — jangan menyalin teks asing mentah-mentah.
PENTING: Baca dan periksa SEMUA dokumen dalam konteks secara menyeluruh sebelum menyimpulkan apakah informasi tersedia atau tidak. Jangan berhenti di dokumen pertama.

INSTRUKSI:
- Jawab HANYA berdasarkan isi dokumen yang tersedia di bawah ini.
- Kamu BOLEH menganalisa, menyimpulkan, dan merangkum dari isi dokumen.
- JANGAN menambahkan fakta, angka, atau informasi yang tidak ada dalam konteks dokumen.
- JANGAN menggunakan pengetahuan di luar konteks dokumen yang diberikan, meskipun kamu mengetahuinya.
- Periksa SEMUA dokumen (Dokumen 1, 2, 3, dst.) sebelum menyimpulkan. Jangan berhenti di dokumen pertama saja.
- Gunakan informasi teks yang ada meskipun kalimat terpotong di tengah atau chunk mereferensikan gambar/diagram yang tidak tersedia. Teks yang ada tetap valid untuk dijadikan jawaban.
- Jika pengguna mencari file, sebutkan nama file, lokasi, dan ringkasan singkat isinya.
- Selalu sebutkan sumber (nama file) saat memberikan informasi.
- Jika sudah memberikan jawaban, JANGAN tambahkan kalimat "Informasi tersebut tidak ditemukan" di akhir jawaban.
- Jika SEMUA dokumen benar-benar tidak mengandung informasi yang relevan, HANYA tulis: "Informasi tersebut tidak ditemukan dalam dokumen yang tersedia." lalu BERHENTI.

=== KONTEKS DOKUMEN ===
{context}
=== AKHIR KONTEKS ===

Pertanyaan pengguna: {query}

Jawaban:"""

    def _call_ollama(self, prompt: str) -> str:
        """Panggil Ollama API untuk generate response."""
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "top_p": 0.9,
                "num_predict": 1024,
            },
        }

        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()

        result = response.json()
        return result.get("response", "Tidak ada respons dari model.")

    def _parse_json_response(self, raw: str) -> Dict[str, object]:
        """Parse JSON dari output LLM, toleran terhadap code fence atau teks ekstra."""
        if not raw:
            return {}

        candidate = raw.strip()
        if candidate.startswith("```"):
            candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
            candidate = re.sub(r"\s*```$", "", candidate)

        try:
            data = json.loads(candidate)
            return data if isinstance(data, dict) else {}
        except Exception:
            pass

        start = candidate.find("{")
        end = candidate.rfind("}")
        if start >= 0 and end > start:
            try:
                data = json.loads(candidate[start:end + 1])
                return data if isinstance(data, dict) else {}
            except Exception:
                return {}

        return {}

    def _fallback_plan_query(self, query: str) -> Dict[str, object]:
        """Fallback heuristik jika LLM planner gagal."""
        tokens = re.findall(r"[\w:.-]+", query.lower())
        stopwords = {
            "yang", "dan", "atau", "dari", "untuk", "dengan", "pada", "dalam",
            "adalah", "ini", "itu", "ke", "di", "sebagai", "karena", "juga",
            "agar", "jika", "maka", "sudah", "belum", "lebih", "kurang",
            "berapa", "apakah", "tolong", "mohon", "coba", "bisa", "saya", "anda",
            "cari", "carikan", "file", "dokumen", "folder", "lokasi", "letak",
        }

        keywords: List[str] = []
        seen = set()
        for token in tokens:
            clean = token.strip().lower()
            if len(clean) < 2 or clean in stopwords:
                continue
            if clean in seen:
                continue
            seen.add(clean)
            keywords.append(clean)

        if not keywords:
            keywords = tokens[:5]

        rewritten_query = " ".join(keywords) if keywords else query.strip()
        intent = "file_lookup" if any(k in query.lower() for k in ["file", "dokumen", "folder", "lokasi", "letak"]) else "content_qa"

        return {
            "original_query": query.strip(),
            "rewritten_query": rewritten_query,
            "keywords": keywords,
            "intent": intent,
            "used_llm": False,
        }

    def _strip_trailing_not_found(self, answer: str) -> str:
        """Hapus kalimat 'tidak ditemukan' di akhir jawaban jika jawaban sudah ada isinya."""
        NOT_FOUND = "Informasi tersebut tidak ditemukan dalam dokumen yang tersedia."
        stripped = answer.strip()
        if stripped.endswith(NOT_FOUND):
            candidate = stripped[: -len(NOT_FOUND)].strip()
            if candidate:
                return candidate
        return stripped

    def _tokenize(self, text: str) -> set[str]:
        """Tokenisasi sederhana untuk scoring dukungan jawaban."""
        if not text:
            return set()

        stopwords = {
            "yang", "dan", "atau", "dari", "untuk", "dengan", "pada", "dalam",
            "adalah", "ini", "itu", "ke", "di", "sebagai", "karena", "juga",
            "agar", "jika", "maka", "sudah", "belum", "lebih", "kurang",
        }

        tokens = {
            t for t in re.findall(r"\w+", text.lower())
            if len(t) >= 3 and t not in stopwords
        }
        return tokens

    def _fallback_response(self, search_results: List[Dict]) -> str:
        """Fallback jika LLM tidak tersedia: tampilkan daftar file saja."""
        lines = []
        for i, r in enumerate(search_results, 1):
            sim = r['max_similarity']
            lines.append(
                f"{i}. **{r['file_name']}** (similarity: {sim:.4f})\n"
                f"   📁 {r['file_path']}"
            )
        return "\n".join(lines)

    def _metadata_only_file_lookup_response(self, search_results: List[Dict]) -> str:
        """Fallback untuk mode lookup saat konten belum terbaca, berbasis metadata file."""
        lines = ["Saya menemukan file yang paling terkait berdasarkan nama/lokasi:"]
        for i, result in enumerate(search_results[:5], 1):
            file_name = result.get("file_name", "-")
            file_path = result.get("file_path", "-")
            lines.append(f"{i}. {file_name}\\n   Lokasi: {file_path}")
        return "\n".join(lines)
