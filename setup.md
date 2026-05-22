# 🔍 Chatbot Pencarian File — Setup Guide

Sistem chatbot pencarian file berbasis **Sentence-BERT (SBERT)** dan **Retrieval-Augmented Generation (RAG)**.

---

## 📋 Arsitektur Sistem

```
┌─────────────────┐     HTTP      ┌─────────────────┐
│   Streamlit UI  │ ◄──────────► │  FastAPI Backend │
│  (frontend/)    │               │  (backend/)      │
└─────────────────┘               └────────┬─────────┘
                                           │
                     ┌─────────────────────┼─────────────────────┐
                     │                     │                     │
              ┌──────▼──────┐      ┌───────▼───────┐     ┌──────▼──────┐
              │    SBERT    │      │  PostgreSQL   │     │   Ollama    │
              │  Embedding  │      │  + pgvector   │     │    LLM      │
              └─────────────┘      └───────────────┘     └─────────────┘
```

---

## 🛠️ Prerequisites

1. **Python 3.10+** — [Download](https://www.python.org/downloads/)
2. **Docker Desktop** — [Download](https://www.docker.com/products/docker-desktop/)
3. **Ollama** — [Download](https://ollama.com/download)
4. **Git** (opsional)

---

## 🚀 Langkah-Langkah Setup

### 1️⃣ Jalankan PostgreSQL + pgvector (via Docker)

```bash
cd c:\SKRIPSI
docker-compose up -d
```

Ini akan menjalankan PostgreSQL di port `5432` dengan ekstensi pgvector.

Verifikasi:
```bash
docker ps
# Harus terlihat container 'skripsi_postgres' berjalan
```

### 2️⃣ Install & Jalankan Ollama LLM

```bash
# Install Ollama dari https://ollama.com/download

# Download model (pilih salah satu):
ollama pull qwen2.5:7b              # Rekomendasi (multilingual, 4.7GB)
# ATAU
ollama pull mistral-nemo             # Alternatif
# ATAU
ollama pull llama3.1:8b              # Alternatif

# Ollama akan otomatis berjalan di http://localhost:11434
```

### 3️⃣ Setup Backend (FastAPI)

```bash
cd c:\SKRIPSI\backend

# Buat virtual environment
python -m venv venv
venv\Scripts\activate              # Windows

# Install dependencies
pip install -r requirements.txt
```

### 4️⃣ Konfigurasi Environment

Edit file `backend/.env`:

```env
# Sesuaikan path direktori yang ingin di-crawl
CRAWL_DIRECTORIES=["C:/Users/NamaAnda/Documents","D:/Data Skripsi"]

# Sesuaikan model Ollama yang sudah di-download
OLLAMA_MODEL=qwen2.5:7b
```

### 5️⃣ Jalankan Backend

```bash
cd c:\SKRIPSI\backend
venv\Scripts\activate
uvicorn app.main:app --reload --port 8000
```

Buka http://localhost:8000/docs untuk melihat dokumentasi API.

### 6️⃣ Setup & Jalankan Frontend (Streamlit)

Buka terminal baru:

```bash
cd c:\SKRIPSI\frontend

# Buat virtual environment terpisah (opsional, bisa pakai yang sama)
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Jalankan
streamlit run app.py
```

Streamlit akan terbuka di http://localhost:8501

---

## 📂 Cara Menggunakan

### Indexing File (Pertama Kali)

1. Buka Streamlit UI di browser
2. Di sidebar kiri, masukkan path direktori yang berisi file Anda
3. Klik **🔄 Full Index** untuk index semua file
4. Tunggu proses selesai (tergantung jumlah file)

### Chatting / Pencarian

Setelah indexing selesai, ketik pertanyaan di chat:
- `"cari file laporan bulanan"`
- `"apa isi dokumen akreditasi?"`
- `"file apa saja yang membahas tentang visi misi?"`

### Re-Indexing (Jika Ada File Baru/Berubah)

Klik **⚡ Incremental** di sidebar — hanya file baru/berubah yang diproses.

---

## 📁 Struktur Project

```
SKRIPSI/
├── docker-compose.yml              # PostgreSQL + pgvector
├── backend/
│   ├── .env                        # Konfigurasi environment
│   ├── requirements.txt            # Dependencies backend
│   └── app/
│       ├── main.py                 # FastAPI entry point
│       ├── config.py               # Settings/konfigurasi
│       ├── database.py             # Koneksi database
│       ├── models/
│       │   └── file_model.py       # SQLAlchemy models (files, chunks, chat)
│       ├── services/
│       │   ├── crawler.py          # File crawler (traversal rekursif)
│       │   ├── extractor.py        # Ekstraksi teks (pdf, docx, xlsx, dll)
│       │   ├── embedder.py         # SBERT embedding + chunking
│       │   ├── searcher.py         # Cosine similarity search (pgvector)
│       │   ├── rag.py              # RAG pipeline (Ollama LLM)
│       │   └── indexer.py          # Orchestrator indexing
│       ├── api/
│       │   ├── schemas.py          # Pydantic request/response models
│       │   └── routes.py           # FastAPI endpoints
│       └── utils/
│           └── helpers.py          # Utility functions
├── frontend/
│   ├── requirements.txt            # Dependencies frontend
│   └── app.py                      # Streamlit chatbot UI
└── evaluation/
    ├── evaluate.py                 # Recall@K, Precision@K, MRR
    ├── evaluate_rag.py             # BLEU score
    ├── ground_truth.json           # (Anda buat sendiri)
    └── rag_ground_truth.json       # (Anda buat sendiri)
```

---

## 🔗 API Endpoints

| Method | Endpoint                  | Deskripsi                          |
|--------|---------------------------|------------------------------------|
| POST   | `/api/chat`               | Chat + RAG (cari file & jawab)     |
| POST   | `/api/search`             | Pencarian saja (tanpa RAG)         |
| POST   | `/api/index`              | Full indexing                      |
| POST   | `/api/index/incremental`  | Incremental indexing               |
| GET    | `/api/stats`              | Statistik database                 |
| GET    | `/api/history`            | Riwayat chat                       |

---

## 📊 Evaluasi

### Retrieval Metrics (Recall@K, Precision@K, MRR)

1. Buat file `evaluation/ground_truth.json`:
```json
[
  {
    "query": "laporan akreditasi",
    "relevant": ["Laporan Evaluasi Diri.pdf", "Borang Akreditasi.docx"]
  }
]
```

2. Jalankan:
```bash
cd c:\SKRIPSI
python -m evaluation.evaluate
```

### RAG Metrics (BLEU Score)

1. Buat file `evaluation/rag_ground_truth.json`:
```json
[
  {
    "query": "Apa visi program studi?",
    "reference_answer": "Visi program studi adalah..."
  }
]
```

2. Jalankan:
```bash
cd c:\SKRIPSI
python -m evaluation.evaluate_rag
```

---

## ⚠️ Troubleshooting

| Masalah | Solusi |
|---------|--------|
| `Connection refused` ke PostgreSQL | Pastikan Docker berjalan: `docker-compose up -d` |
| `Connection refused` ke Ollama | Pastikan Ollama berjalan: `ollama serve` |
| Model SBERT lambat di-download | Normal untuk pertama kali (~250MB). Butuh internet. |
| Error "extension vector does not exist" | pgvector belum terinstall. Gunakan image `pgvector/pgvector:pg16` |
| File tidak ter-extract | Pastikan file tidak terkunci password / terenkripsi |
