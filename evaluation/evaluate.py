"""
Evaluasi Retrieval — Menghitung metrik pencarian dokumen:
  - Recall@K    : Proporsi dokumen relevan yang berhasil ditemukan di top-K
  - Precision@K : Proporsi dokumen di top-K yang benar-benar relevan
  - MRR         : Mean Reciprocal Rank — rata-rata 1/rank dokumen relevan pertama

Cara penggunaan:
  1. Siapkan ground_truth dalam format JSON (lihat contoh di bawah).
  2. Jalankan: python -m evaluation.evaluate
"""

import json
import time
import requests
import logging
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_BASE_URL = "http://localhost:8000/api"

# =====================================================================
# CONTOH FORMAT GROUND TRUTH
# =====================================================================
# Setiap entry berisi:
#   - query       : pertanyaan/pencarian pengguna
#   - relevant     : list nama file yang dianggap relevan (ground truth)
#
# Simpan sebagai evaluation/ground_truth.json

SAMPLE_GROUND_TRUTH = [
    {
        "query": "laporan akreditasi program studi",
        "relevant": [
            "Laporan Evaluasi Diri.pdf",
            "Borang Akreditasi 2024.docx",
            "Data Pendukung Akreditasi.xlsx",
        ],
    },
    {
        "query": "tugas dan tanggung jawab kepala bidang",
        "relevant": [
            "Uraian Tugas Kepala Bidang.pdf",
            "SK Pembagian Tugas.docx",
        ],
    },
    {
        "query": "jadwal kuliah semester ganjil",
        "relevant": [
            "Jadwal Kuliah Ganjil 2024-2025.xlsx",
            "Pengumuman Jadwal.pdf",
        ],
    },
]


# =====================================================================
# METRICS
# =====================================================================
def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """
    Recall@K = |relevant ∩ retrieved[:k]| / |relevant|

    Berapa proporsi dokumen relevan yang berhasil masuk top-K.
    """
    if not relevant:
        return 0.0

    retrieved_at_k = set(r.lower() for r in retrieved[:k])
    relevant_set = set(r.lower() for r in relevant)

    found = retrieved_at_k & relevant_set
    return len(found) / len(relevant_set)


def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """
    Precision@K = |relevant ∩ retrieved[:k]| / K

    Berapa proporsi dari top-K yang benar-benar relevan.
    """
    if k == 0:
        return 0.0

    retrieved_at_k = set(r.lower() for r in retrieved[:k])
    relevant_set = set(r.lower() for r in relevant)

    found = retrieved_at_k & relevant_set
    return len(found) / k


def reciprocal_rank(retrieved: List[str], relevant: List[str]) -> float:
    """
    Reciprocal Rank = 1 / rank dokumen relevan pertama yang ditemukan.
    Jika tidak ditemukan, return 0.
    """
    relevant_set = set(r.lower() for r in relevant)

    for i, doc in enumerate(retrieved):
        if doc.lower() in relevant_set:
            return 1.0 / (i + 1)

    return 0.0


def mean_reciprocal_rank(results: List[Dict]) -> float:
    """
    MRR = rata-rata dari reciprocal_rank semua query.
    """
    if not results:
        return 0.0

    rr_sum = sum(r["reciprocal_rank"] for r in results)
    return rr_sum / len(results)


# =====================================================================
# EVALUASI UTAMA
# =====================================================================
def evaluate_retrieval(
    ground_truth: List[Dict],
    k_values: List[int] = [1, 3, 5, 10],
) -> Dict:
    """
    Jalankan evaluasi retrieval terhadap backend API.

    Args:
        ground_truth: List berisi dict {query, relevant}
        k_values: List nilai K untuk Recall@K dan Precision@K

    Returns:
        Dict berisi hasil evaluasi per query dan rata-rata keseluruhan.
    """
    all_results = []

    for idx, gt in enumerate(ground_truth):
        query = gt["query"]
        relevant = gt["relevant"]

        logger.info(f"[{idx+1}/{len(ground_truth)}] Evaluating: {query}")

        # Panggil API search
        start_time = time.time()
        try:
            resp = requests.post(
                f"{API_BASE_URL}/search",
                json={"query": query, "top_k": max(k_values)},
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.error(f"  Error: {e}")
            continue

        elapsed_ms = int((time.time() - start_time) * 1000)

        # Ambil nama file yang di-retrieve
        retrieved_names = [r["file_name"] for r in data.get("results", [])]

        # Hitung metrik untuk setiap K
        metrics = {}
        for k in k_values:
            metrics[f"recall@{k}"] = recall_at_k(retrieved_names, relevant, k)
            metrics[f"precision@{k}"] = precision_at_k(retrieved_names, relevant, k)

        rr = reciprocal_rank(retrieved_names, relevant)

        result = {
            "query": query,
            "relevant_files": relevant,
            "retrieved_files": retrieved_names,
            "reciprocal_rank": rr,
            "response_time_ms": elapsed_ms,
            **metrics,
        }
        all_results.append(result)

        logger.info(f"  RR={rr:.4f}, Time={elapsed_ms}ms")
        for k in k_values:
            logger.info(
                f"  Recall@{k}={metrics[f'recall@{k}']:.4f}, "
                f"Precision@{k}={metrics[f'precision@{k}']:.4f}"
            )

    # Hitung rata-rata keseluruhan
    if all_results:
        avg_metrics = {}
        for k in k_values:
            avg_metrics[f"avg_recall@{k}"] = sum(
                r[f"recall@{k}"] for r in all_results
            ) / len(all_results)
            avg_metrics[f"avg_precision@{k}"] = sum(
                r[f"precision@{k}"] for r in all_results
            ) / len(all_results)

        avg_metrics["MRR"] = mean_reciprocal_rank(all_results)
        avg_metrics["avg_response_time_ms"] = sum(
            r["response_time_ms"] for r in all_results
        ) / len(all_results)
    else:
        avg_metrics = {}

    return {
        "per_query_results": all_results,
        "average_metrics": avg_metrics,
        "total_queries": len(all_results),
    }


# =====================================================================
# MAIN
# =====================================================================
if __name__ == "__main__":
    import os

    # Coba load ground truth dari file, fallback ke sample
    gt_path = os.path.join(os.path.dirname(__file__), "ground_truth.json")

    if os.path.exists(gt_path):
        with open(gt_path, "r", encoding="utf-8") as f:
            ground_truth = json.load(f)
        logger.info(f"Loaded ground truth dari {gt_path}")
    else:
        ground_truth = SAMPLE_GROUND_TRUTH
        logger.info("Menggunakan sample ground truth (buat ground_truth.json untuk data sendiri)")

    # Jalankan evaluasi
    results = evaluate_retrieval(ground_truth, k_values=[1, 3, 5, 10])

    # Tampilkan hasil
    print("\n" + "=" * 70)
    print("  HASIL EVALUASI RETRIEVAL")
    print("=" * 70)

    avg = results["average_metrics"]
    if avg:
        print(f"\n  Total queries: {results['total_queries']}")
        print(f"  MRR: {avg.get('MRR', 0):.4f}")
        print(f"  Avg Response Time: {avg.get('avg_response_time_ms', 0):.0f} ms")
        print()
        for k in [1, 3, 5, 10]:
            r = avg.get(f"avg_recall@{k}", 0)
            p = avg.get(f"avg_precision@{k}", 0)
            print(f"  Recall@{k}: {r:.4f}    Precision@{k}: {p:.4f}")
    else:
        print("  Tidak ada hasil.")

    # Simpan hasil ke file
    output_path = os.path.join(os.path.dirname(__file__), "retrieval_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n  Hasil lengkap disimpan ke: {output_path}")
    print("=" * 70)
