"""
Evaluasi RAG — Menghitung BLEU score untuk jawaban yang dihasilkan LLM.

BLEU (Bilingual Evaluation Understudy) mengukur seberapa mirip
jawaban yang dihasilkan sistem dengan jawaban referensi.

Cara penggunaan:
  1. Siapkan rag_ground_truth.json (lihat format di bawah).
  2. Jalankan: python -m evaluation.evaluate_rag
"""

import json
import os
import time
import logging
import requests
from typing import List, Dict
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_BASE_URL = "http://localhost:8000/api"

# =====================================================================
# CONTOH FORMAT GROUND TRUTH RAG
# =====================================================================
SAMPLE_RAG_GROUND_TRUTH = [
    {
        "query": "Apa visi program studi?",
        "reference_answer": (
            "Visi program studi adalah menjadi program studi unggulan "
            "yang menghasilkan lulusan kompeten di bidang teknologi informasi."
        ),
    },
    {
        "query": "Siapa kepala program studi saat ini?",
        "reference_answer": (
            "Kepala program studi saat ini adalah Dr. Ahmad Sudirman, M.Kom "
            "berdasarkan SK Rektor tahun 2024."
        ),
    },
]


# =====================================================================
# BLEU SCORE IMPLEMENTATION
# =====================================================================
def get_ngrams(tokens: List[str], n: int) -> Counter:
    """Hitung n-grams dari list token."""
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def bleu_score(
    reference: str,
    hypothesis: str,
    max_n: int = 4,
    weights: List[float] | None = None,
) -> Dict:
    """
    Hitung BLEU score antara reference dan hypothesis.

    Args:
        reference: Jawaban referensi (ground truth)
        hypothesis: Jawaban yang dihasilkan sistem
        max_n: Maksimum n-gram (default: 4 → BLEU-1 sampai BLEU-4)
        weights: Bobot untuk setiap n-gram (default: uniform)

    Returns:
        Dict berisi BLEU-1 sampai BLEU-{max_n} dan BLEU (gabungan)
    """
    if weights is None:
        weights = [1.0 / max_n] * max_n

    # Tokenisasi sederhana (lowercase + split)
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()

    if not hyp_tokens or not ref_tokens:
        return {f"bleu_{i+1}": 0.0 for i in range(max_n)} | {"bleu": 0.0}

    # Brevity penalty
    bp = min(1.0, len(hyp_tokens) / len(ref_tokens)) if ref_tokens else 0.0

    # Hitung precision per n-gram
    precisions = {}
    log_avg = 0.0

    for n in range(1, max_n + 1):
        ref_ngrams = get_ngrams(ref_tokens, n)
        hyp_ngrams = get_ngrams(hyp_tokens, n)

        # Clipped count
        clipped = 0
        total = 0
        for ngram, count in hyp_ngrams.items():
            clipped += min(count, ref_ngrams.get(ngram, 0))
            total += count

        precision = clipped / total if total > 0 else 0.0
        precisions[f"bleu_{n}"] = precision

        if precision > 0:
            import math
            log_avg += weights[n - 1] * math.log(precision)
        else:
            # Jika ada precision = 0, BLEU keseluruhan = 0
            precisions["bleu"] = 0.0
            return precisions

    import math
    precisions["bleu"] = bp * math.exp(log_avg)

    return precisions


# =====================================================================
# EVALUASI UTAMA
# =====================================================================
def evaluate_rag(ground_truth: List[Dict]) -> Dict:
    """
    Jalankan evaluasi RAG terhadap backend API.

    Args:
        ground_truth: List berisi dict {query, reference_answer}

    Returns:
        Dict berisi hasil BLEU per query dan rata-rata.
    """
    all_results = []

    for idx, gt in enumerate(ground_truth):
        query = gt["query"]
        reference = gt["reference_answer"]

        logger.info(f"[{idx+1}/{len(ground_truth)}] Evaluating: {query}")

        # Panggil API chat
        start_time = time.time()
        try:
            resp = requests.post(
                f"{API_BASE_URL}/chat",
                json={"message": query},
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.error(f"  Error: {e}")
            continue

        elapsed_ms = int((time.time() - start_time) * 1000)

        hypothesis = data.get("answer", "")
        scores = bleu_score(reference, hypothesis)

        result = {
            "query": query,
            "reference_answer": reference,
            "generated_answer": hypothesis,
            "response_time_ms": elapsed_ms,
            **scores,
        }
        all_results.append(result)

        logger.info(
            f"  BLEU={scores.get('bleu', 0):.4f}, "
            f"BLEU-1={scores.get('bleu_1', 0):.4f}, "
            f"Time={elapsed_ms}ms"
        )

    # Hitung rata-rata
    if all_results:
        avg_metrics = {
            "avg_bleu": sum(r["bleu"] for r in all_results) / len(all_results),
            "avg_bleu_1": sum(r["bleu_1"] for r in all_results) / len(all_results),
            "avg_bleu_2": sum(r["bleu_2"] for r in all_results) / len(all_results),
            "avg_bleu_3": sum(r["bleu_3"] for r in all_results) / len(all_results),
            "avg_bleu_4": sum(r["bleu_4"] for r in all_results) / len(all_results),
            "avg_response_time_ms": sum(
                r["response_time_ms"] for r in all_results
            ) / len(all_results),
        }
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
    gt_path = os.path.join(os.path.dirname(__file__), "rag_ground_truth.json")

    if os.path.exists(gt_path):
        with open(gt_path, "r", encoding="utf-8") as f:
            ground_truth = json.load(f)
        logger.info(f"Loaded RAG ground truth dari {gt_path}")
    else:
        ground_truth = SAMPLE_RAG_GROUND_TRUTH
        logger.info("Menggunakan sample ground truth (buat rag_ground_truth.json untuk data sendiri)")

    # Jalankan evaluasi
    results = evaluate_rag(ground_truth)

    # Tampilkan hasil
    print("\n" + "=" * 70)
    print("  HASIL EVALUASI RAG (BLEU Score)")
    print("=" * 70)

    avg = results["average_metrics"]
    if avg:
        print(f"\n  Total queries: {results['total_queries']}")
        print(f"  BLEU (combined): {avg.get('avg_bleu', 0):.4f}")
        print(f"  BLEU-1: {avg.get('avg_bleu_1', 0):.4f}")
        print(f"  BLEU-2: {avg.get('avg_bleu_2', 0):.4f}")
        print(f"  BLEU-3: {avg.get('avg_bleu_3', 0):.4f}")
        print(f"  BLEU-4: {avg.get('avg_bleu_4', 0):.4f}")
        print(f"  Avg Response Time: {avg.get('avg_response_time_ms', 0):.0f} ms")

        # Detail per query
        print(f"\n  {'Query':<40} {'BLEU':>8} {'BLEU-1':>8} {'Time(ms)':>10}")
        print("  " + "-" * 68)
        for r in results["per_query_results"]:
            q = r["query"][:38]
            print(
                f"  {q:<40} {r['bleu']:>8.4f} {r['bleu_1']:>8.4f} "
                f"{r['response_time_ms']:>10}"
            )
    else:
        print("  Tidak ada hasil.")

    # Simpan hasil
    output_path = os.path.join(os.path.dirname(__file__), "rag_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n  Hasil lengkap disimpan ke: {output_path}")
    print("=" * 70)
