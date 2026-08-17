#!/usr/bin/env python3
"""
RAG Retrieval 품질 평가 스크립트

사용 예시:
  python scripts/evaluate_retrieval.py --dataset eval/qa_eval_set.jsonl --k 6
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rag.qa_pipeline import RAGPipeline
from src.rag.embedder import get_embed_model


def load_eval_set(path: Path) -> list[dict]:
    samples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    return samples


def evaluate_sample(pipeline: RAGPipeline, sample: dict, k: int, force_keyword_fallback: bool = False) -> dict:
    query = sample["query"]
    expected_years = sample.get("expected_years", [])
    expected_sections = sample.get("expected_sections", [])
    must_keywords = sample.get("must_include_keywords", [])
    year_mode = sample.get("year_mode", "any")

    fallback_mode = "none"
    if force_keyword_fallback:
        contexts = pipeline._keyword_search(query, k=k, year_filter=None)
        fallback_mode = "keyword_only"
    else:
        try:
            contexts = pipeline.get_contexts(query, retrieve_k=k)
        except RuntimeError as exc:
            # 오프라인 환경에서 임베딩 모델 로드가 불가능하면 키워드 검색으로 평가를 계속한다.
            if "임베딩 모델 로드 실패" not in str(exc):
                raise
            contexts = pipeline._keyword_search(query, k=k, year_filter=None)
            fallback_mode = "keyword_only"
    years = {int(c["year"]) for c in contexts}
    sections = {str(c["section"]) for c in contexts}
    joined = " ".join(str(c["text"]) for c in contexts)

    if not expected_years:
        year_hit = True
    elif year_mode == "all":
        year_hit = set(expected_years).issubset(years)
    else:
        year_hit = bool(set(expected_years) & years)

    if not expected_sections:
        section_hit = True
    else:
        section_hit = bool(set(expected_sections) & sections)

    keyword_hit = all(kw in joined for kw in must_keywords) if must_keywords else True

    return {
        "id": sample.get("id", ""),
        "query": query,
        "year_hit": year_hit,
        "section_hit": section_hit,
        "keyword_hit": keyword_hit,
        "pass": year_hit and section_hit and keyword_hit,
        "retrieved_years": sorted(years),
        "retrieved_sections": sorted(sections),
        "contexts": contexts,
        "fallback_mode": fallback_mode,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG retrieval evaluator")
    parser.add_argument("--dataset", type=Path, default=Path("eval/qa_eval_set.jsonl"))
    parser.add_argument("--k", type=int, default=4)
    args = parser.parse_args()

    if args.k < 1:
        raise ValueError("--k must be >= 1")
    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")

    samples = load_eval_set(args.dataset)
    pipeline = RAGPipeline()

    force_keyword_fallback = False
    try:
        get_embed_model()
    except Exception as exc:  # noqa: BLE001
        force_keyword_fallback = True
        print(f"[WARN] Embedding model unavailable, switching to keyword-only eval: {exc}")

    rows = [evaluate_sample(pipeline, s, args.k, force_keyword_fallback=force_keyword_fallback) for s in samples]
    total = len(rows)
    passed = sum(1 for r in rows if r["pass"])
    year_acc = sum(1 for r in rows if r["year_hit"]) / total if total else 0.0
    section_acc = sum(1 for r in rows if r["section_hit"]) / total if total else 0.0
    keyword_acc = sum(1 for r in rows if r["keyword_hit"]) / total if total else 0.0

    print("=" * 70)
    print(f"Dataset: {args.dataset} | k={args.k}")
    print(f"Total: {total} | Pass: {passed} ({(passed / total * 100) if total else 0:.1f}%)")
    print(f"Year hit:    {year_acc * 100:.1f}%")
    print(f"Section hit: {section_acc * 100:.1f}%")
    print(f"Keyword hit: {keyword_acc * 100:.1f}%")
    print("=" * 70)

    failed = [r for r in rows if not r["pass"]]
    fallback_count = sum(1 for r in rows if r["fallback_mode"] != "none")
    if fallback_count:
        print(f"Fallback mode used on {fallback_count}/{total} samples (keyword_only).")

    if failed:
        print("Failed samples:")
        for r in failed:
            print(f"- {r['id']} | {r['query']}")
            print(f"  years={r['retrieved_years']}, sections={r['retrieved_sections']}")
    else:
        print("All samples passed.")


if __name__ == "__main__":
    main()
