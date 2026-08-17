import os
import re
import threading
import json
import torch
from threading import Thread
from typing import Generator
from functools import lru_cache
import pandas as pd
from src.rag.embedder import load_chunks_from_csv, build_embeddings, get_embed_model
from src.rag.vector_store import VectorStore

_SUPPORTED_BACKENDS = {"local", "openai", "anthropic"}
_DEFAULT_LOCAL_MODEL = "Qwen/Qwen2.5-3B-Instruct"
_DEFAULT_OPENAI_MODEL = "gpt-5.4-mini"
_DEFAULT_RETRIEVE_K = 4
_DEFAULT_MAX_NEW_TOKENS = 1024

# 단일 연도 / 단일 섹션 질문용
RAG_SYSTEM = """당신은 삼성전자 감사보고서(2014~2024) 전문 분석 어시스턴트입니다.

[답변 규칙]
1. 주어진 참고 문서만을 근거로 정확하고 간결하게 답변하세요.
2. 참고 문서에 수치가 있으면 반드시 구체적인 숫자와 단위(백만원 등)를 포함해 답변하세요.
3. 참고 문서의 텍스트는 HTML에서 추출된 것으로 '영 업 이 익'처럼 글자 사이에 공백이 있을 수 있습니다. 공백을 무시하고 정상 단어로 해석하세요.
4. 참고 문서에서 직접적 답을 찾기 어렵더라도 관련 정보를 종합해 추론 가능하면 추론 결과를 제시하세요.
5. 참고 문서와 전혀 관련 없는 질문일 때만 '해당 정보를 참고 문서에서 찾을 수 없습니다'라고 답하세요.
6. 답변 끝에 근거 연도와 섹션을 간단히 표기하세요. 예: (출처: 2020년 포괄손익)
7. 재무수치(매출, 영업이익, 자산 등) 질문 시, 포괄손익계산서·재무상태표·현금흐름표에 직접 기재된 수치를 최우선으로 사용하세요. 주석에 포함된 종속기업·관계기업의 요약 재무정보는 삼성전자 전체 수치가 아니므로 혼동하지 마세요.
8. 답변은 핵심만 간결하게 작성하세요. 참고 문서의 원문을 그대로 길게 인용하지 말고 핵심 사실을 요약하여 서술하세요."""

# 멀티연도 비교·트렌드 질문용
RAG_SYSTEM_MULTI = """당신은 삼성전자 감사보고서(2014~2024) 전문 분석 어시스턴트입니다.
주어진 참고 문서들은 여러 연도의 자료입니다.

[답변 규칙]
1. 연도별로 정보를 구분하여 시계열 흐름이 드러나도록 답변하세요.
2. 변화·트렌드를 설명할 때는 연도 오름차순으로 서술하세요.
3. 모든 연도를 개별 나열하지 마세요. 연도당 1~2문장 이내로 핵심만 서술하고, 변화가 두드러지는 연도를 중심으로 설명하세요.
4. 수치가 있으면 반드시 숫자와 단위를 포함하세요.
5. 참고 문서의 텍스트는 HTML 추출본으로 글자 사이 공백이 있을 수 있습니다. 정상 단어로 해석하세요.
6. 참고 문서에 없는 내용은 추측하지 마세요. 데이터가 없는 연도는 '자료 없음'으로 표기하세요.
7. '변경'이나 '변화' 여부를 물을 경우, 단순히 연도가 경과한 것이 아니라 실제 내용(예: 감사의견 등)이 이전과 달라진 시점을 의미합니다. 모든 연도의 내용이 동일하다면 '변경된 연도가 없습니다'라고 명확히 답변하세요.
8. 재무수치(매출, 영업이익, 자산 등) 질문 시, 포괄손익계산서·재무상태표·현금흐름표에 직접 기재된 수치를 최우선으로 사용하세요. 주석에 포함된 종속기업·관계기업의 요약 재무정보는 삼성전자 전체 수치가 아니므로 혼동하지 마세요.
9. 여러 연도의 결론이 동일한 경우, 각 연도를 개별 나열하지 말고 '2014~2024년 전 기간' 형식으로 묶어 한 문장으로 요약하세요. 예: '2014~2024년 모든 연도에서 적정의견이 표명되어 감사의견 변경은 없었습니다.'
10. 수치(금액, 비율 등)는 반드시 참고 문서에 명시된 것만 사용하세요. 문서에 수치가 없는 연도에 대해 숫자를 추론하거나 생성하지 마세요."""

_FINANCIAL_QUERY_KEYWORDS = {
    "영업이익", "매출", "순이익", "당기순이익", "영업손익", "자산", "부채", "자본",
    "위험", "리스크", "재무위험", "유동성", "신용위험",
}
_FINANCIAL_SECTIONS = {"포괄손익", "재무상태표", "현금흐름"}
_FINANCIAL_CSV_PATH = "data/processed/financial_data.csv"

_QUERY_TYPE_LABELS = {
    "numeric": "정량형",
    "narrative": "서술형",
    "hybrid": "혼합형",
}

_NUMERIC_INTENT_MARKERS = {
    "얼마", "금액", "수치", "합계", "총액", "총계", "차이", "차액",
    "증가율", "감소율", "비율", "평균", "최대", "최소",
}

_HYBRID_INTENT_MARKERS = {
    "비교", "추이", "트렌드", "변화", "변동", "증가", "감소", "달라",
    "설명", "이유", "배경", "의미", "시사점", "영향", "평가", "해석",
}

_NARRATIVE_SECTION_NAMES = {"감사의견", "핵심감사사항", "감사의견근거", "주석"}

_METRIC_DEFINITIONS = [
    {
        "name": "매출액",
        "aliases": ["매출액", "매출"],
        "sections": {"포괄손익"},
        "table_types": {"income_statement"},
    },
    {
        "name": "영업이익",
        "aliases": ["영업이익", "영업손익"],
        "sections": {"포괄손익"},
        "table_types": {"income_statement"},
    },
    {
        "name": "당기순이익",
        "aliases": ["당기순이익", "순이익"],
        "sections": {"포괄손익", "현금흐름"},
        "table_types": {"income_statement", "cash_flow"},
    },
    {
        "name": "자산총계",
        "aliases": ["자산총계", "총자산", "자산 총계"],
        "sections": {"재무상태표"},
        "table_types": {"balance_sheet"},
    },
    {
        "name": "부채총계",
        "aliases": ["부채총계", "총부채", "총 부채", "부채 총계"],
        "sections": {"재무상태표"},
        "table_types": {"balance_sheet"},
    },
    {
        "name": "자본총계",
        "aliases": ["자본총계", "총자본", "자본 총계"],
        "sections": {"재무상태표"},
        "table_types": {"balance_sheet"},
    },
    {
        "name": "유동자산",
        "aliases": ["유동자산"],
        "sections": {"재무상태표"},
        "table_types": {"balance_sheet"},
    },
    {
        "name": "비유동자산",
        "aliases": ["비유동자산"],
        "sections": {"재무상태표"},
        "table_types": {"balance_sheet"},
    },
    {
        "name": "유동부채",
        "aliases": ["유동부채"],
        "sections": {"재무상태표"},
        "table_types": {"balance_sheet"},
    },
    {
        "name": "비유동부채",
        "aliases": ["비유동부채"],
        "sections": {"재무상태표"},
        "table_types": {"balance_sheet"},
    },
    {
        "name": "영업활동 현금흐름",
        "aliases": ["영업활동현금흐름", "영업활동 현금흐름", "영업활동으로인한현금흐름"],
        "sections": {"현금흐름"},
        "table_types": {"cash_flow"},
    },
    {
        "name": "투자활동 현금흐름",
        "aliases": ["투자활동현금흐름", "투자활동 현금흐름", "투자활동으로인한현금흐름"],
        "sections": {"현금흐름"},
        "table_types": {"cash_flow"},
    },
    {
        "name": "재무활동 현금흐름",
        "aliases": ["재무활동현금흐름", "재무활동 현금흐름", "재무활동으로인한현금흐름"],
        "sections": {"현금흐름"},
        "table_types": {"cash_flow"},
    },
]

_TABLE_TYPE_LABELS = {
    "income_statement": "포괄손익",
    "balance_sheet": "재무상태표",
    "cash_flow": "현금흐름",
}

_GENERIC_METRIC_HINTS = {
    "부채": ["부채총계"],
    "자산": ["자산총계"],
    "자본": ["자본총계"],
    "현금흐름": ["영업활동 현금흐름", "투자활동 현금흐름", "재무활동 현금흐름"],
}

_MAX_REASONABLE_KRW_MILLION = 10_000_000_000

# 형태소 분석기 없이 간단 처리: 조사/어미 suffix 제거 및 불용어 목록
_KO_SUFFIXES = [
    "이", "가", "은", "는", "을", "를", "의", "에", "에서", "으로", "로",
    "와", "과", "도", "만",                                    # 접속·보조 조사 추가
    "이다", "이고", "이며", "하다", "하고", "하며", "한", "된", "되는",
    "있나요", "인가요", "인지", "나요", "인가", "했나요", "됐나요",
]

_STOPWORDS = {
    "알려주세요", "알려줘", "무엇인가요", "어떻게", "설명해주세요", "설명해줘",
    "대해", "대해서", "지남에", "따른", "인한", "의한", "관련", "변화", "추이",
    "비교", "비교해줘", "얼마인가요", "얼마", "어떤", "있는", "있나요", "시간이", "년도", "연도",
    "내용", "사항", "경우", "통해", "위해", "때문", "결과",  # 범용 명사 추가
}

# 멀티연도 비교/트렌드 질문 감지 패턴
_MULTI_YEAR_PATTERNS = [
    r"트렌드", r"추이", r"변화", r"변경", r"변동",
    r"연도별", r"매년", r"증가", r"감소", r"성장",
    r"비교", r"어느 연도", r"몇 년도", r"언제",
    r"전반적", r"전체적", r"역대", r"이력",
    r"차이", r"다르", r"연도", r"년도", r"달리", r"다름",
]

_YEARLY_QUERY_MARKERS = {"매년", "연도별", "년도별", "추이", "변화", "변동", "증가", "감소"}
_EXTREMA_QUERY_MARKERS = {"역대", "최대", "최소", "가장", "최고", "최저"}
_CHANGE_PRESENCE_MARKERS = {"변경", "바뀌", "달라", "동일", "같", "차이", "유지"}
_BINARY_QUESTION_MARKERS = {"있나", "있나요", "없나", "없나요", "여부", "인지", "인가요", "맞나", "맞나요"}

# 감사보고서 섹션별 대표 키워드 — 섹션 타겟 검색에 사용 (소규모 섹션 보호용)
_SECTION_TARGET_MAP = {
    "감사의견": ["감사의견", "적정의견", "한정의견", "부적정"],
    "핵심감사사항": ["핵심감사사항", "핵심감사", "리스크", "위험", "감사위험", "재무위험"],
    "감사의견근거": ["감사의견근거", "감사인의 책임"],
    "포괄손익": ["영업이익", "매출", "순이익", "당기순이익", "영업손익"],
    "재무상태표": ["자산총계", "부채총계", "총부채", "총 부채", "부채 총계", "부 채 총 계", "자본총계"],
    "현금흐름": ["현금흐름", "영업활동", "투자활동", "재무활동"]
}


@lru_cache(maxsize=1)
def _load_financial_df(csv_path: str = _FINANCIAL_CSV_PATH) -> pd.DataFrame:
    """정형 재무데이터 CSV 로드. 없으면 빈 DataFrame 반환."""
    if not os.path.exists(csv_path):
        return pd.DataFrame(columns=["year", "table_type", "item", "value_raw"])
    try:
        return pd.read_csv(csv_path)
    except Exception:
        return pd.DataFrame(columns=["year", "table_type", "item", "value_raw"])

class RAGPipeline:
    def __init__(
        self,
        csv_path: str = "data/processed/sections.csv",
        rebuild: bool = False,
        backend: str | None = None,
        local_model: str | None = None,
        openai_model: str | None = None,
    ):
        self._load_runtime_config(
            backend=backend,
            local_model=local_model,
            openai_model=openai_model,
        )
        self.csv_path = csv_path
        self.vector_store = VectorStore()
        
        if rebuild or not self.vector_store.is_built():
            print("Building RAG Index...")
            chunks = load_chunks_from_csv(self.csv_path)
            embeddings = build_embeddings(chunks)
            self.vector_store.build(chunks, embeddings)
            
        self.local_model_pipeline = None
        self.tokenizer = None
        self.model = None
        self._index_lock = threading.RLock()

    def _load_runtime_config(
        self,
        backend: str | None = None,
        local_model: str | None = None,
        openai_model: str | None = None,
    ) -> None:
        """실행 시점 환경변수와 UI 인자를 읽어 백엔드/모델 설정을 확정."""
        has_openai_key = bool(os.getenv("OPENAI_API_KEY"))
        backend_default = "openai" if has_openai_key else "local"
        backend = (backend or os.getenv("LLM_BACKEND", backend_default)).strip().lower()

        if backend not in _SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unsupported LLM_BACKEND='{backend}'. "
                f"Choose one of {sorted(_SUPPORTED_BACKENDS)}."
            )
        if backend == "openai" and not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "LLM_BACKEND=openai 이지만 OPENAI_API_KEY가 비어 있습니다."
            )
        if backend == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "LLM_BACKEND=anthropic 이지만 ANTHROPIC_API_KEY가 비어 있습니다."
            )

        retrieve_k = int(os.getenv("RETRIEVE_K", str(_DEFAULT_RETRIEVE_K)))
        max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", str(_DEFAULT_MAX_NEW_TOKENS)))
        if retrieve_k < 1:
            raise ValueError("RETRIEVE_K must be >= 1.")
        if max_new_tokens < 1:
            raise ValueError("MAX_NEW_TOKENS must be >= 1.")

        self.backend = backend
        self.local_model = (local_model or os.getenv("LOCAL_MODEL", _DEFAULT_LOCAL_MODEL)).strip()
        self.openai_model = (
            openai_model or os.getenv("OPENAI_MODEL", _DEFAULT_OPENAI_MODEL) or _DEFAULT_OPENAI_MODEL
        ).strip()
        self.retrieve_k = retrieve_k
        self.max_new_tokens = max_new_tokens

    def _rebuild_vector_index(self) -> None:
        """FAISS 인덱스를 재생성."""
        with self._index_lock:
            print("Rebuilding FAISS index...")
            chunks = load_chunks_from_csv(self.csv_path)
            embeddings = build_embeddings(chunks)
            self.vector_store.build(chunks, embeddings)

    def _auto_detect_years(self, query: str) -> list[int]:
        """질문 텍스트에서 연도를 감지하고 범위 표현(2018~2024, 2018년부터 2024년까지)을 확장."""
        years = {int(y) for y in re.findall(r"(20\d{2})년?", query) if 2014 <= int(y) <= 2024}

        # 예: 2018~2024, 2018-2024, 2018년부터 2024년까지, 2018년에서 2024년까지
        range_patterns = [
            r"(20\d{2})\s*년?\s*[~\-–—]\s*(20\d{2})\s*년?",
            r"(20\d{2})\s*년?\s*부터\s*(20\d{2})\s*년?\s*까지",
            r"(20\d{2})\s*년?\s*에서\s*(20\d{2})\s*년?\s*까지",
        ]
        for pattern in range_patterns:
            for start, end in re.findall(pattern, query):
                s = int(start)
                e = int(end)
                lo, hi = sorted((s, e))
                lo = max(lo, 2014)
                hi = min(hi, 2024)
                if lo <= hi:
                    years.update(range(lo, hi + 1))

        return sorted(years)

    def _is_yearly_query(self, query: str) -> bool:
        """'매년/연도별/추이'처럼 연도 커버리지가 중요한 질문 여부."""
        return any(marker in query for marker in _YEARLY_QUERY_MARKERS)

    def _is_extrema_query(self, query: str) -> bool:
        """'역대 최대/최소/가장'처럼 전체 기간 집계가 필요한 질문 여부."""
        return any(marker in query for marker in _EXTREMA_QUERY_MARKERS)

    def _is_change_presence_query(self, query: str) -> bool:
        """변경/동일 여부를 묻는 이진 질문인지 판단."""
        return (
            any(marker in query for marker in _CHANGE_PRESENCE_MARKERS)
            and any(marker in query for marker in _BINARY_QUESTION_MARKERS)
        )

    def _is_multi_year_query(self, query: str) -> bool:
        """멀티연도 비교·트렌드 질문 여부 판단"""
        return any(re.search(p, query) for p in _MULTI_YEAR_PATTERNS)

    def _detect_target_section(self, query: str) -> str | None:
        """질문 키워드로 가장 관련 높은 섹션명 추론. 없으면 None."""
        for section, keywords in _SECTION_TARGET_MAP.items():
            if any(kw in query for kw in keywords):
                return section
        return None

    def _retrieve_multi_year(self, query: str, total_k: int = 6, target_years: list[int] | None = None) -> list[dict]:
        """
        멀티연도 질문용 검색.
        - total_k를 전체 반환 개수 상한으로 사용한다.
        - total_k < 연도 수면, 연도별 최고 점수 후보를 랭킹해 상위 연도만 선택한다.
        - total_k >= 연도 수면, 연도당 1개를 우선 보장하고 남는 슬롯은 고득점 후보로 채운다.
        protect_section이 감지된 경우, 해당 섹션 청크를 실제로 찾지 못한 연도는 건너뜀.
        (예: 핵심감사사항이 없는 2014~2017년에 재무상태표를 혼합해 환각을 유발하는 문제 방지)
        """
        years = target_years if target_years else list(range(2014, 2025))
        if total_k < 1:
            return []

        protect = self._detect_target_section(query)
        per_year: list[tuple[int, list[dict]]] = []

        for year in years:
            # 연도별 후보 풀은 3개 정도 확보해, total_k 분배 시 변별력을 높인다.
            candidates = self.retrieve(query, k=max(3, min(8, total_k)), year_filter=year)
            # protect_section이 지정됐는데 해당 연도 결과에 실제로 없으면 스킵
            # → 이질적인 섹션 혼합으로 인한 LLM 환각 방지
            if protect and not any(c["section"] == protect for c in candidates):
                continue
            if candidates:
                per_year.append((year, candidates))

        if not per_year:
            return []

        # total_k가 연도 수보다 작으면 "연도 대표 후보" 중에서 상위 total_k개만 선택
        if total_k < len(per_year):
            ranked = sorted(
                per_year,
                key=lambda yc: float(yc[1][0].get("score", 0.0)),
                reverse=True,
            )
            chosen = sorted(ranked[:total_k], key=lambda yc: yc[0])  # 답변 가독성을 위해 연도 오름차순
            return [cands[0] for _, cands in chosen]

        # 연도당 1개 우선 배치
        merged: list[dict] = [cands[0] for _, cands in per_year]
        remain = total_k - len(merged)
        if remain <= 0:
            return merged[:total_k]

        # 남는 슬롯은 연도와 무관하게 고득점 후보를 채택
        extras: list[tuple[float, int, dict]] = []
        for year, cands in per_year:
            for cand in cands[1:]:
                extras.append((float(cand.get("score", 0.0)), year, cand))
        extras.sort(key=lambda x: (-x[0], x[1]))

        seen = {(c["year"], c["section"], c["text"][:50]) for c in merged}
        for _, _, cand in extras:
            key = (cand["year"], cand["section"], cand["text"][:50])
            if key in seen:
                continue
            merged.append(cand)
            seen.add(key)
            remain -= 1
            if remain == 0:
                break

        return merged[:total_k]

    def _extract_keywords(self, query: str) -> list[str]:
        """
        질문에서 조사/어미를 제거한 핵심 키워드 추출.
        어절 단위 분리 후 기호 및 suffix 제거, 불용어 필터링. 2자 이상만 반환.
        """
        query = re.sub(r"[()\[\]~,]", " ", query)
        keywords = []
        for word in query.split():
            clean = word.strip("?!.,;:'\"")
            for suffix in sorted(_KO_SUFFIXES, key=len, reverse=True):  # 긴 suffix 먼저
                if clean.endswith(suffix) and len(clean) - len(suffix) >= 2:
                    clean = clean[:-len(suffix)]
                    break
            if len(clean) >= 2 and clean not in _STOPWORDS:
                keywords.append(clean)
        return list(dict.fromkeys(keywords))  # 중복 제거, 순서 유지

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"\s+", "", str(text or "")).lower()

    def _pick_year_diverse_tool_rows(self, rows: list[dict], target_years: list[int], limit: int) -> list[dict]:
        """
        연도별 추이 질문에서 연도 커버리지를 우선한다.
        1) 각 연도별 최고 점수 1개를 먼저 선택
        2) 남는 슬롯을 점수 순으로 보충
        """
        if limit < 1:
            return []
        if not rows:
            return []

        selected: list[dict] = []
        seen: set[tuple] = set()
        by_year: dict[int, list[dict]] = {}
        for row in rows:
            by_year.setdefault(int(row["year"]), []).append(row)

        # 연도 오름차순으로 1개씩 우선 확보
        for year in sorted(dict.fromkeys(target_years)):
            candidates = by_year.get(year, [])
            if not candidates:
                continue
            top = candidates[0]
            key = (top["year"], top["table_type"], top["item"], tuple(top["values"][:2]))
            if key not in seen:
                selected.append(top)
                seen.add(key)
            if len(selected) >= limit:
                return selected[:limit]

        # 남는 슬롯은 전체 점수 순으로 채우기
        for row in rows:
            key = (row["year"], row["table_type"], row["item"], tuple(row["values"][:2]))
            if key in seen:
                continue
            selected.append(row)
            seen.add(key)
            if len(selected) >= limit:
                break
        return selected[:limit]

    def _lookup_financial_rows(self, query: str, target_years: list[int], limit: int = 6) -> list[dict]:
        """정형 재무 CSV를 조회해 질문과 직접 매칭되는 행을 반환."""
        df = _load_financial_df()
        if df.empty:
            return []

        keywords = self._extract_keywords(query)
        if not keywords:
            return []

        # query normalization 결과와 item normalization 결과를 비교해 점수화
        norm_query = self._normalize_text(query)
        norm_keywords = [self._normalize_text(k) for k in keywords if len(k) >= 2]
        if not norm_keywords:
            return []

        if target_years:
            cand = df[df["year"].isin(target_years)].copy()
        else:
            cand = df.copy()
        if cand.empty:
            return []

        rows = []
        for _, row in cand.iterrows():
            item = str(row.get("item", ""))
            table_type = str(row.get("table_type", ""))
            norm_item = self._normalize_text(item)
            score = 0
            for kw in norm_keywords:
                if kw in norm_item:
                    score += 2
                elif kw in self._normalize_text(table_type):
                    score += 1

            # "총 부채/부채총계" 의도에서는 세부 계정(예: 기타유동부채)보다
            # 상위 집계 항목(유동/비유동/부채총계)을 우선한다.
            asks_total_liability = ("부채" in norm_query) and ("총" in norm_query or "총계" in norm_query)
            if asks_total_liability:
                if "부채총계" in norm_item:
                    score += 5
                elif "유동부채" in norm_item or "비유동부채" in norm_item:
                    score += 2
                if "기타" in norm_item:
                    score -= 1
                if re.match(r"^\s*\d+\.", item):
                    score -= 1
            if score <= 0:
                continue

            raw_val = row.get("value_raw", "")
            try:
                values = json.loads(raw_val) if isinstance(raw_val, str) else raw_val
            except Exception:
                values = [str(raw_val)]
            if not isinstance(values, list):
                values = [values]

            rows.append(
                {
                    "year": int(row.get("year")),
                    "table_type": table_type,
                    "item": item,
                    "values": [str(v) for v in values if str(v).strip()],
                    "score": float(score),
                }
            )

        rows.sort(key=lambda x: (-x["score"], x["year"], x["item"]))
        if (self._is_yearly_query(query) or self._is_extrema_query(query)) and target_years:
            return self._pick_year_diverse_tool_rows(rows, target_years, limit=max(limit, len(target_years)))
        return rows[:max(1, limit)]

    def run_tool_calling(self, query: str, year_filter: list[int] | int | None = None) -> list[dict]:
        """
        Tool Calling 레이어: 정형 재무데이터 조회 도구를 조건부 실행.
        - 재무 키워드가 있는 질문에서만 동작.
        - 연도 필터/질문 내 연도를 우선 적용.
        """
        if not any(kw in query for kw in _FINANCIAL_QUERY_KEYWORDS):
            return []

        if year_filter is None:
            years = self._auto_detect_years(query)
        elif isinstance(year_filter, int):
            years = [year_filter]
        else:
            years = list(year_filter)

        # '역대 최대/최소' 질문인데 연도가 명시되지 않으면 전체 기간(2014~2024)을 강제 조회한다.
        if not years and self._is_extrema_query(query):
            years = list(range(2014, 2025))

        limit = max(6, len(years)) if years else 6
        return self._lookup_financial_rows(query, years, limit=limit)

    def _format_tool_context(self, tool_rows: list[dict]) -> str:
        if not tool_rows:
            return ""
        lines = ["[도구 조회 결과: financial_data.csv]"]
        for row in tool_rows:
            joined = ", ".join(row["values"]) if row["values"] else "값 없음"
            lines.append(
                f"- {row['year']}년 / {row['table_type']} / {row['item']} / 값: {joined}"
            )
        lines.append("도구 조회 결과는 참고 문서와 함께 교차검증하여 답변에 반영하세요.")
        return "\n".join(lines)

    @staticmethod
    def describe_query_type(query_type: str) -> str:
        return _QUERY_TYPE_LABELS.get(query_type, query_type)

    def _contains_normalized_term(self, text: str, terms: set[str]) -> bool:
        norm_text = self._normalize_text(text)
        return any(self._normalize_text(term) in norm_text for term in terms)

    def _match_query_metrics(self, query: str) -> list[str]:
        norm_query = self._normalize_text(query)
        matched = []
        for metric in _METRIC_DEFINITIONS:
            aliases = sorted(metric["aliases"], key=len, reverse=True)
            if any(self._normalize_text(alias) in norm_query for alias in aliases):
                matched.append(metric["name"])
        if not matched:
            for hint, metric_names in _GENERIC_METRIC_HINTS.items():
                if self._normalize_text(hint) in norm_query:
                    matched.extend(metric_names)
        matched = list(dict.fromkeys(matched))
        return matched

    def _get_metric_definition(self, metric_name: str) -> dict | None:
        for metric in _METRIC_DEFINITIONS:
            if metric["name"] == metric_name:
                return metric
        return None

    def _match_metric_from_text(self, text: str, table_type: str = "", section: str = "") -> str | None:
        norm_text = self._normalize_text(text)
        norm_table_type = self._normalize_text(table_type)
        norm_section = self._normalize_text(section)

        for metric in _METRIC_DEFINITIONS:
            aliases = sorted(metric["aliases"], key=len, reverse=True)
            if not any(self._normalize_text(alias) in norm_text for alias in aliases):
                continue

            allowed_table_types = metric.get("table_types") or set()
            if allowed_table_types and table_type and table_type not in allowed_table_types:
                continue

            allowed_sections = metric.get("sections") or set()
            if allowed_sections and section and not any(
                self._normalize_text(s) in norm_section for s in allowed_sections
            ):
                continue
            if allowed_table_types and table_type and any(
                self._normalize_text(t) in norm_table_type for t in allowed_table_types
            ):
                return metric["name"]
            if allowed_sections and section and any(
                self._normalize_text(s) in norm_section for s in allowed_sections
            ):
                return metric["name"]
            if not table_type and not section:
                return metric["name"]
            if table_type in allowed_table_types or section in allowed_sections:
                return metric["name"]
        return None

    def classify_tool_rows(self, query: str, tool_rows: list[dict]) -> dict:
        """
        질문 메트릭과 도구 행의 정합성을 분류.
        반환:
        {
          "metrics": [...],
          "primary_rows": [...],   # 질문 메트릭과 직접 일치
          "aux_rows": [...],       # 비일치(보조 참고)
        }
        """
        if not tool_rows:
            return {"metrics": [], "primary_rows": [], "aux_rows": []}

        metrics = self._match_query_metrics(query)
        if not metrics:
            return {"metrics": [], "primary_rows": [], "aux_rows": list(tool_rows)}

        primary_rows: list[dict] = []
        aux_rows: list[dict] = []
        for row in tool_rows:
            metric = self._match_metric_from_text(
                row.get("item", ""),
                table_type=row.get("table_type", ""),
            )
            if metric in metrics:
                primary_rows.append(row)
            else:
                aux_rows.append(row)

        return {
            "metrics": metrics,
            "primary_rows": primary_rows,
            "aux_rows": aux_rows,
        }

    def _preferred_sections_for_query(self, query: str) -> set[str]:
        preferred = set()
        target_section = self._detect_target_section(query)
        if target_section:
            preferred.add(target_section)

        for metric in self._match_query_metrics(query):
            definition = self._get_metric_definition(metric)
            if not definition:
                continue
            preferred.update(definition.get("sections", set()))
        return preferred

    def _extract_topic_filter_terms(self, query: str) -> list[str]:
        """질문의 명시 토픽(예: 코로나19) 기반 후처리 필터 토큰."""
        norm_query = self._normalize_text(query)
        terms: list[str] = []
        if "코로나19" in norm_query or "covid19" in norm_query:
            terms.extend(["코로나19", "covid19", "covid-19"])
        return list(dict.fromkeys(terms))

    def _filter_contexts_by_topic_terms(self, query: str, contexts: list[dict]) -> list[dict]:
        """
        토픽 명시 질의에서 관련 토픽이 없는 문서를 제거해 출처 혼입을 줄인다.
        필터 후 결과가 비면 원본을 반환한다.
        """
        if not contexts:
            return contexts
        terms = self._extract_topic_filter_terms(query)
        if not terms:
            return contexts

        norm_terms = [self._normalize_text(t) for t in terms]
        filtered = []
        for ctx in contexts:
            text = self._normalize_text(ctx.get("text", ""))
            if any(t in text for t in norm_terms):
                filtered.append(ctx)
        return filtered if filtered else contexts

    def route_query_type(self, query: str, year_filter: list[int] | int | None = None) -> str:
        del year_filter  # 현재는 질의 텍스트만으로 라우팅
        metrics = self._match_query_metrics(query)
        has_metric = bool(metrics)
        has_numeric_intent = self._contains_normalized_term(query, _NUMERIC_INTENT_MARKERS)
        has_hybrid_intent = self._contains_normalized_term(query, _HYBRID_INTENT_MARKERS)
        target_section = self._detect_target_section(query)
        is_narrative_section = target_section in _NARRATIVE_SECTION_NAMES or "주석" in query
        is_financial_section = target_section in _FINANCIAL_SECTIONS

        if is_narrative_section and not has_metric:
            return "narrative"
        if (has_metric or is_financial_section) and has_hybrid_intent:
            return "hybrid"
        if has_metric or has_numeric_intent:
            return "numeric"
        if is_narrative_section:
            return "narrative"
        return "narrative"

    @staticmethod
    def _parse_number_token(token: str | int | float | None) -> int | None:
        if token is None:
            return None
        if isinstance(token, (int, float)):
            return int(token)

        text = str(token).strip()
        if not text or text in {"-", "—", "–"}:
            return None

        negative = text.startswith("(") and text.endswith(")")
        cleaned = text.replace(",", "").replace("(", "").replace(")", "").replace(" ", "")
        if not re.fullmatch(r"-?\d+", cleaned):
            return None

        value = int(cleaned)
        return -abs(value) if negative else value

    def _pick_primary_numeric_value(self, values: list[str] | tuple[str, ...]) -> tuple[int | None, str | None]:
        fallback: tuple[int | None, str | None] = (None, None)
        no_comma_candidates: list[tuple[int, str, int]] = []
        for raw in values:
            parsed = self._parse_number_token(raw)
            if parsed is None:
                continue
            raw_text = str(raw).strip()
            compact = raw_text.replace(",", "").replace("(", "").replace(")", "").replace("-", "")
            if "," in raw_text:
                return parsed, raw_text
            if compact.isdigit() and len(compact) >= 5:
                no_comma_candidates.append((parsed, raw_text, len(compact)))
                continue
            if fallback[0] is None:
                fallback = (parsed, raw_text)
        if no_comma_candidates:
            parsed, raw_text, _ = max(no_comma_candidates, key=lambda item: item[2])
            return parsed, raw_text
        return fallback

    @staticmethod
    def _is_reasonable_numeric_value(value: int | None) -> bool:
        if value is None:
            return False
        return abs(int(value)) <= _MAX_REASONABLE_KRW_MILLION

    @staticmethod
    def _build_relaxed_alias_pattern(alias: str) -> re.Pattern:
        compact = re.sub(r"\s+", "", str(alias or ""))
        if not compact:
            return re.compile(r"$^")
        parts = [re.escape(ch) for ch in compact]
        return re.compile(r"\s*".join(parts))

    def _extract_metric_value_from_context(self, metric_name: str, context_text: str) -> tuple[int | None, str | None]:
        metric = self._get_metric_definition(metric_name)
        if metric is None:
            return (None, None)

        text = str(context_text or "")
        aliases = sorted(metric["aliases"], key=len, reverse=True)
        for alias in aliases:
            pattern = self._build_relaxed_alias_pattern(alias)
            for match in pattern.finditer(text):
                window = text[match.end(): match.end() + 180]
                tokens = re.findall(r"\(?-?\d{1,3}(?:,\d{3})+\)?|\(?-?\d+\)?", window)
                parsed, raw = self._pick_primary_numeric_value(tokens)
                if self._is_reasonable_numeric_value(parsed):
                    return parsed, raw
        return (None, None)

    @staticmethod
    def _format_number(value: int | None) -> str:
        if value is None:
            return "값 없음"
        return f"{value:,}"

    @staticmethod
    def _format_percent(value: float | None) -> str:
        if value is None:
            return "계산 불가"
        return f"{value:.2f}%"

    @staticmethod
    def _is_cashflow_metric(metric_name: str) -> bool:
        return "현금흐름" in str(metric_name or "")

    @staticmethod
    def _describe_cashflow_change(prev_value: int, curr_value: int) -> str:
        prev_abs = abs(int(prev_value))
        curr_abs = abs(int(curr_value))

        if prev_value < 0 and curr_value < 0:
            if curr_abs > prev_abs:
                return "순유출 규모 확대"
            if curr_abs < prev_abs:
                return "순유출 규모 축소"
            return "순유출 규모 변동 없음"
        if prev_value > 0 and curr_value > 0:
            if curr_abs > prev_abs:
                return "순유입 규모 확대"
            if curr_abs < prev_abs:
                return "순유입 규모 축소"
            return "순유입 규모 변동 없음"
        if prev_value == 0 and curr_value == 0:
            return "현금흐름 변동 없음"
        if prev_value == 0:
            return "0에서 방향 전환"
        if curr_value == 0:
            return "순유입/순유출에서 0으로 축소"
        if prev_value < 0 < curr_value:
            return "순유출에서 순유입으로 전환"
        if prev_value > 0 > curr_value:
            return "순유입에서 순유출로 전환"
        return "현금흐름 방향 변화"

    def _collect_source_labels(self, facts: list[dict]) -> list[str]:
        labels = []
        seen = set()
        for fact in facts:
            label = fact.get("source")
            if label and label not in seen:
                labels.append(label)
                seen.add(label)
        return labels

    def _build_numeric_facts(
        self,
        query: str,
        contexts: list[dict],
        tool_rows: list[dict],
        year_filter: list[int] | int | None = None,
    ) -> list[dict]:
        metrics = self._match_query_metrics(query)
        if not metrics:
            return []

        if year_filter is None:
            target_years = self._auto_detect_years(query)
        elif isinstance(year_filter, int):
            target_years = [year_filter]
        else:
            target_years = list(year_filter)

        facts_by_key: dict[tuple[str, int], dict] = {}

        for row in tool_rows:
            metric_name = self._match_metric_from_text(
                row.get("item", ""),
                table_type=row.get("table_type", ""),
            )
            if metric_name not in metrics:
                continue

            year = int(row.get("year"))
            if target_years and year not in target_years:
                continue

            parsed, raw = self._pick_primary_numeric_value(row.get("values", []))
            if parsed is None:
                continue

            table_label = _TABLE_TYPE_LABELS.get(row.get("table_type", ""), row.get("table_type", ""))
            key = (metric_name, year)
            facts_by_key[key] = {
                "metric": metric_name,
                "year": year,
                "value": parsed,
                "value_raw": raw or self._format_number(parsed),
                "source": f"{year}년 {table_label}",
                "source_detail": f"{year}년 / {table_label} / {row.get('item')}",
                "source_kind": "structured",
            }

        for ctx in contexts:
            year = int(ctx["year"])
            if target_years and year not in target_years:
                continue
            for metric_name in metrics:
                key = (metric_name, year)
                if key in facts_by_key:
                    continue

                definition = self._get_metric_definition(metric_name)
                if definition and definition.get("sections") and ctx["section"] not in definition["sections"]:
                    continue

                parsed, raw = self._extract_metric_value_from_context(metric_name, ctx["text"])
                if parsed is None:
                    continue
                facts_by_key[key] = {
                    "metric": metric_name,
                    "year": year,
                    "value": parsed,
                    "value_raw": raw or self._format_number(parsed),
                    "source": f"{year}년 {ctx['section']}",
                    "source_detail": f"{year}년 {ctx['section']}",
                    "source_kind": "context",
                }

        facts = list(facts_by_key.values())
        facts.sort(key=lambda item: (item["metric"], item["year"]))
        return facts

    def _build_numeric_answer(
        self,
        query: str,
        contexts: list[dict],
        tool_rows: list[dict],
        year_filter: list[int] | int | None = None,
    ) -> str:
        facts = self._build_numeric_facts(query, contexts, tool_rows, year_filter=year_filter)
        if not facts:
            sources = self.get_sources(contexts) if contexts else "없음"
            return (
                "정형 데이터와 재무제표 문맥에서 질문에 대응되는 수치를 확정하지 못했습니다.\n\n"
                f"참고 출처: {sources}"
            )

        metrics = self._match_query_metrics(query)
        query_years = self._auto_detect_years(query)
        is_calc_query = self._contains_normalized_term(query, {"차이", "차액", "증가율", "감소율", "비율", "평균"})
        is_extrema_query = self._is_extrema_query(query)

        lines = ["정량 결과"]
        calc_lines = []

        for metric_name in metrics:
            metric_facts = [fact for fact in facts if fact["metric"] == metric_name]
            if not metric_facts:
                continue

            metric_facts.sort(key=lambda item: item["year"])
            for fact in metric_facts:
                lines.append(
                    f"- {fact['year']}년 {metric_name}: {self._format_number(fact['value'])}백만원"
                )

            if len(metric_facts) == 2 and (is_calc_query or len(query_years) >= 2):
                first, second = metric_facts
                delta = second["value"] - first["value"]
                pct = None if first["value"] == 0 else (delta / abs(first["value"])) * 100
                direction = "증가" if delta > 0 else "감소" if delta < 0 else "변동 없음"
                calc_lines.append(
                    f"- {metric_name} 증감액({first['year']}→{second['year']}): {self._format_number(delta)}백만원"
                )
                if pct is not None:
                    calc_lines.append(
                        f"- {metric_name} 증감률({first['year']}→{second['year']}): {self._format_percent(pct)}"
                    )
                calc_lines.append(
                    f"- {metric_name} 증감 방향({first['year']}→{second['year']}): {direction}"
                )
                if self._is_cashflow_metric(metric_name):
                    flow_trend = self._describe_cashflow_change(first["value"], second["value"])
                    calc_lines.append(
                        f"- {metric_name} 해석({first['year']}→{second['year']}): {flow_trend}"
                    )
            elif len(metric_facts) >= 2 and is_extrema_query:
                max_fact = max(metric_facts, key=lambda item: item["value"])
                min_fact = min(metric_facts, key=lambda item: item["value"])
                calc_lines.append(
                    f"- {metric_name} 최대: {max_fact['year']}년 {self._format_number(max_fact['value'])}백만원"
                )
                calc_lines.append(
                    f"- {metric_name} 최소: {min_fact['year']}년 {self._format_number(min_fact['value'])}백만원"
                )

        if calc_lines:
            lines.append("")
            lines.append("계산 결과")
            lines.extend(calc_lines)

        source_labels = self._collect_source_labels(facts)
        if source_labels:
            lines.append("")
            lines.append("출처")
            for source in source_labels:
                lines.append(f"- {source}")

        return "\n".join(lines)

    def _finalize_text_response(self, response: str, stream: bool) -> str | Generator:
        if not stream:
            return response

        def _stream_once():
            yield response

        return _stream_once()

    def _dispatch_backend_response(
        self,
        *,
        user_prompt: str,
        system_prompt: str,
        stream: bool,
        max_new_tokens: int | None = None,
    ) -> str | Generator:
        if self.backend == "anthropic":
            return self._ask_anthropic(user_prompt, stream, system_prompt)
        if self.backend == "openai":
            return self._ask_openai(user_prompt, stream, system_prompt)
        return self._ask_local(user_prompt, stream, system_prompt, max_new_tokens=max_new_tokens)

    def _build_coverage_guard_note(self, query: str, contexts: list[dict]) -> str:
        """
        변경/동일 여부 질의에서 연도 커버리지가 부족할 때 단정 답변을 막기 위한 가드 문구.
        """
        if not self._is_change_presence_query(query):
            return ""

        explicit_years = self._auto_detect_years(query)
        if explicit_years:
            expected_years = sorted(dict.fromkeys(explicit_years))
        else:
            expected_years = list(range(2014, 2025))

        observed_years = sorted({int(c.get("year")) for c in contexts if c.get("year") is not None})
        missing = [y for y in expected_years if y not in observed_years]
        if not missing:
            return ""

        observed_text = ", ".join(str(y) for y in observed_years) if observed_years else "없음"
        missing_text = ", ".join(str(y) for y in missing)
        return (
            "[커버리지 경고]\n"
            f"- 질문에서 기대되는 확인 연도: {expected_years[0]}~{expected_years[-1]}년\n"
            f"- 실제 확인된 연도: {observed_text}\n"
            f"- 누락 연도: {missing_text}\n"
            "- 누락 연도가 있는 상태에서는 '변경 없음/있음'을 단정하지 말고, 확인된 연도 범위 내 결과만 답하세요."
        )

    def _ask_narrative(
        self,
        query: str,
        *,
        stream: bool,
        contexts: list[dict],
        tool_rows: list[dict],
        is_multi: bool,
        max_new_tokens: int,
    ) -> str | Generator:
        system_prompt = RAG_SYSTEM_MULTI if is_multi else RAG_SYSTEM
        tool_context = self._format_tool_context(tool_rows)
        user_prompt = self.build_prompt(query, contexts)
        coverage_note = self._build_coverage_guard_note(query, contexts)
        if coverage_note:
            user_prompt = f"{coverage_note}\n\n{user_prompt}"
        if tool_context:
            user_prompt = f"{tool_context}\n\n{user_prompt}"
        return self._dispatch_backend_response(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            stream=stream,
            max_new_tokens=max_new_tokens,
        )

    def _ask_numeric(
        self,
        query: str,
        *,
        stream: bool,
        contexts: list[dict],
        tool_rows: list[dict],
        year_filter: list[int] | int | None = None,
        is_multi: bool,
        max_new_tokens: int,
    ) -> str | Generator:
        response = self._build_numeric_answer(query, contexts, tool_rows, year_filter=year_filter)
        if response.startswith("정형 데이터와 재무제표 문맥에서 질문에 대응되는 수치를 확정하지 못했습니다."):
            return self._ask_narrative(
                query,
                stream=stream,
                contexts=contexts,
                tool_rows=tool_rows,
                is_multi=is_multi,
                max_new_tokens=max_new_tokens,
            )
        return self._finalize_text_response(response, stream)

    def _ask_hybrid(
        self,
        query: str,
        *,
        stream: bool,
        contexts: list[dict],
        tool_rows: list[dict],
        year_filter: list[int] | int | None = None,
        is_multi: bool,
        max_new_tokens: int,
    ) -> str | Generator:
        numeric_summary = self._build_numeric_answer(query, contexts, tool_rows, year_filter=year_filter)
        if numeric_summary.startswith("정형 데이터와 재무제표 문맥에서 질문에 대응되는 수치를 확정하지 못했습니다."):
            return self._ask_narrative(
                query,
                stream=stream,
                contexts=contexts,
                tool_rows=tool_rows,
                is_multi=is_multi,
                max_new_tokens=max_new_tokens,
            )

        system_prompt = """당신은 삼성전자 감사보고서(2014~2024) 전문 분석 어시스턴트입니다.

[답변 규칙]
1. 아래의 '확정된 정량 결과'는 코드가 계산한 값이므로 절대 수정하거나 다른 숫자로 바꾸지 마세요.
1-1. 정량 결과의 증감 방향/유출·유입 해석(예: 유출 규모 확대/축소)을 절대 뒤집거나 상충되게 서술하지 마세요.
2. 정량 결과를 먼저 짧게 요약하고, 이어서 참고 문서에 기반한 정성 설명을 덧붙이세요.
3. 참고 문서에 없는 해석은 추측하지 마세요.
4. 답변 끝에 근거 연도와 섹션을 간단히 표기하세요."""

        prompt_parts = [
            "[확정된 정량 결과]",
            numeric_summary,
            "",
            "[참고 문서 기반 설명용 문맥]",
            self.build_prompt(query, contexts),
        ]
        coverage_note = self._build_coverage_guard_note(query, contexts)
        if coverage_note:
            prompt_parts.insert(0, coverage_note)
            prompt_parts.insert(1, "")
        tool_context = self._format_tool_context(tool_rows)
        if tool_context:
            prompt_parts.insert(3, tool_context)
            prompt_parts.insert(4, "")
        user_prompt = "\n".join(prompt_parts)
        return self._dispatch_backend_response(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            stream=stream,
            max_new_tokens=max_new_tokens,
        )

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []
        # 한국어 문장 부호 + 줄바꿈 기반 단순 분할
        parts = re.split(r"(?<=[\.\?!다요])\s+|\n+", text)
        return [p.strip() for p in parts if len(p.strip()) >= 10]

    def extract_evidence_spans(self, query: str, contexts: list[dict], max_spans: int = 3) -> list[dict]:
        """
        문서 컨텍스트에서 질문 키워드와 수치(숫자)를 기준으로 근거 스팬을 추출.
        반환: [{"year":..., "section":..., "span":...}, ...]
        """
        keywords = [self._normalize_text(k) for k in self._extract_keywords(query)]
        preferred_sections = self._preferred_sections_for_query(query)
        if not contexts:
            return []

        candidates = []
        for ctx in contexts:
            clean = re.sub(r'^\[\d{4}년 [^\]]+\]\s*', '', ctx.get("text", ""))
            for sent in self._split_sentences(clean):
                norm_sent = self._normalize_text(sent)
                overlap = sum(1 for kw in keywords if kw and kw in norm_sent)
                has_number = bool(re.search(r"\d", sent))
                section_bonus = 3 if ctx.get("section") in preferred_sections else 0
                score = overlap * 2 + (1 if has_number else 0) + section_bonus
                if score <= 0:
                    continue
                candidates.append(
                    {
                        "year": ctx["year"],
                        "section": ctx["section"],
                        "span": sent,
                        "score": score,
                        "section_bonus": section_bonus,
                    }
                )

        if not candidates:
            return []

        preferred_candidates = [
            c for c in candidates if (not preferred_sections) or c["section"] in preferred_sections
        ]
        other_candidates = [c for c in candidates if c["section"] not in preferred_sections]
        preferred_candidates.sort(key=lambda x: (-x["score"], x["year"]))
        other_candidates.sort(key=lambda x: (-x["score"], x["year"]))

        ordered = preferred_candidates + other_candidates
        dedup = []
        seen = set()

        # 변경 여부 질의에서는 근거 연도의 대표성을 먼저 확보한다.
        if self._is_change_presence_query(query):
            seen_years = set()
            for c in ordered:
                key = (c["year"], c["section"], c["span"][:60])
                if key in seen or c["year"] in seen_years:
                    continue
                seen.add(key)
                seen_years.add(c["year"])
                dedup.append({"year": c["year"], "section": c["section"], "span": c["span"]})
                if len(dedup) >= max_spans:
                    return dedup

        for c in ordered:
            key = (c["year"], c["section"], c["span"][:60])
            if key in seen:
                continue
            seen.add(key)
            dedup.append({"year": c["year"], "section": c["section"], "span": c["span"]})
            if len(dedup) >= max_spans:
                break
        return dedup

    def retrieve(self, query: str, k: int = 4, year_filter: int | None = None) -> list[dict]:
        with self._index_lock:
            embed_model = get_embed_model()
            query_vec = embed_model.encode([query], normalize_embeddings=True)
            query_vec = query_vec.reshape(1, -1)

            # 임베딩 모델 변경 후 기존 인덱스가 남아 있는 경우를 자동 복구.
            query_dim = int(query_vec.shape[1])
            index_dim = self.vector_store.get_index_dim()
            if index_dim is not None and index_dim != query_dim:
                print(
                    f"Detected embedding dimension mismatch (query={query_dim}, index={index_dim}). "
                    "Triggering index rebuild."
                )
                self._rebuild_vector_index()

            # post-retrieval filtering 대비하여 k 오버샘플링
            search_k = k * 3 if year_filter else k
            try:
                semantic_results = self.vector_store.search(query_vec, k=search_k)
            except ValueError as exc:
                if "Embedding dimension mismatch" not in str(exc):
                    raise
                # 동시 실행/캐시 상태 등으로 재발하는 경우 한 번 더 강제 재빌드 후 재시도.
                print("Vector index mismatch persisted; forcing one more rebuild.")
                self._rebuild_vector_index()
                semantic_results = self.vector_store.search(query_vec, k=search_k)

            if year_filter is not None:
                semantic_results = [c for c in semantic_results if c["year"] == year_filter]

            keyword_results = self._keyword_search(query, k=k, year_filter=year_filter)

            # 재무수치 질문이면 target_section을 보호 슬롯으로 지정
            protect = self._detect_target_section(query)
            merged = self._merge_results(keyword_results, semantic_results, k=k, protect_section=protect, year_filter=year_filter)
            return merged

    def _keyword_search(self, query: str, k: int, year_filter: int | None) -> list[dict]:
        keywords = self._extract_keywords(query)
        if not keywords:
            return []
        
        # 재무수치 질문 여부 판단
        is_financial_query = any(kw in query for kw in _FINANCIAL_QUERY_KEYWORDS)
        
        results = []
        for meta in self.vector_store.chunks_meta:
            if year_filter and meta["year"] != year_filter:
                continue
            text = meta["text"]
            text_no_space = text.replace(" ", "")  # 원본 텍스트에 공백이 포함된 경우 대비본
            score = sum(1 for kw in keywords if kw in text or kw in text_no_space)
            
            if score > 0:
                # 재무수치 질문이고 재무제표 섹션이면 점수 5배 부여
                if is_financial_query and meta["section"] in _FINANCIAL_SECTIONS:
                    score *= 5
                results.append({**meta, "score": score / len(keywords)})
        
        results.sort(key=lambda x: -x["score"])
        return results[:k]

    def _merge_results(self, keyword_res: list, semantic_res: list,
                       k: int, protect_section: str | None = None, year_filter: int | None = None) -> list[dict]:
        """
        keyword_res를 우선 배치하고 semantic_res로 나머지를 채움.
        protect_section이 지정되면 해당 섹션 청크를 최소 1개 보장(강제 병합 포함).
        """
        seen = set()
        merged = []

        def _add(r):
            key = (r["year"], r["section"], r["text"][:50])
            if key not in seen:
                seen.add(key)
                merged.append(r)

        # 보호 섹션 청크를 먼저 1개 삽입
        if protect_section:
            found = False
            for r in keyword_res + semantic_res:
                if r["section"] == protect_section:
                    _add(r)
                    found = True
                    break  # 1개만
                    
            # 키워드 및 시맨틱 서치 결과에 후보가 단 하나도 없으면 원본 DB에서 강제 조달
            # 주석과 같이 거대한 섹션은 의미 없이 첫 번째 청크가 삽입될 위험이 있어 강제 조달 제외
            if not found and protect_section != "주석":
                for meta in self.vector_store.chunks_meta:
                    if meta["section"] == protect_section and (year_filter is None or meta["year"] == year_filter):
                        _add({**meta, "score": 1.0})
                        break

        # 나머지 채우기
        for r in keyword_res + semantic_res:
            if len(merged) >= k:
                break
            _add(r)

        return merged

    def build_prompt(self, query: str, contexts: list[dict]) -> str:
        prompt = ""
        for i, ctx in enumerate(contexts, 1):
            # 임베딩용으로 저장된 '[YYYY년 섹션]' prefix를 프롬프트에서는 제거 (메타데이터와 중복)
            clean_text = re.sub(r'^\[\d{4}년 [^\]]+\] ', '', ctx['text'])
            prompt += f"[참고 문서 {i}] (연도: {ctx['year']}, 섹션: {ctx['section']})\n{clean_text}\n---\n"
        prompt += f"\n질문: {query}"
        return prompt

    def get_sources(self, contexts: list[dict]) -> str:
        sources = [f"{c['year']}년 {c['section']}" for c in contexts]
        return ", ".join(list(dict.fromkeys(sources)))  # 중복 제거 제거 유지

    def _init_local_model(self):
        if self.model is not None:
            return
            
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            print(f"Loading local model: {self.local_model}")
            device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {device}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.local_model)
            model_kwargs = {
                "torch_dtype": torch.float16 if device in ("cuda", "mps") else torch.float32,
            }
            # device_map은 CUDA에서만 자동 샤딩으로 사용하고, MPS/CPU는 명시적으로 .to(device) 이동
            if device == "cuda":
                model_kwargs["device_map"] = "auto"

            self.model = AutoModelForCausalLM.from_pretrained(self.local_model, **model_kwargs)
            if device in ("mps", "cpu"):
                self.model = self.model.to(device)
            self.model.eval()
        except Exception as e:
            print(f"Error loading local model: {e}")
            self.model = None
            self.tokenizer = None
            raise RuntimeError(
                f"Local model load failed for '{self.local_model}': {e}"
            ) from e

    def get_contexts(
        self,
        query: str,
        year_filter: list[int] | int | None = None,
        retrieve_k: int | None = None,
    ) -> list[dict]:
        """ask() 및 Streamlit UI에서 공통으로 사용할 문서들을 미리 가져옵니다. 쿼리 유형에 따라 단일/멀티 연도 라우팅을 수행합니다."""
        effective_k = retrieve_k if retrieve_k is not None else self.retrieve_k
        if effective_k < 1:
            raise ValueError("retrieve_k must be >= 1.")

        if year_filter is None:
            detected_years = self._auto_detect_years(query)
        elif isinstance(year_filter, int):
            detected_years = [year_filter]
        else:
            detected_years = year_filter

        has_multi_pattern = self._is_multi_year_query(query)
        
        if len(detected_years) == 1:
            is_multi = False # 단일 연도가 명시되었으면 단일 연도로 취급
        elif len(detected_years) > 1:
            is_multi = True # 여러 연도가 명시되면 멀티 연도로 취급
        else:
            is_multi = has_multi_pattern

        if is_multi:
            target_years = detected_years if detected_years else list(range(2014, 2025))
            # "변경 여부/없다" 류 질문에서 연도를 명시하지 않았다면
            # 전체 기간을 확인한 뒤에만 단정할 수 있도록 최소 1문서/연도를 강제한다.
            if (
                not detected_years
                and self._is_change_presence_query(query)
                and len(target_years) > effective_k
            ):
                effective_k = len(target_years)
            # 연도 커버리지가 중요한 질의(매년/역대 최대·최소)는
            # k가 연도 수보다 작아도 내부적으로 자동 상향해 연도 누락을 방지한다.
            if (self._is_yearly_query(query) or self._is_extrema_query(query)) and len(target_years) > effective_k:
                effective_k = len(target_years)
            contexts = self._retrieve_multi_year(query, total_k=effective_k, target_years=target_years)
        else:
            target_year = detected_years[0] if detected_years else None
            contexts = self.retrieve(query, k=effective_k, year_filter=target_year)

        return self._filter_contexts_by_topic_terms(query, contexts)

    def _compute_max_tokens(self, query: str, is_multi: bool) -> int:
        """쿼리 유형에 따라 적절한 max_new_tokens를 결정.
        - 단일 연도 / yes·no 성격 질문: 512 (속도 우선)
        - 멀티연도 트렌드·설명 질문: self.max_new_tokens (품질 우선)
        """
        if not is_multi:
            return min(self.max_new_tokens, 512)
        # 멀티연도이더라도 상세 서술이 필요한 경우에만 풀 토큰 사용
        detail_markers = {"트렌드", "추이", "설명", "어떻게", "비교", "분석"}
        if any(m in query for m in detail_markers):
            return self.max_new_tokens
        # 변경 여부, 있나요 형 yes/no 멀티 쿼리는 짧게
        return min(self.max_new_tokens, 512)

    def ask(
        self,
        query: str,
        year_filter: list[int] | int | None = None,
        stream: bool = False,
        contexts: list[dict] | None = None,
        tool_rows: list[dict] | None = None,
        retrieve_k: int | None = None,
    ) -> str | Generator:
        # 1. 쿼리 유형 분기 파악용
        if year_filter is None:
            detected_years = self._auto_detect_years(query)
        elif isinstance(year_filter, int):
            detected_years = [year_filter]
        else:
            detected_years = year_filter

        has_multi_pattern = self._is_multi_year_query(query)

        if len(detected_years) == 1:
            is_multi = False
        elif len(detected_years) > 1:
            is_multi = True
        else:
            is_multi = has_multi_pattern

        effective_max_tokens = self._compute_max_tokens(query, is_multi)
        query_type = self.route_query_type(query, year_filter=year_filter)

        # 2. Contexts 준비 (앱에서 전달되지 않았을 경우 자체 로드)
        if contexts is None:
            contexts = self.get_contexts(query, year_filter, retrieve_k=retrieve_k)

        if tool_rows is None:
            tool_rows = self.run_tool_calling(query, year_filter=year_filter)

        if query_type == "numeric":
            return self._ask_numeric(
                query,
                stream=stream,
                contexts=contexts,
                tool_rows=tool_rows,
                year_filter=year_filter,
                is_multi=is_multi,
                max_new_tokens=effective_max_tokens,
            )
        if query_type == "hybrid":
            return self._ask_hybrid(
                query,
                stream=stream,
                contexts=contexts,
                tool_rows=tool_rows,
                year_filter=year_filter,
                is_multi=is_multi,
                max_new_tokens=effective_max_tokens,
            )
        return self._ask_narrative(
            query,
            stream=stream,
            contexts=contexts,
            tool_rows=tool_rows,
            is_multi=is_multi,
            max_new_tokens=effective_max_tokens,
        )

    def _ask_anthropic(self, user_prompt: str, stream: bool, system_prompt: str = RAG_SYSTEM):
        import anthropic
        client = anthropic.Anthropic()
        messages = [{"role": "user", "content": user_prompt}]
        
        if stream:
            def streamer():
                with client.messages.stream(
                    max_tokens=2048,
                    system=system_prompt,
                    messages=messages,
                    model="claude-3-5-haiku-20241022",
                ) as response:
                    for text in response.text_stream:
                        yield text
            return streamer()
        else:
            message = client.messages.create(
                max_tokens=2048,
                system=system_prompt,
                messages=messages,
                model="claude-3-5-haiku-20241022",
            )
            return message.content[0].text

    def _ask_openai(self, user_prompt: str, stream: bool, system_prompt: str = RAG_SYSTEM):
        from openai import OpenAI
        client = OpenAI()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        if stream:
            def streamer():
                response = client.chat.completions.create(
                    model=self.openai_model,
                    messages=messages,
                    stream=True
                )
                for chunk in response:
                    text = chunk.choices[0].delta.content
                    if text:
                        yield text
            return streamer()
        else:
            response = client.chat.completions.create(
                model=self.openai_model,
                messages=messages
            )
            return response.choices[0].message.content

    def _ask_local(self, user_prompt: str, stream: bool, system_prompt: str = RAG_SYSTEM, max_new_tokens: int | None = None):
        self._init_local_model()
        if self.model is None or self.tokenizer is None:
            raise RuntimeError(
                "Local model is not initialized. "
                "Please check LOCAL_MODEL / memory / transformers compatibility."
            )
        effective_max_tokens = max_new_tokens if max_new_tokens is not None else self.max_new_tokens
        from transformers import TextIteratorStreamer

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        text_input = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text_input, return_tensors="pt").to(self.model.device)

        if stream:
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=effective_max_tokens, do_sample=False)
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()

            def gen_stream():
                for piece in streamer:
                    yield piece
            return gen_stream()
        else:
            outputs = self.model.generate(**inputs, max_new_tokens=effective_max_tokens, do_sample=False)
            response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
            return response
