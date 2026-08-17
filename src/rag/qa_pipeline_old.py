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

# 감사보고서 섹션별 대표 키워드 — 섹션 타겟 검색에 사용 (소규모 섹션 보호용)
_SECTION_TARGET_MAP = {
    "감사의견": ["감사의견", "적정의견", "한정의견", "부적정"],
    "핵심감사사항": ["핵심감사사항", "핵심감사", "리스크", "위험", "감사위험", "재무위험"],
    "감사의견근거": ["감사의견근거", "감사인의 책임"],
    "포괄손익": ["영업이익", "매출", "순이익", "당기순이익", "영업손익"],
    "재무상태표": ["자산총계", "부채총계", "자본총계"],
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
    def __init__(self, csv_path: str = "data/processed/sections.csv", rebuild: bool = False):
        self._load_runtime_config()
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

    def _load_runtime_config(self) -> None:
        """실행 시점 환경변수를 읽어 백엔드/모델 설정을 확정."""
        has_openai_key = bool(os.getenv("OPENAI_API_KEY"))
        backend_default = "openai" if has_openai_key else "local"
        backend = os.getenv("LLM_BACKEND", backend_default).strip().lower()

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
        self.local_model = os.getenv("LOCAL_MODEL", _DEFAULT_LOCAL_MODEL)
        self.retrieve_k = retrieve_k
        self.max_new_tokens = max_new_tokens

    def _rebuild_vector_index(self) -> None:
        """현재 임베딩 모델 기준으로 FAISS 인덱스를 재생성."""
        with self._index_lock:
            print("Rebuilding FAISS index to match current embedding model...")
            chunks = load_chunks_from_csv(self.csv_path)
            embeddings = build_embeddings(chunks)
            self.vector_store.build(chunks, embeddings)

    def _auto_detect_years(self, query: str) -> list[int]:
        """질문 텍스트에서 연도를 모두 감지. 예: '2020년', '2018년과 2019년' → [2018, 2019]"""
        m = set(re.findall(r"(20\d{2})년?", query))
        years = [int(y) for y in m if 2014 <= int(y) <= 2024]
        return sorted(years)

    def _is_multi_year_query(self, query: str) -> bool:
        """멀티연도 비교·트렌드 질문 여부 판단"""
        return any(re.search(p, query) for p in _MULTI_YEAR_PATTERNS)

    def _detect_target_section(self, query: str) -> str | None:
        """질문 키워드로 가장 관련 높은 섹션명 추론. 없으면 None."""
        for section, keywords in _SECTION_TARGET_MAP.items():
            if any(kw in query for kw in keywords):
                return section
        return None

    def _retrieve_multi_year(self, query: str, k_per_year: int = 1, target_years: list[int] | None = None) -> list[dict]:
        """
        멀티연도 질문용: 각 연도에서 k_per_year개씩 균등 검색.
        protect_section이 감지된 경우, 해당 섹션 청크를 실제로 찾지 못한 연도는 건너뜀.
        (예: 핵심감사사항이 없는 2014~2017년에 재무상태표를 혼합해 환각을 유발하는 문제 방지)
        """
        years = target_years if target_years else list(range(2014, 2025))
        all_contexts = []
        protect = self._detect_target_section(query)

        for year in years:
            candidates = self.retrieve(query, k=max(2, k_per_year * 2), year_filter=year)
            # protect_section이 지정됐는데 해당 연도 결과에 실제로 없으면 스킵
            # → 이질적인 섹션 혼합으로 인한 LLM 환각 방지
            if protect and not any(c["section"] == protect for c in candidates):
                continue
            all_contexts.extend(candidates[:k_per_year])

        return all_contexts

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

    def _lookup_financial_rows(self, query: str, target_years: list[int]) -> list[dict]:
        """정형 재무 CSV를 조회해 질문과 직접 매칭되는 행을 반환."""
        df = _load_financial_df()
        if df.empty:
            return []

        keywords = self._extract_keywords(query)
        if not keywords:
            return []

        # query normalization 결과와 item normalization 결과를 비교해 점수화
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
        return rows[:6]

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

        return self._lookup_financial_rows(query, years)

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
        if not contexts:
            return []

        candidates = []
        for ctx in contexts:
            clean = re.sub(r'^\[\d{4}년 [^\]]+\]\s*', '', ctx.get("text", ""))
            for sent in self._split_sentences(clean):
                norm_sent = self._normalize_text(sent)
                overlap = sum(1 for kw in keywords if kw and kw in norm_sent)
                has_number = bool(re.search(r"\d", sent))
                score = overlap * 2 + (1 if has_number else 0)
                if score <= 0:
                    continue
                candidates.append(
                    {
                        "year": ctx["year"],
                        "section": ctx["section"],
                        "span": sent,
                        "score": score,
                    }
                )

        if not candidates:
            return []

        candidates.sort(key=lambda x: (-x["score"], x["year"]))
        dedup = []
        seen = set()
        for c in candidates:
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
            k_py = max(1, effective_k // max(1, len(target_years)))
            return self._retrieve_multi_year(query, k_per_year=k_py, target_years=target_years)
        else:
            target_year = detected_years[0] if detected_years else None
            return self.retrieve(query, k=effective_k, year_filter=target_year)

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

        system_prompt = RAG_SYSTEM_MULTI if is_multi else RAG_SYSTEM
        effective_max_tokens = self._compute_max_tokens(query, is_multi)

        # 2. Contexts 준비 (앱에서 전달되지 않았을 경우 자체 로드)
        if contexts is None:
            contexts = self.get_contexts(query, year_filter, retrieve_k=retrieve_k)

        tool_rows = self.run_tool_calling(query, year_filter=year_filter)
        tool_context = self._format_tool_context(tool_rows)
        user_prompt = self.build_prompt(query, contexts)
        if tool_context:
            user_prompt = f"{tool_context}\n\n{user_prompt}"

        if self.backend == "anthropic":
            return self._ask_anthropic(user_prompt, stream, system_prompt)
        elif self.backend == "openai":
            return self._ask_openai(user_prompt, stream, system_prompt)
        else:
            return self._ask_local(user_prompt, stream, system_prompt, max_new_tokens=effective_max_tokens)

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
                    model="gpt-4o-mini",
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
                model="gpt-4o-mini",
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
