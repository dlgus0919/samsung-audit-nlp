"""
RAG 파이프라인 단위 테스트
- 실제 감사보고서 데이터에 의존하지 않고 소형 mock 데이터로 동작 검증
"""
import re
import sys
import tempfile
import os
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(autouse=True)
def mock_embed_model(monkeypatch):
    """네트워크 없이도 동작하도록 임베딩 모델을 테스트 더블로 고정."""
    from src.rag import embedder

    class DummyEmbedModel:
        def encode(self, texts, normalize_embeddings=True, **kwargs):
            if isinstance(texts, str):
                texts = [texts]

            dim = 16
            out = np.zeros((len(texts), dim), dtype=np.float32)
            for i, text in enumerate(texts):
                rng = np.random.default_rng(abs(hash(text)) % (2**32))
                vec = rng.standard_normal(dim).astype(np.float32)
                if normalize_embeddings:
                    norm = np.linalg.norm(vec)
                    if norm > 0:
                        vec /= norm
                out[i] = vec
            return out

    model = DummyEmbedModel()
    monkeypatch.setattr(embedder, "get_embed_model", lambda: model)


# ──────────────────────────────────────────────
# embedder
# ──────────────────────────────────────────────

def _make_csv(tmp_path: Path, rows: list[dict]) -> str:
    path = str(tmp_path / "sections.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_load_chunks_from_csv_basic(tmp_path):
    """기본 CSV 로딩 및 Chunk 구조 검증"""
    from src.rag.embedder import load_chunks_from_csv, Chunk, CHUNK_SIZE

    csv_path = _make_csv(tmp_path, [
        {"year": 2021, "section": "재무상태표", "content": "자산총계 100,000 부채총계 50,000"},
        {"year": 2022, "section": "포괄손익",   "content": "매출액 200,000 영업이익 30,000"},
    ])
    chunks = load_chunks_from_csv(csv_path)
    assert len(chunks) > 0
    for c in chunks:
        assert isinstance(c, Chunk)
        assert c.year in (2021, 2022)
        assert c.section in ("재무상태표", "포괄손익")
        assert len(c.text) >= 30


def test_load_chunks_invalid_columns(tmp_path):
    """필수 컬럼 누락 시 ValueError 발생"""
    from src.rag.embedder import load_chunks_from_csv

    bad_csv = str(tmp_path / "bad.csv")
    pd.DataFrame([{"yr": 2021, "body": "텍스트"}]).to_csv(bad_csv, index=False)

    with pytest.raises(ValueError, match="필수 컬럼 누락"):
        load_chunks_from_csv(bad_csv)


def test_load_chunks_skips_empty_content(tmp_path):
    """빈 content 행은 스킵되고 나머지는 정상 로딩"""
    from src.rag.embedder import load_chunks_from_csv

    csv_path = _make_csv(tmp_path, [
        {"year": 2021, "section": "주석", "content": ""},
        {"year": 2021, "section": "주석", "content": None},
        {"year": 2021, "section": "주석", "content": "유효한 내용이 포함된 텍스트입니다."},
    ])
    chunks = load_chunks_from_csv(csv_path)
    assert len(chunks) >= 1
    for c in chunks:
        assert c.text.strip() != ""


def test_load_chunks_financial_repeat(tmp_path):
    """재무제표 섹션은 REPEAT_COUNT 배수로 청크가 생성됨"""
    from src.rag.embedder import load_chunks_from_csv, REPEAT_COUNT, FINANCIAL_SECTIONS_REPEAT, CHUNK_SIZE

    content = "A" * (CHUNK_SIZE - 10)  # 청크 1개 분량
    section = next(iter(FINANCIAL_SECTIONS_REPEAT))  # 예: "포괄손익"
    csv_path = _make_csv(tmp_path, [
        {"year": 2021, "section": section, "content": content},
    ])
    chunks = load_chunks_from_csv(csv_path)
    assert len(chunks) == REPEAT_COUNT


def test_build_embeddings_shape(tmp_path):
    """임베딩 반환 shape (N, D) 및 dtype=float32 검증"""
    from src.rag.embedder import load_chunks_from_csv, build_embeddings

    csv_path = _make_csv(tmp_path, [
        {"year": 2021, "section": "감사의견", "content": "감사인은 적정의견을 표명합니다."},
        {"year": 2022, "section": "감사의견", "content": "재무제표는 공정하게 표시됩니다."},
    ])
    chunks = load_chunks_from_csv(csv_path)
    embeddings = build_embeddings(chunks)

    assert embeddings.ndim == 2
    assert embeddings.shape[0] == len(chunks)
    assert embeddings.shape[1] > 0
    assert embeddings.dtype == np.float32


def test_build_embeddings_empty():
    """빈 청크 리스트 입력 시 (0, 0) 배열 반환"""
    from src.rag.embedder import build_embeddings
    result = build_embeddings([])
    assert result.shape == (0, 0)


# ──────────────────────────────────────────────
# vector_store
# ──────────────────────────────────────────────

def _make_store_with_data(tmp_path: Path):
    """소형 FAISS 인덱스를 빌드하여 VectorStore 반환"""
    from src.rag.embedder import load_chunks_from_csv, build_embeddings
    from src.rag.vector_store import VectorStore

    csv_path = _make_csv(tmp_path, [
        {"year": 2020, "section": "재무상태표", "content": "자산총계는 100,000백만원이며 부채총계는 50,000백만원입니다."},
        {"year": 2021, "section": "포괄손익",   "content": "당기 매출액은 200,000백만원이고 영업이익은 30,000백만원입니다."},
        {"year": 2022, "section": "현금흐름",   "content": "영업활동으로 인한 현금흐름은 50,000백만원 유입되었습니다."},
    ])
    chunks = load_chunks_from_csv(csv_path)
    embeddings = build_embeddings(chunks)

    index_path = str(tmp_path / "test.index")
    meta_path  = str(tmp_path / "test_meta.pkl")
    store = VectorStore(index_path=index_path, meta_path=meta_path)
    store.build(chunks, embeddings)
    return store, chunks


def test_vector_store_build_and_search(tmp_path):
    """FAISS 빌드 후 검색 결과가 올바른 구조를 반환하는지 검증"""
    store, chunks = _make_store_with_data(tmp_path)

    from src.rag.embedder import get_embed_model
    model = get_embed_model()
    q_vec = model.encode(["자산총계"], normalize_embeddings=True).reshape(1, -1)

    results = store.search(q_vec, k=2)
    assert len(results) <= 2
    for r in results:
        assert "text" in r and "year" in r and "section" in r and "score" in r


def test_vector_store_is_built(tmp_path):
    """build() 후 is_built() == True"""
    store, _ = _make_store_with_data(tmp_path)
    assert store.is_built()


def test_vector_store_dimension_mismatch(tmp_path):
    """차원이 다른 쿼리 벡터 입력 시 ValueError 발생"""
    store, _ = _make_store_with_data(tmp_path)

    wrong_vec = np.random.randn(1, 4).astype(np.float32)  # 차원 4 (불일치)
    with pytest.raises(ValueError, match="Embedding dimension mismatch"):
        store.search(wrong_vec, k=1)


# ──────────────────────────────────────────────
# qa_pipeline
# ──────────────────────────────────────────────

def test_extract_keywords():
    """조사/어미 제거 및 불용어 필터링 후 핵심 키워드 추출"""
    from src.rag.qa_pipeline import RAGPipeline
    # RAGPipeline 인스턴스 없이 _extract_keywords 호출하기 위해 임시 객체 생성 불가
    # 메서드를 언바운드로 직접 호출
    pipeline_cls = RAGPipeline
    dummy = object.__new__(pipeline_cls)

    keywords = dummy._extract_keywords("삼성전자의 부채비율을 알려주세요")
    assert "삼성전자" in keywords or "부채비율" in keywords or "부채" in keywords


def test_build_prompt_no_duplicate_prefix(tmp_path):
    """build_prompt 결과에서 '[YYYY년 섹션]' prefix가 본문에 중복되지 않음"""
    from src.rag.qa_pipeline import RAGPipeline

    dummy = object.__new__(RAGPipeline)

    contexts = [
        {"year": 2021, "section": "재무상태표", "text": "[2021년 재무상태표] 자산총계 100,000"},
        {"year": 2022, "section": "포괄손익",   "text": "[2022년 포괄손익] 매출액 200,000"},
    ]
    prompt = dummy.build_prompt("자산총계를 알려주세요", contexts)

    # prefix가 제거되어 본문에 '[2021년 재무상태표]' 같은 패턴이 없어야 함
    assert not re.search(r'\[\d{4}년 [^\]]+\]', prompt), \
        "build_prompt 결과에 '[YYYY년 섹션]' prefix가 남아 있음"
    # 메타데이터 표기는 유지되어야 함
    assert "(연도: 2021" in prompt
    assert "자산총계" in prompt


def test_auto_detect_years():
    """연도 감지 로직이 질문에서 연도를 정확히 추출"""
    from src.rag.qa_pipeline import RAGPipeline
    dummy = object.__new__(RAGPipeline)

    years = dummy._auto_detect_years("2018년과 2019년 현금흐름을 비교해주세요")
    assert set(years) == {2018, 2019}


def test_auto_detect_years_expands_range_expression():
    """연도 범위 표현(부터~까지, ~)을 1년 단위로 확장"""
    from src.rag.qa_pipeline import RAGPipeline
    dummy = object.__new__(RAGPipeline)

    years = dummy._auto_detect_years("총 부채는 2018년부터 2024년까지 매년 어떻게 변화했나요?")
    assert years == [2018, 2019, 2020, 2021, 2022, 2023, 2024]

    years2 = dummy._auto_detect_years("2019~2021 현금흐름 추이")
    assert years2 == [2019, 2020, 2021]


def test_is_multi_year_query():
    """멀티연도 패턴 감지"""
    from src.rag.qa_pipeline import RAGPipeline
    dummy = object.__new__(RAGPipeline)

    assert dummy._is_multi_year_query("감사의견 추이를 알려주세요") is True
    assert dummy._is_multi_year_query("2021년 매출액은?") is False


def test_extract_evidence_spans():
    """근거 스팬 추출 시 year/section/span 구조가 유지되고 max_spans를 넘지 않음"""
    from src.rag.qa_pipeline import RAGPipeline
    dummy = object.__new__(RAGPipeline)

    contexts = [
        {"year": 2021, "section": "포괄손익", "text": "[2021년 포괄손익] 매출은 100이고 영업이익은 30입니다. 기타 설명입니다."},
        {"year": 2022, "section": "포괄손익", "text": "[2022년 포괄손익] 매출은 120이고 영업이익은 32입니다."},
    ]
    spans = dummy.extract_evidence_spans("매출과 영업이익 추이를 알려주세요", contexts, max_spans=2)
    assert len(spans) <= 2
    if spans:
        assert {"year", "section", "span"}.issubset(spans[0].keys())


def test_format_tool_context():
    """도구 조회 결과가 프롬프트 컨텍스트 문자열로 직렬화됨"""
    from src.rag.qa_pipeline import RAGPipeline
    dummy = object.__new__(RAGPipeline)
    text = dummy._format_tool_context([
        {
            "year": 2021,
            "table_type": "income_statement",
            "item": "매출액",
            "values": ["100", "90"],
            "score": 3.0,
        }
    ])
    assert "도구 조회 결과" in text
    assert "2021년" in text
    assert "매출액" in text


def test_run_tool_calling_uses_range_expansion_and_limit():
    """매년/범위 질의에서 연도 범위 확장 및 limit>=연도수 적용"""
    from src.rag.qa_pipeline import RAGPipeline

    dummy = object.__new__(RAGPipeline)
    captured = {}

    def _fake_lookup(query, target_years, limit=6):
        captured["years"] = target_years
        captured["limit"] = limit
        return []

    dummy._lookup_financial_rows = _fake_lookup
    dummy.run_tool_calling("총 부채는 2018년부터 2024년까지 매년 어떻게 변화했나요?")

    assert captured["years"] == [2018, 2019, 2020, 2021, 2022, 2023, 2024]
    assert captured["limit"] >= 7


def test_run_tool_calling_extrema_without_year_uses_full_range():
    """연도 미지정 '역대 최대/최소' 질의는 2014~2024 전체를 조회"""
    from src.rag.qa_pipeline import RAGPipeline

    dummy = object.__new__(RAGPipeline)
    captured = {}

    def _fake_lookup(query, target_years, limit=6):
        captured["years"] = target_years
        captured["limit"] = limit
        return []

    dummy._lookup_financial_rows = _fake_lookup
    dummy.run_tool_calling("역대 총 부채액이 최대인 해는 언제인가요?")

    assert captured["years"] == list(range(2014, 2025))
    assert captured["limit"] >= 11


def test_pick_year_diverse_tool_rows_prioritizes_year_coverage():
    """연도별 질문에서는 각 연도를 우선 포함하도록 도구 행을 선택"""
    from src.rag.qa_pipeline import RAGPipeline

    dummy = object.__new__(RAGPipeline)
    rows = [
        {"year": 2018, "table_type": "balance_sheet", "item": "부채A", "values": ["1"], "score": 9.0},
        {"year": 2018, "table_type": "balance_sheet", "item": "부채B", "values": ["2"], "score": 8.0},
        {"year": 2019, "table_type": "balance_sheet", "item": "부채A", "values": ["3"], "score": 7.0},
        {"year": 2020, "table_type": "balance_sheet", "item": "부채A", "values": ["4"], "score": 6.0},
    ]
    picked = dummy._pick_year_diverse_tool_rows(rows, target_years=[2018, 2019, 2020], limit=3)

    assert [r["year"] for r in picked] == [2018, 2019, 2020]


def test_lookup_financial_rows_extrema_enforces_year_coverage(monkeypatch):
    """extrema 질의에서도 연도 커버리지 우선 선택이 동작함"""
    from src.rag import qa_pipeline

    dummy = object.__new__(qa_pipeline.RAGPipeline)
    df = pd.DataFrame([
        {"year": 2014, "table_type": "balance_sheet", "item": "부 채 총 계", "value_raw": '["1"]'},
        {"year": 2015, "table_type": "balance_sheet", "item": "부 채 총 계", "value_raw": '["2"]'},
        {"year": 2016, "table_type": "balance_sheet", "item": "부 채 총 계", "value_raw": '["3"]'},
    ])
    monkeypatch.setattr(qa_pipeline, "_load_financial_df", lambda csv_path=qa_pipeline._FINANCIAL_CSV_PATH: df)

    rows = dummy._lookup_financial_rows(
        "역대 총 부채액이 최대인 해는 언제인가요?",
        target_years=[2014, 2015, 2016],
        limit=3,
    )
    assert [r["year"] for r in rows] == [2014, 2015, 2016]


def test_lookup_financial_rows_prefers_aggregate_for_total_liability(monkeypatch):
    """'총 부채' 질의에서는 기타 세부항목보다 집계성 항목을 우선 선택"""
    from src.rag import qa_pipeline

    dummy = object.__new__(qa_pipeline.RAGPipeline)
    df = pd.DataFrame([
        {"year": 2018, "table_type": "balance_sheet", "item": "10. 기타유동부채", "value_raw": '["100"]'},
        {"year": 2018, "table_type": "balance_sheet", "item": "Ⅰ. 유 동 부 채", "value_raw": '["1000"]'},
        {"year": 2018, "table_type": "balance_sheet", "item": "Ⅱ. 비 유 동 부 채", "value_raw": '["500"]'},
    ])
    monkeypatch.setattr(qa_pipeline, "_load_financial_df", lambda csv_path=qa_pipeline._FINANCIAL_CSV_PATH: df)

    rows = dummy._lookup_financial_rows(
        "총 부채는 2018년부터 2018년까지 매년 어떻게 변화했나요?",
        target_years=[2018],
        limit=1,
    )
    assert rows
    assert rows[0]["item"] != "10. 기타유동부채"


def test_detect_target_section_for_total_liability_query():
    """총 부채 질의는 재무상태표 섹션 보호 대상으로 감지된다."""
    from src.rag.qa_pipeline import RAGPipeline
    dummy = object.__new__(RAGPipeline)

    section = dummy._detect_target_section("총 부채는 2018년부터 2024년까지 매년 어떻게 변화했나요?")
    assert section == "재무상태표"


def test_get_contexts_applies_retrieve_k_override_without_year_detected():
    """연도가 미검출된 단일 경로에서도 retrieve_k 오버라이드가 적용됨"""
    from src.rag.qa_pipeline import RAGPipeline

    dummy = object.__new__(RAGPipeline)
    dummy.retrieve_k = 4
    dummy._auto_detect_years = lambda q: []
    dummy._is_multi_year_query = lambda q: False
    captured = {}

    def _fake_retrieve(query, k, year_filter):
        captured["k"] = k
        captured["year_filter"] = year_filter
        return [{"year": 2021, "section": "감사의견", "text": "x"}]

    dummy.retrieve = _fake_retrieve
    rows = dummy.get_contexts("감사의견 알려줘", retrieve_k=7)

    assert rows
    assert captured["k"] == 7


def test_get_contexts_applies_retrieve_k_single_year():
    """retrieve_k 오버라이드가 단일 연도 검색 경로에 반영됨"""
    from src.rag.qa_pipeline import RAGPipeline

    dummy = object.__new__(RAGPipeline)
    dummy.retrieve_k = 4
    dummy._auto_detect_years = lambda q: [2021]
    dummy._is_multi_year_query = lambda q: False
    captured = {}

    def _fake_retrieve(query, k, year_filter):
        captured["k"] = k
        captured["year_filter"] = year_filter
        return [{"year": 2021, "section": "감사의견", "text": "x"}]

    dummy.retrieve = _fake_retrieve
    rows = dummy.get_contexts("2021년 감사의견", retrieve_k=7)

    assert rows
    assert captured["k"] == 7
    assert captured["year_filter"] == 2021


def test_get_contexts_applies_retrieve_k_multi_year():
    """멀티연도 경로에서 retrieve_k가 total_k로 전달됨"""
    from src.rag.qa_pipeline import RAGPipeline

    dummy = object.__new__(RAGPipeline)
    dummy.retrieve_k = 4
    dummy._auto_detect_years = lambda q: [2018, 2019, 2020]
    dummy._is_multi_year_query = lambda q: True
    captured = {}

    def _fake_multi(query, total_k, target_years):
        captured["total_k"] = total_k
        captured["target_years"] = target_years
        return [{"year": 2018, "section": "현금흐름", "text": "x"}]

    dummy._retrieve_multi_year = _fake_multi
    rows = dummy.get_contexts("2018~2020 현금흐름 비교", retrieve_k=9)

    assert rows
    assert captured["total_k"] == 9
    assert captured["target_years"] == [2018, 2019, 2020]


def test_get_contexts_auto_expands_k_for_yearly_query():
    """연도 수 > k 인 매년 질의에서는 total_k를 연도 수로 자동 상향"""
    from src.rag.qa_pipeline import RAGPipeline

    dummy = object.__new__(RAGPipeline)
    dummy.retrieve_k = 4
    dummy._auto_detect_years = lambda q: [2018, 2019, 2020, 2021, 2022, 2023, 2024]
    dummy._is_multi_year_query = lambda q: True
    captured = {}

    def _fake_multi(query, total_k, target_years):
        captured["total_k"] = total_k
        captured["target_years"] = target_years
        return [{"year": y, "section": "재무상태표", "text": "x"} for y in target_years[:total_k]]

    dummy._retrieve_multi_year = _fake_multi
    rows = dummy.get_contexts("2018년부터 2024년까지 총 부채는 매년 어떻게 변화했나요?", retrieve_k=4)

    assert rows
    assert captured["target_years"] == [2018, 2019, 2020, 2021, 2022, 2023, 2024]
    assert captured["total_k"] == 7


def test_get_contexts_auto_expands_k_for_extrema_query_without_years():
    """연도 미지정 역대 질의는 전체 연도 수(2014~2024)로 total_k 자동 상향"""
    from src.rag.qa_pipeline import RAGPipeline

    dummy = object.__new__(RAGPipeline)
    dummy.retrieve_k = 4
    dummy._auto_detect_years = lambda q: []
    dummy._is_multi_year_query = lambda q: True  # '역대' 패턴 등으로 멀티 판정
    captured = {}

    def _fake_multi(query, total_k, target_years):
        captured["total_k"] = total_k
        captured["target_years"] = target_years
        return [{"year": y, "section": "재무상태표", "text": "x"} for y in target_years[:total_k]]

    dummy._retrieve_multi_year = _fake_multi
    rows = dummy.get_contexts("역대 총 부채액이 최대인 해는 언제인가요?", retrieve_k=4)

    assert rows
    assert captured["target_years"] == list(range(2014, 2025))
    assert captured["total_k"] == 11


def test_ask_accepts_retrieve_k_keyword():
    """ask(..., retrieve_k=...) 호출이 TypeError 없이 동작"""
    from src.rag.qa_pipeline import RAGPipeline

    dummy = object.__new__(RAGPipeline)
    dummy.backend = "openai"
    dummy.max_new_tokens = 1024
    dummy._auto_detect_years = lambda q: []
    dummy._is_multi_year_query = lambda q: False
    dummy._compute_max_tokens = lambda q, m: 512
    dummy.get_contexts = lambda q, y, retrieve_k=None: [{"year": 2021, "section": "감사의견", "text": "x"}]
    dummy.build_prompt = lambda q, c: "prompt"
    dummy._ask_openai = lambda prompt, stream, system_prompt: "ok"

    res = dummy.ask("질문", retrieve_k=8)
    assert res == "ok"


def test_runtime_config_accepts_explicit_backend_and_models(monkeypatch):
    """UI에서 전달한 백엔드/모델 인자가 환경변수보다 우선한다."""
    from src.rag.qa_pipeline import RAGPipeline

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("LLM_BACKEND", "local")
    monkeypatch.setenv("OPENAI_MODEL", "env-openai-model")
    monkeypatch.setenv("LOCAL_MODEL", "env-local-model")

    dummy = object.__new__(RAGPipeline)
    dummy._load_runtime_config(
        backend="openai",
        openai_model="ui-openai-model",
        local_model="ui-local-model",
    )

    assert dummy.backend == "openai"
    assert dummy.openai_model == "ui-openai-model"
    assert dummy.local_model == "ui-local-model"


def test_runtime_config_explicit_openai_requires_api_key(monkeypatch):
    """OpenAI 백엔드는 UI 선택이어도 API 키가 없으면 초기화하지 않는다."""
    from src.rag.qa_pipeline import RAGPipeline

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("LLM_BACKEND", "local")

    dummy = object.__new__(RAGPipeline)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        dummy._load_runtime_config(backend="openai")


def test_retrieve_multi_year_respects_total_k():
    """_retrieve_multi_year는 total_k를 상한으로 결과를 반환"""
    from src.rag.qa_pipeline import RAGPipeline

    dummy = object.__new__(RAGPipeline)
    dummy._detect_target_section = lambda q: None

    def _fake_retrieve(query, k, year_filter):
        return [
            {"year": year_filter, "section": "감사의견", "text": f"{year_filter}-A", "score": 0.9},
            {"year": year_filter, "section": "감사의견", "text": f"{year_filter}-B", "score": 0.8},
        ]

    dummy.retrieve = _fake_retrieve
    rows = dummy._retrieve_multi_year("감사의견 추이", total_k=2, target_years=[2018, 2019, 2020])
    assert len(rows) == 2


def test_extract_metric_value_from_context_avoids_number_concatenation():
    """문맥 숫자 추출 시 공백 제거로 인한 숫자 결합이 발생하지 않아야 한다."""
    from src.rag.qa_pipeline import RAGPipeline

    dummy = object.__new__(RAGPipeline)
    context = (
        "Ⅰ. 매 출 액\n"
        "29\n"
        "199,744,705\n"
        "166,311,191\n"
        "Ⅳ. 영 업 이 익\n"
        "29\n"
        "31,993,162\n"
        "20,518,974\n"
    )

    sales_val, _ = dummy._extract_metric_value_from_context("매출액", context)
    op_val, _ = dummy._extract_metric_value_from_context("영업이익", context)

    assert sales_val == 199_744_705
    assert op_val == 31_993_162


def test_extract_metric_value_from_context_filters_unreasonable_huge_numbers():
    """비정상적으로 큰 숫자 토큰은 정량 근거로 채택하지 않는다."""
    from src.rag.qa_pipeline import RAGPipeline

    dummy = object.__new__(RAGPipeline)
    context = "매출액 29199744705166311191"
    sales_val, _ = dummy._extract_metric_value_from_context("매출액", context)
    assert sales_val is None


def test_match_query_metrics_maps_generic_balance_terms():
    """일반어(부채/자산/자본) 질의도 총계 메트릭으로 매핑된다."""
    from src.rag.qa_pipeline import RAGPipeline

    dummy = object.__new__(RAGPipeline)
    assert "부채총계" in dummy._match_query_metrics("2021년 부채가 얼마인가요?")
    assert "자산총계" in dummy._match_query_metrics("2021년 자산은 얼마인가요?")
    assert "자본총계" in dummy._match_query_metrics("2021년 자본은 얼마인가요?")


def test_route_query_type_narrative_section_not_forced_numeric_by_few_marker():
    """서술 섹션 질문은 '몇' 표현이 있어도 narrative를 우선한다."""
    from src.rag.qa_pipeline import RAGPipeline

    dummy = object.__new__(RAGPipeline)
    assert dummy.route_query_type("핵심감사사항이 몇 가지야?") == "narrative"


def test_ask_numeric_falls_back_to_narrative_when_numeric_facts_missing():
    """정량 경로에서 확정 수치가 없으면 서술형 경로로 안전하게 폴백한다."""
    from src.rag.qa_pipeline import RAGPipeline

    dummy = object.__new__(RAGPipeline)
    dummy._build_numeric_answer = lambda *args, **kwargs: (
        "정형 데이터와 재무제표 문맥에서 질문에 대응되는 수치를 확정하지 못했습니다.\n\n참고 출처: 없음"
    )
    dummy._ask_narrative = lambda query, **kwargs: f"fallback:{query}"
    dummy._finalize_text_response = lambda response, stream: response

    result = dummy._ask_numeric(
        "부채가 얼마야?",
        stream=False,
        contexts=[],
        tool_rows=[],
        year_filter=None,
        is_multi=False,
        max_new_tokens=512,
    )
    assert result == "fallback:부채가 얼마야?"


def test_get_contexts_for_change_presence_query_expands_to_full_year_coverage():
    """연도 미지정 변경 여부 질의는 2014~2024 전체 커버리지를 강제한다."""
    from src.rag.qa_pipeline import RAGPipeline

    dummy = object.__new__(RAGPipeline)
    dummy.retrieve_k = 4
    dummy._auto_detect_years = lambda q: []
    dummy._is_multi_year_query = lambda q: True
    captured = {}

    def _fake_multi(query, total_k, target_years):
        captured["total_k"] = total_k
        captured["target_years"] = target_years
        return [{"year": y, "section": "감사의견", "text": "x"} for y in target_years[:total_k]]

    dummy._retrieve_multi_year = _fake_multi
    rows = dummy.get_contexts("감사의견이 변경된 연도가 있나요?", retrieve_k=4)

    assert rows
    assert captured["target_years"] == list(range(2014, 2025))
    assert captured["total_k"] == 11


def test_classify_tool_rows_splits_primary_and_auxiliary():
    """질문 메트릭과 일치하는 도구 행만 primary로 분류한다."""
    from src.rag.qa_pipeline import RAGPipeline

    dummy = object.__new__(RAGPipeline)
    rows = [
        {"year": 2021, "table_type": "balance_sheet", "item": "3. 매출채권", "values": ["33,088,247"]},
        {"year": 2021, "table_type": "cash_flow", "item": "Ⅰ. 영업활동 현금흐름", "values": ["22,796,257"]},
    ]
    classified = dummy.classify_tool_rows("2019년 영업활동 현금흐름은 어떻게 달라졌나요?", rows)

    assert classified["metrics"] == ["영업활동 현금흐름"]
    assert len(classified["primary_rows"]) == 1
    assert classified["primary_rows"][0]["item"] == "Ⅰ. 영업활동 현금흐름"
    assert len(classified["aux_rows"]) == 1
    assert classified["aux_rows"][0]["item"] == "3. 매출채권"


def test_extract_evidence_spans_prioritizes_target_section():
    """질문 타겟 섹션(예: 현금흐름) 근거가 동일 점수일 때 우선 노출된다."""
    from src.rag.qa_pipeline import RAGPipeline

    dummy = object.__new__(RAGPipeline)
    query = "2018년과 2019년의 현금흐름은 어떻게 달라졌나요?"
    contexts = [
        {"year": 2019, "section": "감사의견", "text": "2019년 현금흐름을 포함한 재무제표를 감사했습니다."},
        {"year": 2019, "section": "현금흐름", "text": "2019년 영업활동 현금흐름은 22,796,257입니다."},
    ]
    spans = dummy.extract_evidence_spans(query, contexts, max_spans=1)

    assert spans
    assert spans[0]["section"] == "현금흐름"


def test_build_coverage_guard_note_blocks_definitive_claim_on_missing_years():
    """변경 여부 질의에서 연도 누락 시 단정 금지 가드 문구를 생성한다."""
    from src.rag.qa_pipeline import RAGPipeline

    dummy = object.__new__(RAGPipeline)
    contexts = [{"year": 2014, "section": "감사의견", "text": "적정의견"}]
    note = dummy._build_coverage_guard_note("감사의견이 변경된 연도가 있나요?", contexts)

    assert "커버리지 경고" in note
    assert "단정" in note


def test_match_query_metrics_maps_generic_cashflow_term():
    """일반 '현금흐름' 질의도 주요 현금흐름 메트릭으로 매핑된다."""
    from src.rag.qa_pipeline import RAGPipeline

    dummy = object.__new__(RAGPipeline)
    metrics = dummy._match_query_metrics("2019년 현금흐름은 어떻게 달라졌나요?")
    assert "영업활동 현금흐름" in metrics
    assert "투자활동 현금흐름" in metrics
    assert "재무활동 현금흐름" in metrics


def test_build_numeric_answer_adds_cashflow_interpretation_for_negative_series():
    """현금흐름(음수) 2개년 비교 시 유출 규모 확대/축소 해석을 명시한다."""
    from src.rag.qa_pipeline import RAGPipeline

    dummy = object.__new__(RAGPipeline)
    query = "2018년과 2019년 재무활동 현금흐름은 어떻게 달라졌나요?"
    contexts = []
    tool_rows = [
        {
            "year": 2018,
            "table_type": "cash_flow",
            "item": "Ⅲ. 재무활동 현금흐름",
            "values": ["(12,818,480)", "(11,801,987)"],
            "score": 5.0,
        },
        {
            "year": 2019,
            "table_type": "cash_flow",
            "item": "Ⅲ. 재무활동 현금흐름",
            "values": ["(9,787,719)", "(12,818,480)"],
            "score": 5.0,
        },
    ]

    answer = dummy._build_numeric_answer(query, contexts, tool_rows)
    assert "재무활동 현금흐름 해석(2018→2019): 순유출 규모 축소" in answer


def test_filter_contexts_by_topic_terms_removes_irrelevant_year_for_covid():
    """코로나 질의에서는 코로나 언급이 없는 문서를 후처리로 제거한다."""
    from src.rag.qa_pipeline import RAGPipeline

    dummy = object.__new__(RAGPipeline)
    contexts = [
        {"year": 2014, "section": "주석", "text": "한국채택국제회계기준 일반 설명"},
        {"year": 2020, "section": "주석", "text": "코로나19(COVID-19) 관련 임차료 할인"},
        {"year": 2021, "section": "주석", "text": "코로나19 관련 실무적 간편법 적용"},
    ]
    filtered = dummy._filter_contexts_by_topic_terms(
        "코로나19(COVID-19)로 인한 불확실성이 언급된 연도와 내용을 알려줘",
        contexts,
    )

    assert [c["year"] for c in filtered] == [2020, 2021]


def test_extract_evidence_spans_change_query_prefers_year_diversity():
    """변경 여부 질의에서는 근거 스팬이 동일 연도에 편중되지 않아야 한다."""
    from src.rag.qa_pipeline import RAGPipeline

    dummy = object.__new__(RAGPipeline)
    query = "감사의견이 변경된 연도가 있나요?"
    contexts = [
        {"year": 2014, "section": "감사의견", "text": "재무제표를 공정하게 표시하고 있습니다."},
        {"year": 2015, "section": "감사의견", "text": "재무제표를 공정하게 표시하고 있습니다."},
        {"year": 2016, "section": "감사의견", "text": "재무제표를 공정하게 표시하고 있습니다."},
        {"year": 2017, "section": "감사의견", "text": "재무제표를 공정하게 표시하고 있습니다."},
    ]
    spans = dummy.extract_evidence_spans(query, contexts, max_spans=3)

    assert len(spans) == 3
    assert len({s["year"] for s in spans}) == 3
