# STEP 5: RAG QA 시스템 구현 계획

본 계획은 기정제된 `data/processed/sections.csv` 데이터를 활용하여 Streamlit 기반의 RAG (Retrieval-Augmented Generation) Q&A 파이프라인을 구축하는 것을 목표로 합니다.

## 검토 및 조치 사항 (사전 검증)
- `sections.csv`의 컬럼은 `['id', 'year', 'section', 'content', 'char_count', 'source_file', 'extracted_at']`로 확인되었습니다. `embedder.py`에서 `year`, `section`, `content` 컬럼을 사용해 Chunk를 분리하겠습니다.
- 환경 검증 결과 `sentence-transformers`, `faiss-cpu`, `streamlit`, `openai`, `transformers`는 설치되어 있으나, `anthropic`과 `accelerate` 패키지가 누락되어 있습니다. 이들은 구현 전에 `pip install anthropic accelerate`로 설치해야 교체 가능한 백엔드 구조가 동작합니다.

## User Review Required
> [!IMPORTANT]
> - `anthropic` 라이브러리와 로컬 LLM을 더 빠르고 안정적으로 구동하기 위한 `accelerate` 라이브러리 설치를 자동으로 진행해도 될까요?
> - `sections.csv` 내에서 `content`가 누락된 행(결측치)이나 지나치게 짧은 문구들은 Chunk 구성 단위에서 배제(스킵)하는 것이 좋을지, 로직 그대로 적용해도 될지 확인 부탁드립니다.

## 구체적 구현 파일 및 역할

### 1. `src/rag/embedder.py`
- **목적:** CSV 데이터를 정해진 길이(`CHUNK_SIZE=400`)로 분할(Sliding Window/단순 분할)하고, `snunlp/KR-SBERT-V40K-klueNLI-augSTS` 모델을 사용해 임베딩을 수행합니다.
- **주요 구성요소:**
  - `@dataclass Chunk`
  - `load_chunks_from_csv(csv_path)`: 결측치 건너뛰기, 단위 분할.
  - `get_embed_model()`: @lru_cache 또는 유사한 싱글톤 캐시 처리로 모델 재로드 방지.
  - `build_embeddings()`: `.encode(normalize_embeddings=True)`로 코사인 유사도 연산을 위한 L2 정규화 적용.

### 2. `src/rag/vector_store.py`
- **목적:** 분할된 임베딩 벡터를 저장 및 고속 유사도 검색(FAISS)에 활용합니다.
- **주요 구성요소:**
  - `VectorStore` 클래스
  - `build()`: `faiss.IndexFlatIP()` 사용, `.faiss` 및 `chunks_meta.pkl` 로컬 파일에 상태 저장.
  - `search()`: `query_vec`와 유사도가 높은 상위 `k`개 검색 (`index.search` 사용), 메타데이터 반환.

### 3. `src/rag/qa_pipeline.py`
- **목적:** Retrieval과 LLM 생성을 관장하는 백엔드 코어 파이프라인.
- **주요 구성요소:**
  - `RAGPipeline` 클래스: 생성 시 `rebuild` 인자를 통해 `VectorStore`의 강제 최신화를 수행. LLM 백엔드(`local`, `openai`, `anthropic`)를 환경 변수 기반으로 동적 라우팅.
  - `retrieve()`: 연도 필터링(`year_filter`)을 검색 결과(Post-retrieval filtering)에서 직접 수행하도록 로직 적용.
  - `ask()`: `Qwen/Qwen3-0.6B`의 경우 `transformers.pipeline` 또는 `AutoModelForCausalLM`의 `streamer`를 활용. (맥 환경: `device_map="auto"` 또는 `.to('mps')` 분기 처리). 스트리밍 반환.

### 4. `app.py` (Streamlit 인터페이스)
- **목적:** 사용자가 브라우저에서 사용할 수 있는 챗봇 UI.
- **주요 구성요소:**
  - `@st.cache_resource`로 메모리에 `RAGPipeline` 영구 상주.
  - 세션 기반의 히스토리 관리(`st.session_state["messages"]`).
  - 사용자 질의가 들어오면, `pipeline.ask(stream=True)`의 Generator를 `st.write_stream()`으로 화면에 실시간 출력.
  - `st.expander`를 통해 어떤 문서를 참고했는지 출처 스니펫 노출.

---

## Open Questions
> [!TIP]
> - `Qwen3-0.6B`와 같은 로컬 모델의 경우, Mac 환경(Apple Silicon)에 맞춰 `device='mps'`를 우선 시도하도록 코드를 구성하겠습니다. 이 방향이 맞을까요?
> - `rebuild=True` 버튼을 누르면 기존 저장된 FAISS/pkl을 지우고 새로 구성됩니다. 이 로직을 사이드바 구성 시 함께 포함하겠습니다.

## Verification Plan

### 수동 검증:
1. `pip install anthropic accelerate` 실행
2. Python 스크립트로 `RAGPipeline(rebuild=True)` 작동 및 `ask('2023년 핵심감사사항은 무엇인가요?')` 의 응답 확인.
3. `streamlit run app.py`를 터미널 백그라운드로 실행하고 화면상의 UI 렌더링 검토.
