# 삼성전자 감사보고서 NLP 분석 시스템

4조 김범준 김진식 문이현 이다현 전지현

삼성전자의 2014–2024년 감사보고서(11개 연도)를 기반으로 구축한 **금융 도메인 특화 RAG QA 시스템**입니다.
HTML 파싱 → 데이터 스키마 → 벡터 인덱스 → Streamlit 챗봇 인터페이스까지 E2E 파이프라인을 구현했습니다.

---

## 시스템 아키텍처

```
[HTML 감사보고서 11개 연도 (2014–2024)]
        │
        ▼  HTML 파싱 & 텍스트 정제  (src/parser/html_parser.py)
        │
        ▼  데이터 스키마 저장  (CSV / SQLite)  (src/schema/data_store.py)
        │
        ▼  청킹 & 임베딩  (upskyy/bge-m3-korean)  (src/rag/embedder.py)
        │
        ▼  FAISS 벡터 인덱스  (src/rag/vector_store.py)
        │
        ▼  하이브리드 검색 (시맨틱 + 키워드) + LLM 답변 생성  (src/rag/qa_pipeline.py)
        │
        ▼  Streamlit 멀티턴 QA 인터페이스  (app.py)
```

---

## 요구 환경

| 항목 | 버전 |
|:---|:---|
| Python | 3.11 이상 |
| 운영체제 | macOS (Apple Silicon MPS 지원) / Linux / Windows |
| 디스크 여유 공간 | 최소 8 GB 권장 (로컬 모델/캐시 사용 시 10 GB 이상 필요 가능) |

---

## 설치 방법

```bash
# 1. 레포지토리 클론
git clone https://github.com/koreaben777/samsung-audit-nlp.git
cd samsung-audit-nlp

# 2. 가상환경 생성 및 활성화
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. 의존성 설치
pip install -r requirements.txt
```

---

## 원본 데이터부터 재현하는 방법 (E2E)

아래 절차는 `data/raw`의 HTML만으로 `processed` 산출물(`sections.csv`, `financial_data.csv`, `faiss.index`)을 다시 만드는 표준 재현 절차입니다.

```bash
source venv/bin/activate

# 1) HTML 파싱 -> SQLite/CSV 저장 + 검증
python - <<'PY'
from src.parser.html_parser import parse_all_reports
from src.schema.data_store import save_all_reports, init_db, export_csv, validate

db_path = "data/processed/audit.db"
reports = parse_all_reports("data/raw")
save_all_reports(reports, db_path)

conn = init_db(db_path)
export_csv(conn, "data/processed")
print("validate:", validate(conn))
conn.close()
PY

# 2) 벡터 인덱스 재생성
python - <<'PY'
from src.rag.qa_pipeline import RAGPipeline
RAGPipeline(csv_path="data/processed/sections.csv", rebuild=True)
print("FAISS rebuild complete")
PY

# 3) 회귀 테스트
pytest tests/ -q

# 4) 핵심 회귀(질의 품질) 테스트
pytest -q tests/test_rag.py
```

예상 결과:
- `data/processed/sections.csv`
- `data/processed/financial_data.csv`
- `data/processed/faiss.index`
- `data/processed/chunks_meta.pkl`

재현성 강화 권장:
```bash
# 현재 환경 스냅샷(발표/제출 직전 권장)
pip freeze > requirements-lock.txt

# 동일 환경 재설치
pip install -r requirements-lock.txt
```

---

## 실행 방법

### 권장 실행 프로필

로컬(Qwen)과 API(OpenAI)를 혼동 없이 분리하기 위해 실행 스크립트를 제공합니다.

```bash
# 로컬 Qwen 테스트 (API 키 무시)
./scripts/run_local_qwen.sh

# OpenAI API 테스트 (키를 입력하거나 OPENAI_API_KEY 사전 설정)
./scripts/run_local_openai.sh
```

> `run_local_qwen.sh`는 `OPENAI_API_KEY/ANTHROPIC_API_KEY`를 자동 해제하고 `LLM_BACKEND=local`을 강제합니다.

앱 실행 후에는 사이드바의 **LLM 설정**에서 `OpenAI API`와 `로컬` 백엔드를 전환할 수 있습니다.
OpenAI API를 선택하려면 `OPENAI_API_KEY`가 환경변수 또는 Streamlit Secrets에 설정되어 있어야 합니다.
로컬 백엔드는 첫 답변 생성 시점에 모델을 로드하므로 최초 응답이 느릴 수 있습니다.

### 수동 실행 (필요 시)

```bash
source venv/bin/activate
export LLM_BACKEND=local
streamlit run app.py
```

또는

```bash
source venv/bin/activate
export OPENAI_API_KEY="sk-proj-..."
export LLM_BACKEND=openai
streamlit run app.py
```

### 웹 배포 실행 원칙

Streamlit Community Cloud에서는 API 모드만 사용을 권장합니다.

- 로컬 개발/검증: `local (Qwen)`
- 배포 환경: `openai` (Secrets 사용)
- 게시 앱에서도 사이드바 토글은 표시되지만, 로컬 모델은 클라우드 메모리/디스크 제약으로 실패할 수 있습니다.

### 환경변수 요약

| 환경변수 | 기본값 | 설명 |
|:---|:---|:---|
| `LLM_BACKEND` | `openai`(키 있을 때) / `local`(없을 때) | 백엔드 강제 지정 (`local` / `openai` / `anthropic`) |
| `OPENAI_API_KEY` | 없음 | OpenAI API 키 |
| `OPENAI_MODEL` | `gpt-5.4-mini` | OpenAI 백엔드에서 사용할 모델 ID |
| `ANTHROPIC_API_KEY` | 없음 | Anthropic API 키 |
| `LOCAL_MODEL` | `Qwen/Qwen2.5-3B-Instruct` | 로컬 모델 HuggingFace ID |
| `RETRIEVE_K` | `4` | RAG 검색 문서 수 |
| `MAX_NEW_TOKENS` | `1024` | 로컬 모델 최대 생성 토큰 |

샘플 파일: `.env.example`

---

## 프로젝트 구조

```
samsung-audit-nlp/
├── app.py                        # Streamlit 웹 인터페이스
├── requirements.txt              # Python 의존성
├── .env.example                  # 실행 환경변수 예시
├── scripts/
│   ├── run_local_qwen.sh         # 로컬 Qwen 강제 실행
│   └── run_local_openai.sh       # OpenAI API 실행
├── data/
│   ├── raw/                      # 원본 HTML 감사보고서 (2014–2024)
│   └── processed/
│       ├── sections.csv          # 파싱된 섹션 텍스트
│       ├── financial_data.csv    # 추출된 재무 수치
│       ├── faiss.index           # 사전 빌드된 FAISS 벡터 인덱스
│       └── chunks_meta.pkl       # 청크 메타데이터
├── src/
│   ├── parser/
│   │   └── html_parser.py        # HTML 파싱 모듈
│   ├── rag/
│   │   ├── embedder.py           # 청킹 & 임베딩 (bge-m3-korean)
│   │   ├── vector_store.py       # FAISS 벡터 스토어
│   │   └── qa_pipeline.py        # RAG + LLM 파이프라인
│   └── schema/
│       └── data_store.py         # 데이터 스키마 (CSV / SQLite)
└── tests/
    ├── conftest.py
    ├── test_parser.py
    ├── test_rag.py
    └── test_schema.py
```

---

## 테스트 실행

```bash
source venv/bin/activate
pytest tests/ -v
```

커버리지 포함:

```bash
pytest tests/ --cov=src --cov-report=term-missing
```

---

## 개발/CI 운영 규칙

- 브랜치 전략: `main`(배포), `develop`(통합 개발)
- 커밋 컨벤션: `feat:`, `fix:`, `chore:`, `docs:`, `test:`
- CI: `.github/workflows/ci.yml`에서 `pytest` 자동 실행

---

## 웹 배포 (Streamlit Community Cloud / Hugging Face Spaces)

클라우드 환경에서는 로컬 모델 대신 API 백엔드를 사용하도록 Secrets를 설정합니다.

**필수 Secrets:**

```
OPENAI_API_KEY = sk-proj-...
LLM_BACKEND = openai
```

또는

```
ANTHROPIC_API_KEY = sk-ant-...
LLM_BACKEND = anthropic
```

**Streamlit Community Cloud:** [share.streamlit.io](https://share.streamlit.io) → New App → GitHub 레포 연결 → Branch: `main` → Main file: `app.py`

**Hugging Face Spaces:** New Space → SDK: Streamlit → GitHub 레포 연동 → Secrets 설정

---

## API 비용 및 보안 주의

- OpenAI/Anthropic API는 사용량 기반 과금이므로 테스트 시 요청량을 제한하세요.
- API 키는 코드에 하드코딩하지 말고 환경변수 또는 플랫폼 Secrets로만 관리하세요.
- `.env` 파일은 Git에 커밋하지 않고, `.env.example`만 공유하세요.

운영 시 비용/제한 체크리스트:
- OpenAI 기본 모델은 `OPENAI_MODEL`로 제어합니다.
- 질의당 검색 문서 수(`k`)를 필요 이상으로 올리지 않습니다. (권장: 단건 2 이상 4 이하, 트렌드 4 이상 6 이하)
- 배포 전 OpenAI/Anthropic 대시보드에서 프로젝트별 월 예산/알림을 설정합니다.
- 장시간 데모 전에는 API rate-limit 정책(분당 요청/토큰)을 확인하고, 동일 질문 반복 호출을 피합니다.

참고: 현재 OpenAI 호출부는 `chat.completions`를 사용하며 모델별 기본 토큰 제한 정책을 따릅니다.

---

## 주요 기능

- **멀티턴 QA**: 2014–2024년 전 기간 감사보고서 질의 응답
- **하이브리드 검색**: 시맨틱 검색(FAISS) + 키워드 매칭 결합
- **도구 호출(Tool Calling) 레이어**: 재무 수치 질의 시 `financial_data.csv` 정형 데이터를 우선 조회해 답변 근거 강화
- **도구 결과 정합성 분류**: 질문 메트릭과 일치하는 도구 행은 `primary`, 비일치 행은 `보조참고(aux)`로 분리 표시
- **멀티연도 트렌드 분석**: "추이", "비교", "변화" 키워드 감지 시 연도별 균등 검색 자동 전환
- **변경 여부 질의 커버리지 가드**: 연도 미지정 `변경/없다` 류 질문은 2014–2024 전체 확인 후에만 단정 답변 허용
- **섹션 보호**: 재무제표·핵심감사사항 등 소규모 섹션 우선 확보로 환각 방지
- **근거 스팬 재랭킹**: 질문 타겟 섹션(예: 현금흐름)을 우선하고, 변경 여부 질의는 연도 다양성을 우선 확보
- **숫자 추출 안전장치**: 문맥 숫자 결합 오검출 방지, 비정상 대형 수치 필터링, 현금흐름 부호 기반 해석(유출/유입 확대·축소) 보강
- **토픽 후처리 필터**: `코로나19/COVID-19`처럼 명시 토픽 질의에서 비관련 컨텍스트를 후처리로 제외
- **UI 마크다운 안전 처리**: `~` 문자를 이스케이프해 범위 표기(`2014~2024`)가 취소선으로 깨지지 않도록 처리
- **대화 내보내기**: Markdown / JSON 형식으로 저장
- **인덱스 재빌드**: 사이드바 버튼으로 FAISS 인덱스 즉시 갱신

---

## 최신 질의 품질 검증 포인트

아래 항목은 최근 패치에서 강화된 동작이며, 회귀 테스트(`tests/test_rag.py`)로 검증됩니다.

- 정량 추출에서 숫자 결합(문장 내 인접 숫자 합쳐짐) 방지
- 비정상 대형 수치(현실적으로 불가능한 자리수) 자동 제외
- 연도 미지정 변경 여부 질문의 전체 연도(2014–2024) 커버리지 강제
- 질문 메트릭 기반 도구 행 `primary/aux` 분류
- 근거 스팬의 타겟 섹션 우선 + 변경 여부 질의의 연도 다양성 확보
- 현금흐름(음수) 2개년 비교 시 "유출 규모 확대/축소" 해석 일관성 유지
- `코로나19` 질의에서 비관련 연도 컨텍스트 제거

---

## 데이터 가용 범위

| 섹션 | 가용 연도 | 비고 |
|:---|:---|:---|
| 감사의견 | 2014–2024 | 전 연도 적정의견 |
| 감사의견근거 | 2014–2024 | |
| 핵심감사사항 | 2018–2024 | 2014–2017 해당 없음 |
| 재무상태표 | 2014–2024 | |
| 포괄손익 | 2014–2024 | |
| 현금흐름 | 2014–2024 | |
| 주석 | 2014–2024 | |

---

## 검색 품질 평가셋 (Retrieval Eval)

질문셋 기반으로 retrieval 품질을 정량 점검할 수 있습니다.

```bash
source venv/bin/activate
python scripts/evaluate_retrieval.py --dataset eval/qa_eval_set.jsonl --k 6
```

평가 지표:
- `Year hit`: 기대 연도 매칭 성공률
- `Section hit`: 기대 섹션 매칭 성공률
- `Keyword hit`: 기대 키워드가 컨텍스트에 포함되는 비율

---

## 참고

- 임베딩 모델: [upskyy/bge-m3-korean](https://huggingface.co/upskyy/bge-m3-korean)
- 로컬 LLM: [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
- 과정: 서울대학교 빅데이터 핀테크 전문가 과정 12기
- 전체 출처 목록: [docs/SOURCES.md](docs/SOURCES.md)
