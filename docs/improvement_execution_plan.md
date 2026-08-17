# 개선안 1~4 + Streamlit 접근성 개선 실행계획 (최종본)

작성일: 2026-04-03

## 0. 목표

- 개선안 1~4를 코드/문서로 반영해 과제 평가 적합성과 시연 신뢰도를 높인다.
- 외부 사용자 무로그인 접속 이슈를 복구 가능한 운영 절차로 표준화한다.

## 1. Planner v1 (초안)

### 개선안 1. Tool Calling 레이어
- `financial_data.csv`를 도구 데이터 소스로 사용
- 재무 키워드 질의 감지 시 정형 행 조회 결과를 LLM 프롬프트에 주입
- UI에서 도구 조회 결과를 별도 영역으로 노출

### 개선안 2. 평가셋 구축
- `eval/qa_eval_set.jsonl` 생성 (단일/멀티연도 혼합)
- `scripts/evaluate_retrieval.py`로 Year/Section/Keyword hit 측정

### 개선안 3. k 슬라이더 실반영
- `app.py`의 슬라이더 값을 `pipeline.get_contexts(..., retrieve_k=...)`로 전달
- `ask(..., retrieve_k=...)`까지 전달해 경로 일관성 유지

### 개선안 4. 근거 스팬 인용
- 컨텍스트 문장 단위 점수화(질문 키워드 + 숫자 포함) 후 상위 근거 3개 추출
- UI와 내보내기(MD/JSON)에 근거 스팬 표시

### Streamlit 접근 이슈
- 앱 공유 설정/배포 상태/워크스페이스 권한/지원 요청까지 운영 런북 문서화

## 2. Reviewer 1차 검토

발견 결점:
1. Tool Calling이 API 함수호출이 아니어도 “도구 실행 흐름”이 명확히 드러나야 함
2. 평가셋은 실행 커맨드/지표 정의가 README에 반드시 포함되어야 함
3. 접근 이슈는 코드 변경만으로 해결 불가하므로 운영 조치 책임과 순서를 명시해야 함
4. UI에 근거 스팬만 있고 원문 출처(연도/섹션) 연결이 약하면 발표 설득력이 떨어짐

## 3. Planner v2 (수정안)

수정 반영:
- Tool Calling 결과를 `tool_rows`로 구조화해 프롬프트 주입 + UI/내보내기 동시 노출
- README에 Retrieval Eval 실행법/지표를 명시
- Streamlit 접근 이슈는 별도 런북(`docs/streamlit_access_fix.md`)으로 분리
- 근거 스팬 출력에 연도/섹션 메타를 강제 포함

## 4. Reviewer 2차 검토 결과

판정: 실행 가능(Approved)

승인 근거:
- 1~4번 개선안이 코드와 운영 문서 모두에 반영됨
- 과제 평가 관점(재현성/신뢰성/시연 완성도)에서 직접적인 개선 효과가 있음

## 5. 제작자 핸드오프(실행 순서)

1. `main` 기준 코드 Pull
2. `pytest tests/ -q`로 회귀 확인
3. `python scripts/evaluate_retrieval.py --dataset eval/qa_eval_set.jsonl --k 6` 실행
4. Streamlit Cloud에서 공유 설정을 Public으로 변경
5. 시크릿 창/타 기기에서 앱 URL 접속 검증
6. 발표 전날 데모 질의 5개로 리허설

## 6. 완료 기준 (Definition of Done)

- 테스트 전부 통과
- Retrieval Eval 지표 산출 완료
- 앱에서 도구 조회/근거 스팬이 실제 노출
- 외부 무로그인 접속 성공 확인(운영 검증)

## 7. 최신 문서 반영 이력 (2026-04-06)

- README에 최신 질의 품질 개선 사항을 명시:
  - 변경 여부 질의의 전체 연도(2014~2024) 커버리지 가드
  - 도구 조회 결과 `primary/aux` 분리
  - 근거 스팬 재랭킹(타겟 섹션 우선, 변경 질의 연도 다양성)
  - 숫자 결합 오검출/비정상 대형 수치 필터
  - 토픽(예: 코로나19) 후처리 필터
  - UI `~` 마크다운 안전 처리
- 재현 절차에 `tests/test_rag.py` 핵심 회귀 테스트 실행 단계를 추가.
