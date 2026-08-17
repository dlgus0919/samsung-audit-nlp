# SOURCES (자료/코드 출처)

작성일: 2026-04-06

본 문서는 프로젝트 구현/운영 과정에서 참고한 주요 자료와 문서를 기록합니다.

## 1. 과제 원문 및 데이터

- 과제 지침 노트북: `핀테크_12기_최종과제.ipynb`
- 원본 데이터: `data/raw/감사보고서_2014.htm` ~ `data/raw/감사보고서_2024.htm`
- 과정 정보: 서울대학교 빅데이터 핀테크 전문가 과정 12기

## 2. 모델/라이브러리 공식 문서

- Sentence Transformers: <https://www.sbert.net/>
- Hugging Face Transformers: <https://huggingface.co/docs/transformers/index>
- FAISS: <https://github.com/facebookresearch/faiss>
- Streamlit: <https://docs.streamlit.io/>
- PyTorch: <https://pytorch.org/docs/stable/index.html>
- Pandas: <https://pandas.pydata.org/docs/>

## 3. 사용 모델

- 임베딩 모델: `upskyy/bge-m3-korean`  
  <https://huggingface.co/upskyy/bge-m3-korean>
- 로컬 LLM: `Qwen/Qwen2.5-3B-Instruct`  
  <https://huggingface.co/Qwen/Qwen2.5-3B-Instruct>

## 4. 외부 API

- OpenAI API 문서: <https://platform.openai.com/docs>
- Anthropic API 문서: <https://docs.anthropic.com/>

## 5. 저장소 내 설계/운영 참고 문서

- 개선 실행 계획: `docs/improvement_execution_plan.md`
- Streamlit 접근성 이슈 런북: `docs/streamlit_access_fix.md`
- 파서 구현/결과 요약:
  - `src/parser/walkthrough.md`
  - `src/parser/parsing_results_report.md`
- 스키마 구현 요약: `src/schema/walkthrough.md`

## 6. 출처 관리 원칙

- 코드/문서에 외부 의존(모델, SDK, API)이 추가될 때 본 문서에 즉시 반영
- 발표자료/최종 보고서의 참고문헌 섹션과 본 문서를 동기화
- 예시 코드/프롬프트를 외부에서 인용한 경우, 파일/라인 단위로 출처 메모 남기기
