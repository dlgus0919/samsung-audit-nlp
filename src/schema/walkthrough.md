# DB 저장 (Data Store) 모듈 구현 요약

요청하신 문서의 저장 전략과 DB 스키마 명세를 바탕으로 `src/schema/data_store.py` 구현을 모두 완료했습니다. 주요 내용과 구동 방식은 아래와 같습니다.

## 구현 위치
* `src/schema/__init__.py`
* [src/schema/data_store.py](file:///Users/june_kim/samsung_audit_nlp/src/schema/data_store.py)

## 주요 구현 사항
1. **SQLite 3개 테이블 스키마 구성 (`init_db`)**
   * **`sections`**: 추출된 문장/텍스트 섹션을 이력 저장. 본문의 `char_count` 정보도 함께 적재 (쿼리 효율 향상 목적성)
   * **`financial_data`**: 추출된 재무제표 보조 데이터 저장 (`table_type`로 구분. 연도별 항목 값을 유동적으로 가져오기 위해 JSON Array 파서를 사용해 `value_raw`를 보존)
   * **`parse_log`**: 데이터 추출 및 이관 시 오류 내역(`error_msg`)과 상태(`status`) 추적 테이블

2. **데이터 처리 및 덮어쓰기 로직 (`save_report`, `save_all_reports`)**
   * 파서로 읽어들인 Dict 형태를 순회하며 DB에 저장합니다.
   * 저장 시 `DELETE FROM ... WHERE year=?` 로직을 선행하여 **동일 연도 중복 삽입 시 무결성을 확보할 수 있도록 덮어쓰기 로직**을 구성했습니다.

3. **CSV 이관 (`export_csv`)**
   * `data/processed/sections.csv` 및 `financial_data.csv` 로 Pandas를 활용한 일괄 덤프 코드를 이식했습니다.

4. **검증 로직 (`validate`)**
   * 누락된 섹션을 추적하되, 의도된 결측인 `핵심감사사항` (2020년 미만)은 예외 통과하도록 룰을 구성했습니다. (참고: 2014-2017, 2019 등의 구형 보고서에서는 `감사의견근거` 또한 원래 문서에 누락되어 있어 False/Missing이 발생할 수 있습니다.)

5. **조회 편의 (`load_sections`)**
   * 튜플 파라미터 기반 `read_sql_query` 호출로 조건부(연도, 섹션명 조회 지원) 데이터 로딩 지원 

## 결과 확인
터미널에서 `data/processed/audit.db` 파일이 생성됨과 동시에 CSV 이관, 테이블 및 데이터 정상 구조화 검증 과정을 모두 통과했습니다. (Validataion Output 결과 `Total 11 reports parsed.`와 누락 이력 검출 정상 작동 확인)
