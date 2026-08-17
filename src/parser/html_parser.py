"""
src/parser/html_parser.py
=========================
삼성전자 감사보고서 .htm 파일 파서.

파일 특성:
- 인코딩  : euc-kr (fallback: cp949)
- 레이아웃: 테이블 기반 (h1~h4 안에 텍스트 없음)
- 섹션 구분: 키워드 매칭 (태그 구조가 아닌 내용 기반)
- 핵심감사사항: 2020년부터 추가
"""

from __future__ import annotations

import logging
import os
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from bs4 import BeautifulSoup, Tag

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 섹션 경계 키워드 정의
# ---------------------------------------------------------------------------

# 감사보고서 본문 시작 앵커 (h2 텍스트)
_REPORT_ANCHOR = "독립된 감사인의 감사보고서"

# 섹션 순서 및 탐지 키워드 (순서 중요 — 앞 섹션 끝이 다음 섹션 시작)
_SECTION_KEYWORDS: dict[str, list[str]] = {
    "감사의견":     ["감사의견"],
    "감사의견근거": ["감사의견근거", "감사인의 책임"],
    "핵심감사사항": ["핵심감사사항"],
    # 재무제표 섹션은 h2 "(첨부)재 무 제 표" 이하에 위치
    "재무상태표":   ["재 무 상 태 표", "재무상태표"],
    "포괄손익":     ["손 익 계 산 서", "손익계산서", "포 괄 손 익 계 산 서", "포괄손익계산서"],
    "현금흐름":     ["현 금 흐 름 표", "현금흐름표"],
    "주석":         ["주   석", "주 석", "주석"],
}

# 재무제표 테이블 분류 키워드
_TABLE_KEYWORDS: dict[str, list[str]] = {
    "balance_sheet":     ["자산", "부채", "자본금", "이익잉여금"],
    "income_statement":  ["매출", "영업이익", "당기순이익", "법인세"],
    "cash_flow":         ["현금", "영업활동", "투자활동", "재무활동"],
}


# ---------------------------------------------------------------------------
# 공개 함수
# ---------------------------------------------------------------------------

def load_html(filepath: str) -> BeautifulSoup:
    """
    euc-kr 인코딩으로 .htm 파일을 읽고 BeautifulSoup 객체 반환.
    인코딩 실패 시 cp949로 fallback.
    """
    filepath = str(filepath)
    for enc in ("euc-kr", "cp949"):
        try:
            with open(filepath, encoding=enc, errors="strict") as fh:
                raw = fh.read()
            return BeautifulSoup(raw, "lxml")
        except (UnicodeDecodeError, LookupError):
            logger.warning("load_html: %s 인코딩 실패, fallback 시도 (%s)", enc, filepath)
    # 마지막 수단: 오류 무시
    with open(filepath, encoding="euc-kr", errors="replace") as fh:
        raw = fh.read()
    logger.warning("load_html: 오류 문자 대체하여 로드 완료 (%s)", filepath)
    return BeautifulSoup(raw, "lxml")


def extract_year(filepath: str) -> int:
    """
    파일명(감사보고서_2014.htm)에서 연도 추출.

    Raises
    ------
    ValueError
        파일명에 4자리 연도가 없을 경우.
    """
    name = Path(filepath).stem          # e.g. '감사보고서_2014'
    m = re.search(r"(\d{4})", name)
    if not m:
        raise ValueError(f"파일명에서 연도를 찾을 수 없습니다: {filepath}")
    return int(m.group(1))


def extract_sections(soup: BeautifulSoup) -> dict[str, str]:
    """
    섹션별 순수 텍스트를 추출하여 딕셔너리로 반환.

    반환 키:
        '감사의견', '감사의견근거', '핵심감사사항',
        '재무상태표', '포괄손익', '현금흐름', '주석'

    Notes
    -----
    - 감사보고서 본문은 h2 '독립된 감사인의 감사보고서' 이후.
    - 재무제표 본문은 h2 '(첨부)재 무 제 표' 이후.
    - 핵심감사사항이 없으면 빈 문자열 반환.
    - 섹션 경계는 <p> 또는 <span> 내 키워드 매칭으로 결정.
    """
    # 텍스트 노드 시퀀스를 하나의 리스트로 평탄화
    # (tag, text) 쌍 — tag.name in {p, span, td, th, h2, h3}
    nodes = _collect_text_nodes(soup)

    sections: dict[str, str] = {k: "" for k in _SECTION_KEYWORDS}

    # ── 감사보고서 섹션 (h2 '독립된 감사인의 감사보고서' ~ h2 '(첨부)재 무 제 표') ──
    audit_nodes  = _slice_nodes_between(nodes, _REPORT_ANCHOR, "(첨부)")
    # ── 재무제표 섹션 (h2 '(첨부)재 무 제 표' ~ 문서 끝) ──
    fs_nodes     = _slice_nodes_from(nodes, "(첨부)")

    # 감사보고서 본문: 감사의견 / 감사의견근거 / 핵심감사사항 추출
    _extract_audit_sections(audit_nodes, sections)

    # 재무제표: 재무상태표 / 포괄손익 / 현금흐름 / 주석 추출
    _extract_fs_sections(fs_nodes, sections)

    return sections


def extract_tables(soup: BeautifulSoup) -> list[pd.DataFrame]:
    """
    soup 내 모든 <table> 태그를 pandas DataFrame 리스트로 변환.

    - 빈 테이블(데이터 행 < 2) 제외
    - 첫 행을 columns로 설정 (중복 컬럼은 자동 suffix 처리)
    - 셀 내 공백/개행 정제
    """
    tables: list[pd.DataFrame] = []
    for tbl in soup.find_all("table"):
        rows = tbl.find_all("tr")
        if len(rows) < 2:
            continue

        matrix: list[list[str]] = []
        for tr in rows:
            cells = tr.find_all(["td", "th"])
            row = [_clean_cell(c) for c in cells]
            if any(row):                # 완전히 빈 행 제외
                matrix.append(row)

        if len(matrix) < 2:
            continue

        # 컬럼 수 통일 (가장 긴 행 기준, 짧은 행은 빈 문자열로 패딩)
        max_cols = max(len(r) for r in matrix)
        matrix = [r + [""] * (max_cols - len(r)) for r in matrix]

        header = matrix[0]
        # 중복 컬럼명 처리
        header = _deduplicate_columns(header)

        df = pd.DataFrame(matrix[1:], columns=header)
        tables.append(df)

    return tables


def extract_financial_tables(soup: BeautifulSoup) -> dict[str, pd.DataFrame]:
    """
    extract_tables() 결과 중 재무제표 핵심 테이블만 분류하여 반환.

    반환 키: 'balance_sheet', 'income_statement', 'cash_flow'
    키워드 매칭 점수가 가장 높은 테이블 1개씩 선택.
    """
    all_tables = extract_tables(soup)
    result: dict[str, pd.DataFrame] = {}

    for category, keywords in _TABLE_KEYWORDS.items():
        best_df: Optional[pd.DataFrame] = None
        best_score = 0
        for df in all_tables:
            score = _table_keyword_score(df, keywords)
            if score > best_score:
                best_score = score
                best_df = df
        if best_df is not None and best_score > 0:
            result[category] = best_df

    return result


def parse_single_report(filepath: str) -> dict:
    """
    단일 .htm 파일 전체 파싱 통합 함수.

    반환 구조::

        {
          'year':             int,
          'source_file':      str,
          'extracted_at':     str,   # ISO 8601 타임스탬프
          'sections':         dict,  # extract_sections() 결과
          'financial_tables': dict,  # extract_financial_tables() 결과
        }
    """
    filepath = str(filepath)
    soup = load_html(filepath)
    year = extract_year(filepath)
    sections = extract_sections(soup)
    financial_tables = extract_financial_tables(soup)

    return {
        "year":             year,
        "source_file":      os.path.abspath(filepath),
        "extracted_at":     datetime.now(timezone.utc).isoformat(),
        "sections":         sections,
        "financial_tables": financial_tables,
    }


def parse_all_reports(data_dir: str = "data/raw") -> list[dict]:
    """
    data/raw/ 내 .htm 파일 전체를 순회하며 parse_single_report() 호출.

    - 실패한 파일은 건너뛰고 경고 출력 (전체 중단 방지)
    - 반환: list of dict (연도 오름차순 정렬)
    """
    data_path = Path(data_dir)
    htm_files = sorted(data_path.glob("*.htm")) + sorted(data_path.glob("*.html"))

    if not htm_files:
        logger.warning("parse_all_reports: .htm 파일이 없습니다 (%s)", data_dir)
        return []

    results: list[dict] = []
    for fp in htm_files:
        try:
            result = parse_single_report(str(fp))
            results.append(result)
            logger.info("파싱 완료: %s (year=%d)", fp.name, result["year"])
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"파싱 실패, 건너뜀: {fp.name} — {exc}", stacklevel=2)

    results.sort(key=lambda r: r["year"])
    return results


# ---------------------------------------------------------------------------
# 내부 헬퍼
# ---------------------------------------------------------------------------

def _clean_cell(cell: Tag) -> str:
    """셀 텍스트 정제: 개행·연속 공백 제거."""
    text = cell.get_text(separator=" ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _deduplicate_columns(cols: list[str]) -> list[str]:
    """중복 컬럼명에 _1, _2 … suffix 추가."""
    seen: dict[str, int] = {}
    result: list[str] = []
    for c in cols:
        if c in seen:
            seen[c] += 1
            result.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 0
            result.append(c)
    return result


def _table_keyword_score(df: pd.DataFrame, keywords: list[str]) -> int:
    """DataFrame 전체 텍스트에서 키워드 출현 수 합산."""
    flat = " ".join(
        str(v) for v in df.values.flatten()
    ) + " " + " ".join(str(c) for c in df.columns)
    return sum(flat.count(kw) for kw in keywords)


def _collect_text_nodes(soup: BeautifulSoup) -> list[tuple[Tag, str]]:
    """
    soup 전체를 순회하면서 (tag, 정제된 텍스트) 쌍을 반환.

    중첩 방지 전략:
    - h2, h3 는 항상 포함 (섹션 경계 앵커)
    - p, td, th 는 하위에 span 이 없을 때만 포함
    - span 은 부모가 p/td/th 가 아닐 때(= 독립 span)만 포함
      → p 안의 span 이 이미 p 텍스트에 포함되는 중복을 방지
    단, p 내부가 span 으로만 구성될 경우 span 각각을 개별 노드로 처리하기 위해
    p 에 span 자식이 있으면 p 자체는 건너뛰고 자식 span 을 직접 등록.
    """
    nodes: list[tuple[Tag, str]] = []

    def _text(tag: Tag) -> str:
        t = tag.get_text(separator=" ", strip=True)
        return re.sub(r"\s+", " ", t).strip()

    for tag in soup.find_all(["h2", "h3", "p", "span", "td"]):
        name = tag.name

        if name in ("h2", "h3"):
            t = _text(tag)
            if t:
                nodes.append((tag, t))

        elif name in ("p", "td"):
            # 내부에 span 이 있으면 span 들을 개별 노드로 — p 자체는 스킵
            child_spans = tag.find_all("span", recursive=False)
            if child_spans:
                from bs4 import NavigableString
                for child in tag.children:
                    if isinstance(child, NavigableString):
                        t = re.sub(r"\s+", " ", str(child)).strip()
                        if len(t) > 10:   # 의미있는 길이만
                            nodes.append((tag, t))
                    elif child.name == "span":
                        t = _text(child)
                        if t:
                            nodes.append((child, t))
            else:
                t = _text(tag)
                if t:
                    nodes.append((tag, t))

        elif name == "span":
            # 부모가 p/td/th 면 이미 위에서 처리 → 스킵
            parent = tag.parent
            if parent and parent.name not in ("p", "td"):
                t = _text(tag)
                if t:
                    nodes.append((tag, t))

    return nodes


def _match_keyword(text: str, keywords: list[str]) -> bool:
    """keywords 중 하나라도 text에 포함되면 True."""
    return any(kw in text for kw in keywords)


def _slice_nodes_between(
    nodes: list[tuple[Tag, str]],
    start_kw: str,
    end_kw: str,
) -> list[tuple[Tag, str]]:
    """start_kw 이후 ~ end_kw 직전까지 노드 슬라이싱."""
    start = None
    end = None
    for i, (_, t) in enumerate(nodes):
        if start is None and start_kw in t:
            start = i
        elif start is not None and end_kw in t:
            end = i
            break
    if start is None:
        return []
    if end is None:
        return nodes[start:]
    return nodes[start:end]


def _slice_nodes_from(
    nodes: list[tuple[Tag, str]],
    start_kw: str,
) -> list[tuple[Tag, str]]:
    """start_kw 이후 끝까지 노드 슬라이싱."""
    for i, (_, t) in enumerate(nodes):
        if start_kw in t:
            return nodes[i:]
    return []


def _extract_audit_sections(
    nodes: list[tuple[Tag, str]],
    sections: dict[str, str],
) -> None:
    """
    감사보고서 본문 노드에서 감사의견 / 감사의견근거 / 핵심감사사항 추출.

    섹션 순서: 감사의견 → 감사의견근거 → (핵심감사사항) → 기타사항
    경계: 다음 섹션 키워드가 등장하면 이전 섹션 닫힘.

    Notes
    -----
    - '감사의견근거' 키워드는 '감사의견' 을 포함하므로 긴 것 우선 매칭.
    - '기타사항', '재무제표에 대한 경영진' 등이 나오면 현 섹션을 닫음.
    """
    # 감사보고서 섹션 순서 (긴 키워드 먼저 — '감사의견근거'가 '감사의견' 포함하므로)
    audit_order = ["핵심감사사항", "감사의견근거", "감사의견"]
    # 섹션 종료 트리거: 감사보고서 구간은 이미 (첨부) 이전으로 슬라이싱됨.
    # 서명부(감사법인명/대표이사) 등장 시 섹션 종료.
    # '내부회계관리', '기타사항', '재무제표에 대한 경영진' 등은 본문에 자연히
    # 등장할 수 있으므로 end_trigger로 사용하지 않음.
    end_triggers = [
        "(첨부)",                       # h2 재무제표 앵커
        "회 계 법 인", "회계법인",      # 감사법인 서명
        "대 표 이 사", "대표이사",      # 법인 대표 서명
    ]

    current_section: Optional[str] = None
    buf: list[str] = []

    for tag, text in nodes:
        # 종료 트리거 확인
        if any(kw in text for kw in end_triggers):
            if current_section:
                existing = sections.get(current_section, "")
                new_text = _join_buf(buf)
                if existing:
                    sections[current_section] = existing + "\n" + new_text
                else:
                    sections[current_section] = new_text
            break

        # 새 섹션 시작 감지 (헤더성 짧은 텍스트)
        matched_section = None
        if tag.name != "th":
            matched_section = _detect_audit_section(text, audit_order)
            
        if matched_section and matched_section != current_section:
            # 이전 섹션 저장
            if current_section:
                existing = sections.get(current_section, "")
                new_text = _join_buf(buf)
                if existing:
                    sections[current_section] = existing + "\n" + new_text
                else:
                    sections[current_section] = new_text
            current_section = matched_section
            buf = []
            continue  # 헤더 텍스트 자체는 본문에서 제외

        # 현재 섹션에 텍스트 추가
        if current_section:
            buf.append(text)

    # 마지막 섹션 저장
    if current_section:
        existing = sections.get(current_section, "")
        new_text = _join_buf(buf)
        if existing:
            sections[current_section] = existing + "\n" + new_text
        else:
            sections[current_section] = new_text


def _detect_audit_section(text: str, order: list[str]) -> Optional[str]:
    """
    짧은 텍스트(≤12자)이고 섹션 키워드와 정확히 일치하면 섹션명 반환.
    긴 텍스트는 본문으로 간주.

    Notes
    -----
    감사의견 헤더는 보통 '감사의견' 4자 단독으로 등장.
    감사의견근거는 '감사의견근거' 7자.
    핵심감사사항은 '핵심감사사항' 6자 (후행 공백 포함 가능).
    """
    stripped = text.strip()
    # 헤더 판단 기준: 12자 이하이고 키워드가 텍스트 전체를 구성
    if len(stripped) <= 12:
        for sec in order:
            kws = _SECTION_KEYWORDS[sec]
            if any(kw in stripped for kw in kws):
                return sec
    return None

'''
def _extract_fs_sections(
    nodes: list[tuple[Tag, str]],
    sections: dict[str, str],
) -> None:
    """
    재무제표 본문 노드에서 재무상태표 / 포괄손익 / 현금흐름 / 주석 추출.
    """
    fs_order = ["재무상태표", "포괄손익", "현금흐름", "주석"]
    current_section: Optional[str] = None
    buf: list[str] = []

    for tag, text in nodes:
        matched_section = None
        if tag.name != "th":
            matched_section = _detect_fs_section(text, fs_order)
            
        if matched_section and matched_section != current_section:
            if current_section:
                existing = sections.get(current_section, "")
                new_text = _join_buf(buf)
                if existing:
                    sections[current_section] = existing + "\n" + new_text
                else:
                    sections[current_section] = new_text
            current_section = matched_section
            buf = []
            continue

        if current_section:
            buf.append(text)

    # 마지막 섹션 저장 (주석은 문서 끝까지)
    if current_section:
        existing = sections.get(current_section, "")
        new_text = _join_buf(buf)
        if existing:
            sections[current_section] = existing + "\n" + new_text
        else:
            sections[current_section] = new_text
'''
def _extract_fs_sections(
    nodes: list[tuple[Tag, str]],
    sections: dict[str, str],
) -> None:
    """
    재무제표 본문 노드에서 재무상태표 / 포괄손익 / 현금흐름 / 주석 추출.
    주석 섹션 진입 후에는 섹션 전환 Lock — 내부 표 제목 오탐 방지.
    """
    fs_order = ["재무상태표", "포괄손익", "현금흐름", "주석"]
    current_section: Optional[str] = None
    buf: list[str] = []
    section_locked = False  # 주석 진입 후 전환 차단 플래그

    for tag, text in nodes:
        matched_section = None

        # Lock 상태(주석 진입 후)에서는 섹션 전환 판단 자체를 건너뜀
        if not section_locked and tag.name != "th":
            matched_section = _detect_fs_section(text, fs_order)

        if matched_section and matched_section != current_section:
            if current_section:
                existing = sections.get(current_section, "")
                new_text = _join_buf(buf)
                if existing:
                    sections[current_section] = existing + "\n" + new_text
                else:
                    sections[current_section] = new_text
            current_section = matched_section
            buf = []

            # 주석 섹션에 진입하는 순간 Lock 활성화
            if current_section == "주석":
                section_locked = True
            continue

        if current_section:
            buf.append(text)

    # 마지막 섹션 저장 (주석은 문서 끝까지)
    if current_section:
        existing = sections.get(current_section, "")
        new_text = _join_buf(buf)
        if existing:
            sections[current_section] = existing + "\n" + new_text
        else:
            sections[current_section] = new_text

def _detect_fs_section(text: str, order: list[str]) -> Optional[str]:
    """재무제표 섹션 헤더 감지. 50자 이하 + 키워드 포함."""
    stripped = text.strip()
    if len(stripped) <= 50 and "주석은" not in stripped and "요약" not in stripped:
        for sec in order:
            kws = _SECTION_KEYWORDS[sec]
            if any(kw in stripped for kw in kws):
                return sec
    return None


def _join_buf(buf: list[str]) -> str:
    """버퍼 텍스트를 줄바꿈으로 합치고 중복 공백 정제."""
    raw = "\n".join(buf)
    # 연속 줄바꿈 3개 이상 → 2개로
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    return raw.strip()
