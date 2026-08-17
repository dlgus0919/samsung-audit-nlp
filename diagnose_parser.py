"""
HTML 파서 진단 스크립트 - 2018 재무상태표, 2023/2024 주석 이슈 원인 파악
사용법: cd ~/samsung_audit_nlp && python diagnose_parser.py
"""
import re
import os
from pathlib import Path

try:
    from bs4 import BeautifulSoup, NavigableString
except ImportError:
    print("pip install beautifulsoup4 lxml 실행 필요")
    exit(1)

RAW_DIR = Path("data/raw")

def load_soup(year):
    path = RAW_DIR / f"감사보고서_{year}.htm"
    if not path.exists():
        print(f"파일 없음: {path}")
        return None
    for enc in ["euc-kr", "cp949", "utf-8"]:
        try:
            text = path.read_text(encoding=enc)
            return BeautifulSoup(text, "lxml")
        except Exception:
            continue
    return None

def _text(tag):
    return re.sub(r"\s+", " ", tag.get_text()).strip()

def find_section_context(soup, keywords, label, window=5):
    """키워드 주변 태그들을 출력해서 구조 파악"""
    print(f"\n--- [{label}] 키워드 탐색 ---")
    all_tags = soup.find_all(["h1","h2","h3","p","span","td","th","div"])
    for i, tag in enumerate(all_tags):
        t = _text(tag)
        for kw in keywords:
            if kw in t and len(t) < 80:
                print(f"  [{i}] <{tag.name}> 텍스트: {repr(t[:100])}")
                # 주변 태그 5개씩
                for j in range(max(0, i-2), min(len(all_tags), i+window)):
                    nearby = all_tags[j]
                    nt = _text(nearby)
                    marker = ">>>" if j == i else "   "
                    print(f"  {marker} [{j}] <{nearby.name}> {repr(nt[:80])}")
                print()
                break

def show_section_transition(soup, start_kws, end_kws, label, max_nodes=30):
    """섹션 시작~끝 사이 텍스트 노드 출력"""
    print(f"\n=== [{label}] 섹션 전환 분석 ===")
    all_tags = soup.find_all(["h2","h3","p","span","td"])

    in_section = False
    count = 0
    for tag in all_tags:
        t = _text(tag)
        if not t:
            continue

        # 섹션 시작
        if any(kw in t for kw in start_kws) and len(t) < 60:
            in_section = True
            print(f"  [START] <{tag.name}> {repr(t[:80])}")
            count = 0
            continue

        if in_section:
            count += 1
            if count <= max_nodes:
                print(f"  [{count:02d}] <{tag.name}> len={len(t):>5} {repr(t[:70])}")

            # 섹션 종료 감지
            if any(kw in t for kw in end_kws) and len(t) < 60:
                print(f"  [END] <{tag.name}> {repr(t[:80])}")
                in_section = False

def analyze_2018_bs(soup):
    """2018 재무상태표 전용 분석"""
    print("\n" + "="*60)
    print("2018 재무상태표 분석")
    print("="*60)

    # 재무상태표 키워드들 탐색
    find_section_context(soup, ["재 무 상 태 표", "재무상태표", "財務狀態表"], "재무상태표", window=8)

    # 재무상태표 -> 포괄손익 전환 분석
    show_section_transition(
        soup,
        ["재 무 상 태 표", "재무상태표"],
        ["포 괄 손 익", "포괄손익", "손익계산서"],
        "재무상태표→포괄손익",
        max_nodes=20
    )

    # TABLE 구조 확인
    print("\n--- 재무상태표 테이블 확인 ---")
    tables = soup.find_all("table")
    for i, tbl in enumerate(tables):
        first_cell = _text(tbl.find(["td","th"]) or tbl)
        tbl_text = _text(tbl)
        if "자산" in tbl_text or "부채" in tbl_text or "재무상태" in tbl_text:
            print(f"  테이블[{i}] 첫번째셀={repr(first_cell[:40])} 전체길이={len(tbl_text)}")
            # 테이블 내 주석 키워드 확인
            th_texts = [_text(th) for th in tbl.find_all("th")]
            td_texts = [_text(td) for td in tbl.find_all("td")][:5]
            print(f"    th태그들: {th_texts[:10]}")
            print(f"    첫5 td: {td_texts}")

def analyze_2023_2024_notes(soup, year):
    """2023/2024 주석 섹션 분석"""
    print(f"\n{'='*60}")
    print(f"{year} 주석 분석")
    print(f"{'='*60}")

    # 주석 키워드 탐색
    find_section_context(soup, ["주   석", "주 석", "주석"], "주석 키워드", window=5)

    # 주석 시작 이후 얼마나 수집되는지 확인
    show_section_transition(
        soup,
        ["주   석", "주 석"],
        ["==NEVER_END=="],  # 끝없이 계속
        f"주석 시작 이후",
        max_nodes=15
    )

    # 전체 텍스트 노드 수 확인
    all_tags = soup.find_all(["h2","h3","p","span","td"])
    total_chars = sum(len(_text(t)) for t in all_tags)
    print(f"\n  전체 텍스트 노드: {len(all_tags)}개, 총 {total_chars:,}자")

    # 주석 섹션 이후 내용이 특정 태그 타입에 집중되는지 확인
    print("\n--- 주석 이후 태그 타입 분포 ---")
    all_tags2 = soup.find_all(True)  # 모든 태그
    note_start = False
    tag_counts = {}
    note_chars = 0
    for tag in all_tags2:
        t = _text(tag)
        if not note_start:
            if any(kw in t for kw in ["주   석", "주 석"]) and len(t) < 30:
                note_start = True
        else:
            note_chars += len(t)
            tag_counts[tag.name] = tag_counts.get(tag.name, 0) + 1

    print(f"  주석 이후 태그 분포: {dict(sorted(tag_counts.items(), key=lambda x: -x[1])[:15])}")
    print(f"  주석 이후 누적 문자: {note_chars:,}")

def main():
    # 현재 파서로 각 연도 결과 확인
    print("=== 현재 파서 결과 ===")
    try:
        import sys
        sys.path.insert(0, '.')
        from src.parser.html_parser import parse_all_reports
        reports = parse_all_reports('data/raw')
        for r in sorted(reports, key=lambda x: x['year']):
            y = r['year']
            s = r['sections']
            bs = len(s.get('재무상태표',''))
            notes = len(s.get('주석',''))
            flag = ""
            if bs < 500: flag += f" ❌재무상태표={bs}"
            if notes < 5000 and y >= 2018: flag += f" ❌주석={notes}"
            if flag:
                print(f"  {y}:{flag}")
            else:
                print(f"  {y}: ✅")
    except Exception as e:
        print(f"  파서 임포트 오류: {e}")

    print("\n" + "="*60)
    print("HTML 구조 진단 시작")
    print("="*60)

    # 2018 재무상태표
    soup2018 = load_soup(2018)
    if soup2018:
        analyze_2018_bs(soup2018)

    # 2023 주석
    soup2023 = load_soup(2023)
    if soup2023:
        analyze_2023_2024_notes(soup2023, 2023)

    # 2024 주석
    soup2024 = load_soup(2024)
    if soup2024:
        analyze_2023_2024_notes(soup2024, 2024)

if __name__ == "__main__":
    main()
