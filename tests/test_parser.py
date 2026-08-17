import pytest
from pathlib import Path

def test_extract_year_normal():
    from src.parser.html_parser import extract_year
    assert extract_year("감사보고서_2019.htm") == 2019

def test_extract_year_no_digit_raises():
    from src.parser.html_parser import extract_year
    with pytest.raises(ValueError):
        extract_year("감사보고서_없음.htm")

def test_parse_all_reports_count(all_reports):
    assert len(all_reports) == 11

def test_all_years_present(report_by_year):
    expected = set(range(2014, 2025))
    assert expected == set(report_by_year.keys())

@pytest.mark.parametrize("year", range(2014, 2025))
def test_mandatory_sections_exist(report_by_year, year):
    """감사의견·재무상태표·포괄손익·현금흐름·주석은 전 연도 필수"""
    sections = report_by_year[year]["sections"]
    for key in ["감사의견", "재무상태표", "포괄손익", "현금흐름", "주석"]:
        assert len(sections[key]) > 0, f"{year}년 {key} 비어있음"

@pytest.mark.parametrize("year", range(2018, 2025))
def test_post_2018_sections(report_by_year, year):
    """핵심감사사항·감사의견근거는 2018년 이후 필수"""
    sections = report_by_year[year]["sections"]
    for key in ["핵심감사사항", "감사의견근거"]:
        assert len(sections[key]) > 0, f"{year}년 {key} 비어있음"

THRESHOLDS = {
    "감사의견":     (range(2014, 2025), 50),
    "재무상태표":   (range(2014, 2025), 500),
    "포괄손익":     (range(2014, 2025), 100),
    "현금흐름":     (range(2014, 2025), 1000),
    "주석":         (range(2018, 2025), 5000),
    "핵심감사사항": (range(2018, 2025), 50),
}

@pytest.mark.parametrize("section,year,threshold", [
    (sec, yr, th)
    for sec, (years, th) in THRESHOLDS.items()
    for yr in years
])
def test_section_min_length(report_by_year, section, year, threshold):
    text = report_by_year[year]["sections"].get(section, "")
    assert len(text) >= threshold, \
        f"{year}년 {section}: {len(text)}자 (최소 {threshold}자 필요)"

@pytest.mark.parametrize("year", range(2014, 2025))
def test_financial_tables_extracted(report_by_year, year):
    """balance_sheet, income_statement, cash_flow 테이블 최소 1개 이상 존재"""
    ft = report_by_year[year].get("financial_tables", {})
    assert len(ft) >= 1, f"{year}년 financial_tables 비어있음"

def test_report_metadata_fields(all_reports):
    """year, source_file, extracted_at, sections, financial_tables 키 존재"""
    required_keys = {"year", "source_file", "extracted_at", "sections", "financial_tables"}
    for r in all_reports:
        assert required_keys.issubset(r.keys())

def test_source_file_exists(all_reports):
    """source_file 경로가 실제로 존재하는 파일인지 확인"""
    for r in all_reports:
        assert Path(r["source_file"]).exists(), f"파일 없음: {r['source_file']}"
