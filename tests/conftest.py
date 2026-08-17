# tests/conftest.py
import pytest
from pathlib import Path
import sys, tempfile, os

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"

@pytest.fixture(scope="session")
def all_reports():
    """11개 연도 전체 파싱 결과 (세션 전체에서 1회만 실행)"""
    from src.parser.html_parser import parse_all_reports
    return parse_all_reports(str(RAW_DIR))

@pytest.fixture(scope="session")
def report_by_year(all_reports):
    """연도 → 리포트 딕셔너리"""
    return {r["year"]: r for r in all_reports}

@pytest.fixture()
def tmp_db_path(tmp_path):
    """테스트용 임시 DB 경로"""
    return str(tmp_path / "test_audit.db")

@pytest.fixture()
def tmp_csv_dir(tmp_path):
    """테스트용 임시 CSV 디렉토리"""
    return str(tmp_path / "csv_out")
