import os
import sqlite3
import pandas as pd
from tests.conftest import RAW_DIR

def test_init_db_creates_tables(tmp_db_path):
    """init_db() 후 sections, financial_data, parse_log 테이블 존재"""
    from src.schema.data_store import init_db
    conn = init_db(tmp_db_path)
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cursor.fetchall()}
    assert {"sections", "financial_data", "parse_log"}.issubset(tables)
    conn.close()

def test_save_all_reports_row_count(tmp_db_path, all_reports):
    """save_all_reports() 후 sections 테이블에 11개 연도 행 존재"""
    from src.schema.data_store import init_db, save_all_reports
    
    # save_all_reports 시그니처: (reports: list[dict], db_path: str)
    save_all_reports(all_reports, tmp_db_path)
    
    conn = sqlite3.connect(tmp_db_path)
    cursor = conn.execute("SELECT COUNT(DISTINCT year) FROM sections")
    count = cursor.fetchone()[0]
    assert count == 11
    conn.close()

def test_export_csv_creates_files(tmp_db_path, tmp_csv_dir, all_reports):
    """export_csv() 후 sections.csv, financial_data.csv 파일 생성"""
    from src.schema.data_store import save_all_reports, export_csv
    
    save_all_reports(all_reports, tmp_db_path)
    
    conn = sqlite3.connect(tmp_db_path)
    export_csv(conn, tmp_csv_dir)
    conn.close()
    
    assert os.path.exists(os.path.join(tmp_csv_dir, "sections.csv"))
    assert os.path.exists(os.path.join(tmp_csv_dir, "financial_data.csv"))

def test_validate_passes(tmp_db_path, all_reports):
    """validate() 결과 passed == True"""
    from src.schema.data_store import save_all_reports, validate
    
    save_all_reports(all_reports, tmp_db_path)
    
    conn = sqlite3.connect(tmp_db_path)
    result = validate(conn)
    conn.close()
    
    assert result["passed"] is True, f"검증 실패: {result}"

def test_sections_csv_has_correct_years(tmp_db_path, tmp_csv_dir, all_reports):
    """sections.csv에 2014~2024 모든 연도 포함"""
    from src.schema.data_store import save_all_reports, export_csv
    
    save_all_reports(all_reports, tmp_db_path)
    
    conn = sqlite3.connect(tmp_db_path)
    export_csv(conn, tmp_csv_dir)
    conn.close()
    
    df = pd.read_csv(os.path.join(tmp_csv_dir, "sections.csv"))
    assert set(range(2014, 2025)).issubset(set(df["year"].unique()))
