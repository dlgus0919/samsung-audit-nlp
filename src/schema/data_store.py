import sqlite3
import pandas as pd
import json
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path

def init_db(db_path: str = "data/processed/audit.db") -> sqlite3.Connection:
    """
    DB 파일이 없으면 생성, 있으면 연결 반환.
    3개 테이블을 CREATE TABLE IF NOT EXISTS 로 초기화.
    """
    # Create directory if it doesn't exist
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 1. 섹션 텍스트 테이블
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS sections (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        year        INTEGER NOT NULL,
        section     TEXT NOT NULL,
        content     TEXT,
        char_count  INTEGER,
        source_file TEXT,
        extracted_at TEXT
    )
    ''')
    
    # 2. 재무제표 보조 데이터
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS financial_data (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        year        INTEGER NOT NULL,
        table_type  TEXT NOT NULL,
        item        TEXT,
        value_raw   TEXT,
        source_file TEXT,
        extracted_at TEXT
    )
    ''')
    
    # 3. 파싱 이력
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS parse_log (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        year         INTEGER,
        source_file  TEXT,
        status       TEXT,
        error_msg    TEXT,
        extracted_at TEXT
    )
    ''')
    
    conn.commit()
    return conn

def save_report(report: dict, conn: sqlite3.Connection) -> None:
    """
    parse_single_report() 반환값 1개를 DB에 저장.
    중복 연도 삽입 시 기존 데이터 덮어쓰기 (DELETE → INSERT).
    """
    cursor = conn.cursor()
    target_year = report["year"]
    
    # Delete existing data for this year to allow overwrites
    cursor.execute("DELETE FROM sections WHERE year = ?", (target_year,))
    cursor.execute("DELETE FROM financial_data WHERE year = ?", (target_year,))
    
    extracted_time = report["extracted_at"]
    source_file = report["source_file"]
    
    # Insert Sections
    for section_name, content in report["sections"].items():
        cursor.execute('''
            INSERT INTO sections 
            (year, section, content, char_count, source_file, extracted_at)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            target_year, 
            section_name, 
            content, 
            len(content) if content else 0, 
            source_file, 
            extracted_time
        ))
        
    # Insert Financial Tables
    for table_type, df in report["financial_tables"].items():
        if df.empty or len(df.columns) == 0:
            continue
            
        # 행 단위로 분해: 첫 열이 아이템 명, 나머지가 금액 (JSON 스트링화 저장)
        for _, row in df.iterrows():
            item_name = str(row.iloc[0]).strip()
            # 배열로 나머지 열 값 취합 후 JSON 문자열 처리
            value_raw = json.dumps(row.iloc[1:].tolist(), ensure_ascii=False)
            
            cursor.execute('''
                INSERT INTO financial_data
                (year, table_type, item, value_raw, source_file, extracted_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                target_year,
                table_type,
                item_name,
                value_raw,
                source_file,
                extracted_time
            ))
            
    # Success Log (only recording per report here)
    cursor.execute('''
        INSERT INTO parse_log (year, source_file, status, error_msg, extracted_at)
        VALUES (?, ?, ?, ?, ?)
    ''', (target_year, source_file, 'success', '', extracted_time))
    
    conn.commit()

def save_all_reports(reports: list[dict], db_path: str = "data/processed/audit.db") -> None:
    """
    parse_all_reports() 반환값 전체를 순회하며 save_report() 호출.
    실패한 연도는 parse_log에 상태 기록.
    """
    conn = init_db(db_path)
    cursor = conn.cursor()
    
    for report in reports:
        try:
            save_report(report, conn)
        except Exception as e:
            # Handle possible errors during save and log them specifically into DB
            current_time = datetime.now(timezone.utc).isoformat()
            target_year = report.get("year", None)
            source_file = report.get("source_file", "unknown")
            error_msg = str(e) + "\n" + traceback.format_exc()
            
            cursor.execute('''
                INSERT INTO parse_log (year, source_file, status, error_msg, extracted_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (target_year, source_file, 'failed', error_msg, current_time))
            
            conn.commit()
            print(f"[ERROR] Failed to save DB for year {target_year}: {e}")
            
    # 연결은 마지막에 유지 또는 종결 (현재 사용 패턴 상 연결 정보가 외부 통제될 수도 있으므로 close 수행)
    conn.close()

def export_csv(conn: sqlite3.Connection, out_dir: str = "data/processed") -> None:
    """
    sections, financial_data 테이블을 각각 CSV로 내보냄.
    """
    os.makedirs(out_dir, exist_ok=True)
    
    df_sections = pd.read_sql_query("SELECT * FROM sections", conn)
    df_sections.to_csv(os.path.join(out_dir, "sections.csv"), index=False, encoding='utf-8-sig')
    
    df_financials = pd.read_sql_query("SELECT * FROM financial_data", conn)
    df_financials.to_csv(os.path.join(out_dir, "financial_data.csv"), index=False, encoding='utf-8-sig')
    
def validate(conn: sqlite3.Connection) -> dict:
    """
    데이터 품질 검증. 
    """
    df_sections = pd.read_sql_query("SELECT * FROM sections", conn)
    df_financials = pd.read_sql_query("SELECT * FROM financial_data", conn)
    
    total_years = df_sections['year'].nunique()
    
    # 1. 누락된 섹션 찾기
    # 모든 추출된 연도에 대해 각 년도별로 내용이 빈 (content_length == 0 또는 null) 섹션 색출
    empty_sections = df_sections[(df_sections['content'].isnull()) | (df_sections['char_count'] == 0)]
    missing_sections = []
    
    for _, row in empty_sections.iterrows():
        y = row['year']
        sec = row['section']
        # 2018년 미만의 '핵심감사사항', '감사의견근거'는 비어있는게 정상.
        if sec == '핵심감사사항' and int(y) < 2018:
            continue
        if sec == '감사의견근거' and int(y) < 2018:
            continue
        missing_sections.append({'year': y, 'section': sec})
        
    # 2. 재무 데이터가 비어있는 연도 찾기
    financial_years = df_financials['year'].unique()
    all_years = df_sections['year'].unique()
    
    empty_financial = [int(y) for y in all_years if y not in financial_years]
    
    passed = len(missing_sections) == 0 and len(empty_financial) == 0
    
    result = {
        'total_years': int(total_years),
        'missing_sections': missing_sections,
        'empty_financial': empty_financial,
        'passed': bool(passed)
    }
    return result

def load_sections(conn: sqlite3.Connection, 
                  years: list[int] | None = None, 
                  section: str | None = None) -> pd.DataFrame:
    """
    sections 테이블 조회.
    years: 특정 연도 배열 (ex: [2020, 2021])
    section: 특정 섹션 이름 조회 (ex: '감사의견')
    """
    query = "SELECT * FROM sections WHERE 1=1"
    params = []
    
    if years:
        placeholders = ",".join(["?"] * len(years))
        query += f" AND year IN ({placeholders})"
        params.extend(years)
        
    if section:
        query += " AND section = ?"
        params.append(section)
        
    df = pd.read_sql_query(query, conn, params=params)
    return df
