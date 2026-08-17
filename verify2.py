import sys; sys.path.insert(0, '.')
from src.parser.html_parser import parse_all_reports

reports = parse_all_reports('data/raw')
for r in sorted(reports, key=lambda x: x['year']):
    y = r['year']
    s = r['sections']
    notes = len(s.get('주석',''))
    bs = len(s.get('재무상태표',''))
    cf = len(s.get('현금흐름',''))
    
    issues = []
    if bs < 500: issues.append(f"재무상태표={bs}")
    if cf < 1000: issues.append(f"현금흐름={cf}")
    if y >= 2018 and notes < 5000: issues.append(f"주석={notes}")
    
    print(f"{'❌' if issues else '✅'} {y}: {', '.join(issues) if issues else 'OK'}")
