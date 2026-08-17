# src/parser/__init__.py
from .html_parser import (
    load_html,
    extract_year,
    extract_sections,
    extract_tables,
    extract_financial_tables,
    parse_single_report,
    parse_all_reports,
)

__all__ = [
    "load_html",
    "extract_year",
    "extract_sections",
    "extract_tables",
    "extract_financial_tables",
    "parse_single_report",
    "parse_all_reports",
]
