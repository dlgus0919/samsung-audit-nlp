#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -d "venv" ]]; then
  # shellcheck disable=SC1091
  source venv/bin/activate
fi

# 로컬 Qwen 강제 모드: API 키 환경변수 누수 방지
unset OPENAI_API_KEY || true
unset ANTHROPIC_API_KEY || true
export LLM_BACKEND="local"
export LOCAL_MODEL="${LOCAL_MODEL:-Qwen/Qwen2.5-3B-Instruct}"

printf "[mode] local (Qwen)\n"
printf "[model] %s\n" "$LOCAL_MODEL"
exec streamlit run app.py
