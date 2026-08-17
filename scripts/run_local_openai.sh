#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -d "venv" ]]; then
  # shellcheck disable=SC1091
  source venv/bin/activate
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  read -r -s -p "OPENAI_API_KEY 입력: " OPENAI_API_KEY
  echo
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY가 비어 있어 실행할 수 없습니다."
  exit 1
fi

export OPENAI_API_KEY
export LLM_BACKEND="openai"
unset ANTHROPIC_API_KEY || true

printf "[mode] openai\n"
exec streamlit run app.py
