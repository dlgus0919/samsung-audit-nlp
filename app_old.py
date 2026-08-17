import json
import re
import streamlit as st
from datetime import datetime
from src.rag.qa_pipeline import RAGPipeline

# ──────────────────────────────────────────────────────────────
# 프롬프팅 가이드 (검색 방식 기반으로 작성)
# ──────────────────────────────────────────────────────────────
_PROMPT_GUIDE = """
## 📌 질문 유형별 작성법

### ✅ 단일 연도 · 수치 조회 (정확도 최상)
연도를 명시하고 재무 키워드를 포함하면 해당 섹션이 우선 검색됩니다.
> "**2021년** 매출액과 영업이익은 얼마인가요?"
> "**2023년** 핵심감사사항은 무엇인가요?"
> "**2019년** 현금흐름표에서 영업활동 현금흐름을 알려주세요."

### ✅ 멀티연도 트렌드 · 비교 조회
아래 키워드 중 하나가 포함되면 연도별 균등 검색(1청크/연도)으로 전환됩니다.
`트렌드`, `추이`, `변화`, `변경`, `비교`, `증가`, `감소`, `연도별`, `역대`, `언제`
> "감사의견이 **변경된** 연도가 있나요?"
> "핵심감사사항 **추이**를 설명해주세요."
> "2018년과 2019년의 현금흐름을 **비교**해주세요."

### ✅ 특정 섹션 지정 조회
질문에 섹션 키워드를 포함하면 해당 섹션 청크가 우선 확보됩니다.

| 질문 키워드 | 우선 검색 섹션 |
|---|---|
| 감사의견, 적정의견, 한정의견 | 감사의견 |
| 핵심감사사항, 핵심감사, 리스크, 위험 | 핵심감사사항 |
| 매출, 영업이익, 순이익 | 포괄손익 |
| 자산총계, 부채총계, 자본총계 | 재무상태표 |
| 현금흐름, 영업활동, 투자활동 | 현금흐름 |

---

## ⚠️ 데이터 가용 범위

| 섹션 | 가용 연도 | 비고 |
|---|---|---|
| 감사의견 | 2014 – 2024 | 전 연도 적정의견 |
| 감사의견근거 | 2014 – 2024 | |
| **핵심감사사항** | **2018 – 2024** | 2014~2017 없음 |
| 재무상태표 | 2014 – 2024 | |
| 포괄손익 | 2014 – 2024 | |
| 현금흐름 | 2014 – 2024 | |
| 주석 | 2014 – 2024 | 방대한 분량 |

---

## 🔧 사이드바 옵션 활용법

- **연도 필터**: 단일 연도 집중 분석 시 설정 — "2021년 매출" 질문에서 필터를 2021로 고정하면 정확도 향상
- **검색 문서 수(k)**: 복잡한 트렌드 질문 → 6 ~ 8, 단순 수치 조회 → 2 ~ 3 추천

---

## ❌ 피해야 할 패턴

- **너무 광범위한 질문**: "2021년 전반적인 재무 상황을 알려주세요"
  → 대신: "2021년 매출, 영업이익, 총자산을 각각 알려주세요"
- **2014~2017년에 핵심감사사항 질문**: 해당 섹션이 존재하지 않아 다른 섹션으로 대체
- **두 가지 이상 섹션을 한 번에 조회**: "2021년 감사의견과 영업이익은?"
  → 대신 질문을 나눠서 입력
- **수치 없는 연도에 수치 요구**: 수치가 없는 연도는 '자료 없음'으로 표시됩니다
"""

def _strip_chunk_prefix(text: str) -> str:
    """임베딩용 '[YYYY년 섹션]' prefix를 UI 표시 시 제거"""
    return re.sub(r'^\[\d{4}년 [^\]]+\] ', '', text)


def _build_export_md(messages: list, include_sources: bool = True) -> str:
    """채팅 히스토리를 Markdown 문자열로 직렬화"""
    lines = [
        "# 삼성전자 감사보고서 QA 대화 내역",
        f"내보내기: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
    ]
    q_num = 0
    i = 0
    while i < len(messages):
        msg = messages[i]
        if msg["role"] == "user":
            q_num += 1
            lines.append(f"## Q{q_num}. {msg['content']}")
            lines.append("")
            # 바로 다음 assistant 메시지 탐색
            if i + 1 < len(messages) and messages[i + 1]["role"] == "assistant":
                asst = messages[i + 1]
                lines.append("**답변:**")
                lines.append(asst["content"])
                lines.append("")
                if asst.get("tool_rows"):
                    lines.append("**도구 조회 결과 (정형 데이터):**")
                    for row in asst["tool_rows"]:
                        joined = ", ".join(row.get("values", [])) if row.get("values") else "값 없음"
                        lines.append(
                            f"- {row.get('year')}년 {row.get('table_type')} {row.get('item')}: {joined}"
                        )
                    lines.append("")
                if asst.get("evidence_spans"):
                    lines.append("**핵심 근거 스팬:**")
                    for ev in asst["evidence_spans"]:
                        lines.append(f"- ({ev['year']} {ev['section']}) {ev['span']}")
                    lines.append("")
                if include_sources and asst.get("contexts"):
                    ctx_list = asst["contexts"]
                    sources = ", ".join(
                        list(dict.fromkeys(f"{c['year']}년 {c['section']}" for c in ctx_list))
                    )
                    lines.append(f"**참고 출처:** {sources}")
                    lines.append("")
                    lines.append("### 참고 문서")
                    for j, ctx in enumerate(ctx_list, 1):
                        lines.append(f"**문서 {j}** ({ctx['year']} {ctx['section']})")
                        clean = _strip_chunk_prefix(ctx["text"])
                        for row in clean.splitlines():
                            lines.append(f"> {row}" if row.strip() else ">")
                        lines.append("")
                i += 1  # assistant 메시지 소비
            lines.append("---")
            lines.append("")
        i += 1
    return "\n".join(lines)


def _build_export_json(messages: list) -> str:
    """채팅 히스토리를 JSON 문자열로 직렬화"""
    conversations = []
    i = 0
    turn = 0
    while i < len(messages):
        msg = messages[i]
        if msg["role"] == "user":
            turn += 1
            entry = {"turn": turn, "question": msg["content"], "answer": "", "sources": "", "contexts": []}
            if i + 1 < len(messages) and messages[i + 1]["role"] == "assistant":
                asst = messages[i + 1]
                entry["answer"] = asst["content"]
                entry["tool_rows"] = asst.get("tool_rows", [])
                entry["evidence_spans"] = asst.get("evidence_spans", [])
                if asst.get("contexts"):
                    entry["sources"] = ", ".join(
                        list(dict.fromkeys(f"{c['year']}년 {c['section']}" for c in asst["contexts"]))
                    )
                    entry["contexts"] = [
                        {"year": c["year"], "section": c["section"],
                         "text": _strip_chunk_prefix(c["text"])}
                        for c in asst["contexts"]
                    ]
                i += 1
            conversations.append(entry)
        i += 1
    payload = {
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "total_turns": turn,
        "conversations": conversations,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)

st.set_page_config(
    page_title="삼성전자 감사보고서 QA",
    page_icon="📊",
    layout="wide"
)

@st.cache_resource
def get_pipeline():
    return RAGPipeline()

try:
    pipeline = get_pipeline()
except Exception as e:
    st.error(
        "파이프라인 초기화에 실패했습니다. "
        f"환경변수/Secrets 설정을 확인해주세요.\n\n상세 오류: {e}"
    )
    st.stop()

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if "is_generating" not in st.session_state:
    st.session_state.is_generating = False
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# 사이드바
with st.sidebar:
    is_generating = st.session_state.get("is_generating", False)
    st.caption(f"활성 백엔드: `{pipeline.backend}`")
    if pipeline.backend == "local":
        st.caption(f"로컬 모델: `{pipeline.local_model}`")

    with st.popover("📖 프롬프팅 가이드", use_container_width=True):
        st.markdown(_PROMPT_GUIDE)

    st.header("⚙️ RAG 설정")
    year_options = ["전체"] + list(range(2014, 2025))
    selected_year = st.selectbox("연도 필터", year_options)
    year_filter = None if selected_year == "전체" else int(selected_year)
    
    k_slider = st.slider("검색 문서 수 (k)", min_value=1, max_value=8, value=4)
    
    if st.button("인덱스 재빌드 🔄", disabled=is_generating):
        with st.spinner("FAISS 인덱스를 다시 빌드하는 중..."):
            pipeline._rebuild_vector_index()
        st.success("인덱스 갱신 완료!")
            
    st.divider()
    st.markdown("**대화 내보내기**")
    has_chat = bool(st.session_state.get("messages"))
    include_src = st.checkbox("참고 문서 포함", value=True, disabled=not has_chat)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    col1, col2 = st.columns(2)
    with col1:
        md_data = (
            _build_export_md(st.session_state.messages, include_src)
            if has_chat else ""
        )
        st.download_button(
            label="Markdown",
            data=md_data,
            file_name=f"qa_{ts}.md",
            mime="text/markdown",
            disabled=not has_chat,
            use_container_width=True,
        )
    with col2:
        json_data = (
            _build_export_json(st.session_state.messages)
            if has_chat else ""
        )
        st.download_button(
            label="JSON",
            data=json_data,
            file_name=f"qa_{ts}.json",
            mime="application/json",
            disabled=not has_chat,
            use_container_width=True,
        )
    if not has_chat:
        st.caption("대화를 시작하면 내보내기가 활성화됩니다.")

    st.divider()
    st.markdown("**예시 질문**")
    questions = [
        "2021년 매출과 영업이익은 얼마인가요?",
        "2023년 핵심감사사항은 무엇인가요?",
        "코로나19(COVID-19)로 인한 불확실성이 언급된 연도와 그 내용을 알려주세요.",
        "2018년과 2019년의 현금흐름은 어떻게 달라졌나요?",
        "감사의견이 변경된 연도가 있나요?",
        "주요 재무 리스크 트렌드를 설명해주세요."
    ]
    for q in questions:
        if st.button(q, disabled=is_generating):
            # 사용자가 사이드바 질문을 누르면 채팅 인풋을 대신해 state에 값을 넣기 위험하지만,
            # Streamlit 특성상 button click 시 rerun 되므로 session_state로 초기 텍스트 설정
            st.session_state["shortcut_q"] = q

st.title("📊 삼성전자 감사보고서 분석 QA (2014-2024)")

# 히스토리 렌더링
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("tool_rows"):
            with st.expander("🛠️ 도구 조회 결과"):
                for row in message["tool_rows"]:
                    joined = ", ".join(row.get("values", [])) if row.get("values") else "값 없음"
                    st.markdown(
                        f"- **{row.get('year')}년 / {row.get('table_type')} / {row.get('item')}**: {joined}"
                    )
        if message.get("evidence_spans"):
            with st.expander("🧷 핵심 근거 스팬"):
                for idx, ev in enumerate(message["evidence_spans"], 1):
                    st.markdown(f"**근거 {idx}** ({ev['year']} {ev['section']})")
                    st.caption(ev["span"])
        if "contexts" in message:
            with st.expander("📌 참고 문서"):
                sources = pipeline.get_sources(message["contexts"])
                st.write(f"**출처:** {sources}")
                for i, ctx in enumerate(message["contexts"]):
                    st.markdown(f"**문서 {i+1}** ({ctx['year']} {ctx['section']})")
                    st.text(_strip_chunk_prefix(ctx['text']))

# 사이드바 버튼/직접 입력 질문 처리
input_disabled = st.session_state.get("is_generating", False)
query = st.chat_input("질문을 입력하세요...", disabled=input_disabled)
if not input_disabled and st.session_state.get("shortcut_q"):
    query = st.session_state.pop("shortcut_q")

# 입력을 즉시 처리하지 않고 대기열에 넣은 뒤 rerun하여 입력창 비활성 상태를 먼저 반영
if query and not st.session_state.is_generating and not st.session_state.pending_query:
    st.session_state.pending_query = query
    st.session_state.is_generating = True
    st.rerun()

# 실제 응답 생성은 pending_query가 있을 때만 수행
if st.session_state.pending_query:
    query = st.session_state.pending_query
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        contexts = []
        tool_rows = []
        evidence_spans = []
        retrieval_error = None
        with st.spinner("관련 문서를 검색 중입니다..."):
            try:
                contexts = pipeline.get_contexts(
                    query,
                    year_filter=year_filter,
                    retrieve_k=k_slider,
                )
                tool_rows = pipeline.run_tool_calling(query, year_filter=year_filter)
                evidence_spans = pipeline.extract_evidence_spans(query, contexts, max_spans=3)
            except Exception as e:
                retrieval_error = f"문서 검색 중 오류가 발생했습니다: {str(e)}"

        if retrieval_error:
            response = retrieval_error
            st.error(response)
        else:
            with st.expander("📌 참고 문서"):
                sources = pipeline.get_sources(contexts)
                st.write(f"**출처:** {sources}")
                if tool_rows:
                    st.markdown("**도구 조회 결과 (financial_data.csv):**")
                    for row in tool_rows:
                        joined = ", ".join(row.get("values", [])) if row.get("values") else "값 없음"
                        st.markdown(
                            f"- {row.get('year')}년 / {row.get('table_type')} / {row.get('item')} / 값: {joined}"
                        )
                if evidence_spans:
                    st.markdown("**핵심 근거 스팬:**")
                    for idx, ev in enumerate(evidence_spans, 1):
                        st.markdown(f"{idx}. ({ev['year']} {ev['section']}) {ev['span']}")
                for i, ctx in enumerate(contexts):
                    st.markdown(f"**문서 {i+1}** ({ctx['year']} {ctx['section']})")
                    st.text(_strip_chunk_prefix(ctx['text']))
            
            try:
                # 로컬 대형 모델(MPS/CPU)에서는 스트리밍이 멈춘 것처럼 보일 수 있어
                # 비스트리밍 응답으로 전환해 안정적으로 결과를 반환한다.
                use_stream = pipeline.backend != "local"
                result = pipeline.ask(
                    query,
                    year_filter=year_filter,
                    stream=use_stream,
                    contexts=contexts,
                    retrieve_k=k_slider,
                )
                if use_stream:
                    response = st.write_stream(result)
                else:
                    response = str(result)
                    st.markdown(response)
            except Exception as e:
                response = f"응답 생성 중 오류가 발생했습니다: {str(e)}"
                st.error(response)
            
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "contexts": contexts,
        "tool_rows": tool_rows,
        "evidence_spans": evidence_spans,
    })

    st.session_state.pending_query = None
    st.session_state.is_generating = False
    st.rerun()
