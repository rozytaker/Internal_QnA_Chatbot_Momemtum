"""
Momentum Africa — Internal Data Q&A Copilot (UC05)
====================================================
Streamlit front end for the NL -> pandas -> sandbox -> narrative pipeline
described on the UC05 slides. Run with:

    streamlit run app.py
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).parent))

from core import llm_engine, pipeline
import reranker
from core.data_loader import load_tables, data_window, load_meta
from core.retriever import SchemaRetriever, load_catalogue
from core.doc_store import DocumentStore

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Momentum Africa | Data Q&A Copilot",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

NAVY = "#0B1F3D"
ACCENT = "#C0392B"
TEAL = "#1B6F7E"
PURPLE = "#6B3FA0"
ORANGE = "#D2691E"
GREEN = "#1B7F5C"
BG = "#F4F5F8"
INK = "#1A2238"
MUTED = "#6B7280"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif;
}}
.stApp {{ background-color: {BG}; }}
footer {{ visibility: hidden; }}
section[data-testid="stSidebar"] {{
    background-color: #FFFFFF;
    border-right: 1px solid #E7E9EE;
}}
div.block-container {{ padding-top: 0rem; max-width: 1180px; }}

/* ---------- Header banner ---------- */
.mom-header {{
    background: linear-gradient(135deg, {NAVY} 0%, #142B52 100%);
    border-radius: 14px;
    padding: 28px 32px;
    margin-bottom: 18px;
    position: relative;
    overflow: hidden;
}}
.mom-header::after {{
    content: "";
    position: absolute; right: 0; top: 0; bottom: 0; width: 8px;
    background: {ACCENT};
}}
.mom-eyebrow {{
    color: #8FA7CC; font-size: 12px; font-weight: 700; letter-spacing: 1.5px;
    text-transform: uppercase; margin-bottom: 6px;
}}
.mom-title {{ color: white; font-size: 28px; font-weight: 800; margin: 0 0 6px 0; }}
.mom-subtitle {{ color: #C7D3E8; font-size: 14.5px; margin: 0 0 14px 0; }}
.mom-badge {{
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.18);
    color: #E5ECF7; font-size: 12.5px; font-weight: 600;
    padding: 6px 12px; border-radius: 999px; margin-right: 8px;
}}
.mom-brandrow {{
    color: #6E84AC; font-size: 11px; font-weight: 700; letter-spacing: 1.5px;
    margin-top: 16px; text-transform: uppercase;
}}

/* ---------- KPI ribbon ---------- */
.kpi-card {{
    background: white; border-radius: 10px; padding: 16px 18px;
    border-left: 4px solid {ACCENT};
    box-shadow: 0 1px 3px rgba(16,24,40,0.06);
    height: 100%;
}}
.kpi-value {{ font-size: 24px; font-weight: 800; color: {NAVY}; line-height: 1.1; }}
.kpi-label {{ font-size: 12px; color: {MUTED}; margin-top: 4px; }}

/* ---------- Answer card ---------- */
.answer-card {{
    background: white; border-radius: 10px; padding: 18px 20px;
    border-left: 4px solid {GREEN};
    box-shadow: 0 1px 3px rgba(16,24,40,0.06);
    font-size: 15.5px; color: {INK}; line-height: 1.55;
}}
.error-card {{
    background: #FDF1F0; border-radius: 10px; padding: 16px 20px;
    border-left: 4px solid {ACCENT}; color: #7A2018; font-size: 14.5px;
}}
.access-card {{
    background: #FFF8EC; border-radius: 10px; padding: 16px 20px;
    border-left: 4px solid {ORANGE}; color: #6B4408; font-size: 14.5px;
}}
.trace-chip {{
    display: inline-block; background: #EEF1F7; color: {NAVY};
    font-size: 11.5px; font-weight: 700; padding: 4px 10px; border-radius: 6px;
    margin: 2px 4px 2px 0;
}}
.caption-row {{ color: {MUTED}; font-size: 12px; margin-top: 6px; }}

/* ---------- Sidebar domain cards ---------- */
.domain-label {{
    font-size: 12.5px; font-weight: 800; letter-spacing: .4px;
    padding: 6px 10px; border-radius: 6px; color: white; margin-bottom: 6px;
}}

/* tighten default streamlit button look for example-question buttons */
.stButton button {{
    text-align: left; border-radius: 8px; border: 1px solid #E3E6EC;
    background: #FAFBFC; font-size: 12.8px; padding: 8px 10px; color: {INK};
}}
.stButton button:hover {{ border-color: {ACCENT}; color: {ACCENT}; }}

div[data-testid="stChatMessage"] {{ background: transparent; }}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_retriever():
    return SchemaRetriever()


tables = load_tables()
retriever = get_retriever()
catalogue = load_catalogue()
meta = load_meta()
min_date, max_date = data_window(tables)
TODAY_STR = meta["as_of_date"]

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None
if "doc_store" not in st.session_state:
    st.session_state.doc_store = DocumentStore()
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()
if "generating" not in st.session_state:
    st.session_state.generating = False
doc_store = st.session_state.doc_store

# ---------------------------------------------------------------------------
# Live Ollama status check (used by header + sidebar + input gating)
# ---------------------------------------------------------------------------
ollama_ok, ollama_msg, _ = llm_engine.check_ollama_alive()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
status_label = "Connected" if ollama_ok else "Offline"
st.markdown(f"""
<div class="mom-header">
  <div class="mom-eyebrow">Use Case 05 · All Brands · Operational Efficiency</div>
  <div class="mom-title">Internal Data Q&amp;A Copilot</div>
  <div class="mom-subtitle">Ask anything about claims, policy, Multiply or distribution data — or a document you upload — answered in seconds, in plain English.</div>
  <div class="mom-brandrow">MOMENTUM GROUP &nbsp;·&nbsp; METROPOLITAN &nbsp;·&nbsp; GUARDRISK &nbsp;·&nbsp; MULTIPLY</div>
</div>
""", unsafe_allow_html=True)

if not ollama_ok:
    st.markdown(f"""
    <div class="error-card">
        <b>⚠ Local LLM not reachable.</b> {ollama_msg}<br>
        Run <code>ollama serve</code>, then in another terminal <code>ollama pull qwen2.5:latest</code> and
        <code>ollama pull nomic-embed-text</code>, then refresh this page.
    </div>
    """, unsafe_allow_html=True)
    st.write("")

# ---------------------------------------------------------------------------
# KPI ribbon
# ---------------------------------------------------------------------------
# n_answered = sum(1 for m in st.session_state.messages if m["role"] == "assistant")
# elapsed_list = [m["trace"]["elapsed"] for m in st.session_state.messages
#                  if m["role"] == "assistant" and m.get("trace", {}).get("elapsed")]
# avg_time = f"{sum(elapsed_list)/len(elapsed_list):.1f}s" if elapsed_list else "—"

# k1, k2, k3, k4 = st.columns(4)
# kpis = [
#     (k1, "16", "Live data tables · 4 domains"),
#     (k2, avg_time, "Avg. response time this session"),
#     (k3, str(n_answered), "Questions answered this session"),
#     (k4, f"{min_date.date()} → {max_date.date()}", "Data window covered"),
# ]
# for col, val, label in kpis:
#     with col:
#         st.markdown(f"""<div class="kpi-card"><div class="kpi-value">{val}</div>
#         <div class="kpi-label">{label}</div></div>""", unsafe_allow_html=True)

# if doc_store.filenames:
#     st.markdown(
#         f"<div class='caption-row'>📎 {len(doc_store.filenames)} document(s) indexed this session: "
#         f"{', '.join(doc_store.filenames)}</div>",
#         unsafe_allow_html=True,
#     )

# st.write("")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    # st.markdown(f"<div style='font-weight:800; font-size:16px; color:{NAVY};'>⚙️ Console</div>", unsafe_allow_html=True)
    st.write("")

    # role = st.selectbox(
    #     "Signed in as",
    #     list(pipeline.ROLE_TABLE_ACCESS.keys()),
    #     index=0,
    #     help="Role-based access control — each role only sees the data domains it's permitted to query.",
    # )
    # language = st.selectbox("Answer language", ["English", "Afrikaans"], index=0)

    # st.divider()
    st.markdown(f"<div style='font-weight:800; font-size:13.5px; color:{NAVY};'>📎 Reference documents</div>", unsafe_allow_html=True)
    st.caption("Upload a PDF (policy wording, procedures, circulars) and ask about it in the same chat below.")
    uploaded_files = st.file_uploader(
        "Upload PDF(s)", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed"
    )
    if uploaded_files:
        for f in uploaded_files:
            file_key = f"{f.name}:{f.size}"
            if file_key not in st.session_state.processed_files:
                with st.spinner(f"Indexing {f.name}…"):
                    n_chunks = doc_store.add_pdf(f.read(), f.name)
                st.session_state.processed_files.add(file_key)
                if n_chunks:
                    st.success(f"{f.name} — {n_chunks} passages indexed", icon="✅")
                else:
                    st.warning(f"{f.name} — no extractable text found (likely a scanned PDF).", icon="⚠️")
    # if doc_store.filenames:
    #     st.caption(f"{'🟢' if doc_store.vector_ready else '🟡'} "
    #                 f"{'hybrid: BM25 + vector (RRF)' if doc_store.vector_ready else 'BM25 keyword only'} · "
    #                 f"{len(doc_store.filenames)} document(s) loaded")

    # st.divider()
    st.markdown(f"<div style='font-weight:800; font-size:13.5px; color:{NAVY};'>📚 Try a question</div>", unsafe_allow_html=True)
    # st.caption("Straight from the discovery deck's data layer — click to run.")

    domain_colors = {"Claims Data": TEAL, "Policy Data": PURPLE, "Multiply Data": ORANGE}
    for domain, meta in catalogue["domains"].items():
        if domain == "Distribution Data":
            continue
        color = domain_colors.get(domain, NAVY)
        st.markdown(
            f"<div class='domain-label' style='background:{color};'>{meta['icon']} {domain}</div>",
            unsafe_allow_html=True,
        )
        for q in meta["example_queries"]:
            if st.button(q, key=f"ex_{q}", use_container_width=True):
                st.session_state.pending_question = q
        st.write("")

    # st.divider()
    ce_ready = reranker.cross_encoder_ready()
    status_dot = "🟢" if ollama_ok else "🔴"


    # st.divider()
    if st.button("🗑️ Reset conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------------------
# Chat rendering helpers
# ---------------------------------------------------------------------------
ROUTE_LABEL = {
    "DATA": ("📊 Data", TEAL),
    "DOCUMENT": ("📄 Document", PURPLE),
    "BOTH": ("🔗 Data + Document", ORANGE),
}


def render_trace(trace: dict):
    with st.expander("Citation", expanded=False):
        route = trace.get("route") or "DATA"
        label, color = ROUTE_LABEL.get(route, ROUTE_LABEL["DATA"])
        st.markdown(f"<span class='trace-chip' style='background:{color};color:white;'>{label}</span>",
                    unsafe_allow_html=True)
        st.write("")

        if route in ("DATA", "BOTH"):
            c1, c2 = st.columns([1, 1])
            with c1:
                st.markdown("**Tables consulted**")
                if trace.get("retrieved_tables"):
                    chips = "".join(f"<span class='trace-chip'>{n}</span>" for n, _ in trace["retrieved_tables"])
                    st.markdown(chips, unsafe_allow_html=True)
                else:
                    st.caption("None")
            with c2:
                if trace.get("denied_tables"):
                    st.markdown("**Blocked by role access**")
                    chips = "".join(f"<span class='trace-chip' style='background:#FCE8E6;color:#7A2018;'>{n}</span>"
                                     for n, _ in trace["denied_tables"])
                    st.markdown(chips, unsafe_allow_html=True)
            if trace.get("code"):
                st.markdown("**Generated pandas code**")
                st.code(trace["code"], language="python")
            if trace.get("result_text"):
                st.markdown("**Raw computed result**")
                st.code(trace["result_text"], language="text")

        if route in ("DOCUMENT", "BOTH"):
            st.markdown("**Source passages**")
            if trace.get("doc_chunks"):
                for c in trace["doc_chunks"]:
                    st.markdown(
                        f"<span class='trace-chip'>{c['filename']} · p.{c['page']}</span>",
                        unsafe_allow_html=True,
                    )
                with st.popover("View excerpt text"):
                    for c in trace["doc_chunks"]:
                        st.caption(f"**{c['filename']}, p.{c['page']}**")
                        st.write(c["text"][:600] + ("…" if len(c["text"]) > 600 else ""))
            else:
                st.caption("None matched")

        st.caption(f"⏱ Answered in {trace.get('elapsed', 0):.2f}s · model: {llm_engine.GEN_MODEL}")


def render_assistant_body(content: str, trace: dict):
    if trace.get("error") and not trace.get("narrative"):
        css = "access-card" if trace.get("denied_tables") and not trace.get("code") else "error-card"
        st.markdown(f"<div class='{css}'>{trace['error']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='answer-card'>{content}</div>", unsafe_allow_html=True)
        if trace.get("chart_fig") is not None:
            st.plotly_chart(trace["chart_fig"], use_container_width=True, config={"displayModeBar": False})
    render_trace(trace)


# ---------------------------------------------------------------------------
# Input handling (captured early so history can react to it)
# ---------------------------------------------------------------------------
prompt = st.chat_input("Ask about the data, or about a document you've uploaded…", disabled=not ollama_ok)
if not prompt and st.session_state.pending_question:
    prompt = st.session_state.pending_question
    st.session_state.pending_question = None

# ---------------------------------------------------------------------------
# Chat history (skip last assistant reply if a new prompt is incoming)
# ---------------------------------------------------------------------------
messages_to_show = st.session_state.messages
if prompt:
    last_asst = next((i for i in range(len(messages_to_show) - 1, -1, -1)
                      if messages_to_show[i]["role"] == "assistant"), None)
    if last_asst is not None:
        messages_to_show = messages_to_show[:last_asst] + messages_to_show[last_asst + 1:]

for msg in messages_to_show:
    with st.chat_message(msg["role"], avatar="🧑‍💼" if msg["role"] == "user" else "📊"):
        if msg["role"] == "assistant" and msg.get("trace"):
            render_assistant_body(msg["content"], msg["trace"])
        else:
            st.markdown(msg["content"])


if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💼"):
        st.markdown(prompt)
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Generating Answer…"):
            trace = pipeline.answer_question(prompt, tables, retriever, "English", TODAY_STR, doc_store)
        content = trace.get("narrative") or trace.get("error") or "_No narrative generated._"
        render_assistant_body(content, trace)
    st.session_state.messages.append({"role": "assistant", "content": content, "trace": trace})
