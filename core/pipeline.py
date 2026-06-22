"""Orchestrates the unified copilot. Every question comes through the same
chat box; this module decides per-question whether the answer should come
from the live data tables, an uploaded document, or both — and produces a
step-by-step trace so the UI can show exactly how the answer was produced."""

import time

from core import llm_engine, sandbox, chart_engine

# All domains are accessible (no role-based filtering)
ALL_DOMAINS = {"Claims Data", "Policy Data", "Multiply Data", "Distribution Data"}


def _new_trace(question):
    return {
        "question": question,
        "route": None,
        "retrieved_tables": [],
        "denied_tables": [],
        "code": None,
        "result_text": None,
        "chart_fig": None,
        "doc_chunks": [],
        "narrative": None,
        "error": None,
        "elapsed": 0.0,
    }


def _run_data_subpipeline(question: str, tables: dict, retriever, today: str, trace: dict) -> bool:
    """Fills the data-related fields of `trace`. Returns True on success."""
    retrieved = retriever.retrieve(question, top_k=4)
    allowed = [(name, schema_text, retriever.tables_meta[name]["domain"]) for name, schema_text in retrieved]

    trace["retrieved_tables"] = [(n, d) for n, _, d in allowed]
    trace["denied_tables"] = []

    if not allowed:
        trace["error"] = "No relevant tables found for this question."
        return False

    schema_text = "\n\n".join(s for _, s, _ in allowed)
    try:
        code = llm_engine.generate_code(question, schema_text, today)
    except llm_engine.OllamaError as e:
        trace["error"] = str(e)
        return False
    trace["code"] = code

    outcome = sandbox.run_pandas_code(code, tables)
    if outcome["error"]:
        trace["error"] = outcome["error"]
        return False

    trace["result_text"] = sandbox.result_to_text(outcome["result"])
    trace["chart_fig"] = chart_engine.build_chart(outcome["chart_df"])
    return True


def answer_question(question: str, tables: dict, retriever, language: str,
                     today: str, doc_store=None) -> dict:
    trace = _new_trace(question)
    t0 = time.time()

    doc_names = doc_store.filenames if doc_store else []
    domains = sorted(ALL_DOMAINS)
    route = llm_engine.classify_intent(question, domains, doc_names) if doc_names else "DATA"
    trace["route"] = route

    # ---- DOCUMENT ONLY --------------------------------------------------------
    if route == "DOCUMENT":
        chunks = doc_store.retrieve(question)
        trace["doc_chunks"] = chunks
        if not chunks:
            trace["error"] = ("None of the uploaded documents matched this question. "
                               "Try rephrasing, or upload a relevant PDF.")
            trace["elapsed"] = time.time() - t0
            return trace
        try:
            trace["narrative"] = llm_engine.synthesize_document_answer(question, chunks, language)
        except llm_engine.OllamaError as e:
            trace["error"] = str(e)
        trace["elapsed"] = time.time() - t0
        return trace

    # ---- BOTH -------------------------------------------------------------------
    if route == "BOTH":
        data_ok = _run_data_subpipeline(question, tables, retriever, today, trace)
        chunks = doc_store.retrieve(question) if doc_store else []
        trace["doc_chunks"] = chunks
        if not data_ok and not chunks:
            trace["elapsed"] = time.time() - t0
            return trace
        result_for_llm = trace["result_text"] or "No matching data result was found."
        try:
            trace["narrative"] = llm_engine.synthesize_combined_answer(question, result_for_llm, chunks, language)
            trace["error"] = None
        except llm_engine.OllamaError as e:
            trace["error"] = str(e)
        trace["elapsed"] = time.time() - t0
        return trace

    # ---- DATA ONLY (default) -----------------------------------------------------
    if not _run_data_subpipeline(question, tables, retriever, today, trace):
        trace["elapsed"] = time.time() - t0
        return trace

    try:
        trace["narrative"] = llm_engine.synthesize_answer(question, trace["result_text"], language)
    except llm_engine.OllamaError as e:
        trace["error"] = str(e)

    trace["elapsed"] = time.time() - t0
    return trace
