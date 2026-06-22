"""
All calls to the local Ollama server live here. Nothing in this module ever
leaves the machine it runs on — this is the 'zero data leaves the firewall'
promise from the UC05 slide.

Two LLM steps, matching the architecture diagram exactly:
  Step 1 (generate_code)      NL question -> pandas code (LLM sees schema only)
  Step 3 (synthesize_answer)  execution result -> plain-English narrative
"""

import json
import os
import re

import requests

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
GEN_MODEL = os.environ.get("MOMENTUM_LLM_MODEL", "qwen2.5:latest")
TIMEOUT = 120


class OllamaError(RuntimeError):
    """Raised when the local Ollama server can't be reached or errors out."""


def _chat(messages, temperature=0.1, model=GEN_MODEL):
    try:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": temperature},
            },
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]
    except requests.exceptions.ConnectionError as e:
        raise OllamaError(
            f"Can't reach Ollama at {OLLAMA_HOST}. Start it with `ollama serve` "
            f"and make sure `{model}` is pulled."
        ) from e
    except requests.exceptions.RequestException as e:
        raise OllamaError(f"Ollama request failed: {e}") from e


def ollama_embed(text: str, model: str = "nomic-embed-text") -> list:
    resp = requests.post(
        f"{OLLAMA_HOST}/api/embeddings",
        json={"model": model, "prompt": text},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


# ---------------------------------------------------------------------------
# Step 1 — NL question -> pandas code
# ---------------------------------------------------------------------------
CODE_SYSTEM_PROMPT = """You are the query-planning engine inside Momentum Africa's internal \
Data Q&A Copilot. You translate a manager's plain-English (or Afrikaans) question into a \
single pandas script that runs against dataframes already loaded in memory.

RULES — follow exactly:
1. Only the dataframes listed below exist. They are accessed as Python variables with \
exactly the table name shown (e.g. `policy_master`, `lapse_events`). Do not invent columns.
2. Write plain pandas. Do NOT use import, open(), exec(), eval(), or any file/network access.
3. Assign your final answer to a variable called `result` (a number, Series, or DataFrame).
4. If the question implies a breakdown/comparison (by channel, by market, by month, by cell, \
etc.) ALSO assign a tidy DataFrame called `chart_df` with exactly two columns: a category \
column first, a numeric value column second, sorted descending by value. If the question is a \
single number, skip chart_df.
5. To compute a RATE for a segment (e.g. lapse rate, conversion rate), the denominator is the \
relevant population table (e.g. policy_master) and the numerator is the event table (e.g. \
lapse_events) — merge/join on the shared id, do not just count rows in the event table alone.
5b. If a column (e.g. `channel`, `market`, `product_type`) does not exist on the table you are \
aggregating, JOIN that table to the table that owns the column (e.g. policy_master) before \
grouping. Never assume a column exists on a table unless it appears in the schema below.
6. "Today" is {today}. Compute relative windows ("last 90 days", "last quarter", "Q1 2025") \
using pandas date arithmetic from that anchor — never hardcode a different current date.
7. Round rates/percentages to 1 decimal place where sensible.
8. Output ONLY a single ```python code fence. No prose before or after it.

AVAILABLE TABLES (only the relevant ones are shown for this question):
{schema}
"""

CODE_FEW_SHOT = [
    {
        "role": "user",
        "content": "What was the lapse rate for funeral products in Botswana Q1 by channel?",
    },
    {
        "role": "assistant",
        "content": (
            "```python\n"
            "pol = policy_master[(policy_master['market'] == 'Botswana') & "
            "(policy_master['product_type'] == 'Funeral')]\n"
            "lapsed = lapse_events[lapse_events['policy_id'].isin(pol['policy_id'])]\n"
            "rate = (lapsed.groupby('channel')['policy_id'].count() / "
            "pol.groupby('channel')['policy_id'].count() * 100).round(1)\n"
            "rate = rate.sort_values(ascending=False)\n"
            "result = rate\n"
            "chart_df = rate.reset_index()\n"
            "chart_df.columns = ['channel', 'lapse_rate_pct']\n"
            "```"
        ),
    },
    {
        "role": "user",
        "content": "Which distribution channels have the highest 13-month persistency?",
    },
    {
        "role": "assistant",
        "content": (
            "```python\n"
            "# 13-month persistency: policies that paid in all 13 of their first months\n"
            "# premium_history has no channel column — must join through policy_master\n"
            "ph = premium_history.merge(policy_master[['policy_id', 'channel', 'inception_date']], on='policy_id')\n"
            "ph['months_since_inception'] = (\n"
            "    (ph['period'].dt.year - ph['inception_date'].dt.year) * 12\n"
            "    + (ph['period'].dt.month - ph['inception_date'].dt.month)\n"
            ")\n"
            "ph13 = ph[ph['months_since_inception'] <= 12]\n"
            "paid = ph13.groupby('policy_id')['on_time_flag'].sum()\n"
            "months = ph13.groupby('policy_id')['on_time_flag'].count()\n"
            "persisted = (paid == months) & (months == 13)\n"
            "ph13 = ph13.join(persisted.rename('persisted'), on='policy_id')\n"
            "rate = (ph13.groupby('channel')['persisted'].mean() * 100).round(1)\n"
            "rate = rate.sort_values(ascending=False)\n"
            "result = rate\n"
            "chart_df = rate.reset_index()\n"
            "chart_df.columns = ['channel', 'persistency_pct']\n"
            "```"
        ),
    },
    {
        "role": "user",
        "content": "Which Guardrisk cells had the highest fraud rate last quarter?",
    },
    {
        "role": "assistant",
        "content": (
            "```python\n"
            "cutoff = pd.Timestamp('{today}') - pd.DateOffset(months=3)\n"
            "recent_claims = claims_master[claims_master['submission_date'] >= cutoff]\n"
            "merged = recent_claims.merge(fraud_flags, on='claim_id')\n"
            "merged['is_red'] = (merged['final_tier'] == 'Red').astype(int)\n"
            "rate = (merged.groupby('cell')['is_red'].mean() * 100).round(1)\n"
            "rate = rate.sort_values(ascending=False)\n"
            "result = rate\n"
            "chart_df = rate.reset_index()\n"
            "chart_df.columns = ['cell', 'fraud_rate_pct']\n"
            "```"
        ),
    },
]


def _extract_code(raw: str) -> str:
    match = re.search(r"```(?:python)?\s*(.*?)```", raw, re.DOTALL)
    code = match.group(1).strip() if match else raw.strip()
    return code


def generate_code(question: str, schema_text: str, today: str) -> str:
    system = CODE_SYSTEM_PROMPT.format(today=today, schema=schema_text)
    few_shot = [
        {**m, "content": m["content"].replace("{today}", today)} for m in CODE_FEW_SHOT
    ]
    messages = [{"role": "system", "content": system}, *few_shot,
                {"role": "user", "content": question}]
    raw = _chat(messages, temperature=0.1)
    return _extract_code(raw)


# ---------------------------------------------------------------------------
# Step 3 — execution result -> narrative
# ---------------------------------------------------------------------------
NARRATIVE_SYSTEM_PROMPT = """You are the Momentum Africa Data Q&A Copilot speaking directly \
to a business manager. You are given the question they asked and the computed result. Write a \
short, confident, plain-English answer (2-4 sentences max).

RULES:
- State the headline number(s) first.
- If the result is a breakdown, name the highest and lowest segment and call out anything \
notably above/below the others — that is usually the actual insight.
- Never mention pandas, code, dataframes, or how the number was computed.
- Never invent numbers that are not present in the result given to you.
- Respond in {language}.
- No headings, no bullet points, just plain sentences.
"""


def synthesize_answer(question: str, result_repr: str, language: str = "English") -> str:
    system = NARRATIVE_SYSTEM_PROMPT.format(language=language)
    user = f"Question: {question}\n\nComputed result:\n{result_repr}"
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return _chat(messages, temperature=0.3).strip()


# ---------------------------------------------------------------------------
# Router — decides DATA vs DOCUMENT vs BOTH so the same chat box can answer
# either kind of question without the user having to pick a mode/tab.
# ---------------------------------------------------------------------------
ROUTER_SYSTEM_PROMPT = """You route questions for Momentum Africa's internal copilot, which has two \
possible sources of truth:

- Live STRUCTURED DATA tables (claims, policy, lapse, fraud, Multiply, advisors, leads, conversion — \
numbers and trends computed on the fly).
- Uploaded DOCUMENTS (PDFs the user has attached this session — policy wording, procedures, circulars).

Reply with EXACTLY one word: DATA, DOCUMENT, or BOTH.
- DATA: the question asks for a number, rate, trend, ranking or comparison computed from tables.
- DOCUMENT: the question asks about wording, definitions, rules or content that would live in a \
document, or explicitly references "the document" / "the PDF" / "the policy wording".
- BOTH: the question explicitly connects or compares something in a document to the live data.

Available data domains this session: {domains}
Documents uploaded this session: {documents}
"""


def classify_intent(question: str, domains: list, documents: list) -> str:
    if not documents:
        return "DATA"
    system = ROUTER_SYSTEM_PROMPT.format(
        domains=", ".join(domains) or "none", documents=", ".join(documents)
    )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": question}]
    try:
        raw = _chat(messages, temperature=0.0).strip().upper()
    except OllamaError:
        return "DATA"
    for label in ("BOTH", "DOCUMENT", "DATA"):
        if label in raw:
            return label
    return "DATA"


# ---------------------------------------------------------------------------
# Document-grounded answer synthesis
# ---------------------------------------------------------------------------
DOCUMENT_SYSTEM_PROMPT = """You are the Momentum Africa Data Q&A Copilot, answering from excerpts of \
documents the user uploaded this session. Use ONLY the excerpts given — never invent content. If the \
excerpts don't answer the question, say so plainly. Do NOT include inline citations or source references \
in your answer — citations will be shown separately. Respond in {language}. 2-5 sentences, plain prose, \
no headings or bullet points.
"""

COMBINED_SYSTEM_PROMPT = """You are the Momentum Africa Data Q&A Copilot. You are given a computed \
result from live data AND excerpts from a document the user uploaded this session. Connect the two \
directly — e.g. does the document's stated rule/figure match what the live data shows. Use ONLY the \
information given, never invent numbers. Do NOT include inline citations or source references in your \
answer — citations will be shown separately. Respond in {language}. 2-5 sentences, plain prose, no headings.
"""


def _format_excerpts(chunks: list) -> str:
    return "\n\n".join(
        f"[{i+1}] ({c['filename']}, p.{c['page']}): {c['text']}" for i, c in enumerate(chunks)
    )


def synthesize_document_answer(question: str, chunks: list, language: str = "English") -> str:
    system = DOCUMENT_SYSTEM_PROMPT.format(language=language)
    user = f"Question: {question}\n\nExcerpts:\n{_format_excerpts(chunks)}"
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return _chat(messages, temperature=0.2).strip()


def synthesize_combined_answer(question: str, result_repr: str, chunks: list, language: str = "English") -> str:
    system = COMBINED_SYSTEM_PROMPT.format(language=language)
    user = (
        f"Question: {question}\n\nComputed data result:\n{result_repr}\n\n"
        f"Document excerpts:\n{_format_excerpts(chunks)}"
    )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return _chat(messages, temperature=0.2).strip()


def check_ollama_alive() -> tuple:
    """Returns (ok: bool, message: str, available_models: list)."""
    try:
        resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        gen_ok = any(GEN_MODEL.split(":")[0] in m for m in models)
        if not gen_ok:
            return False, f"Ollama is running but `{GEN_MODEL}` is not pulled yet.", models
        return True, "Connected", models
    except requests.exceptions.RequestException:
        return False, f"Can't reach Ollama at {OLLAMA_HOST}.", []
