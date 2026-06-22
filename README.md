MOMENTUM AFRICA — INTERNAL DATA Q&A COPILOT (UC05)
====================================================
Natural-language Q&A over claims, policy, Multiply and distribution data.
100% local: Ollama (qwen2.5) + ChromaDB + Streamlit. Nothing leaves the
machine it runs on.


WHAT THIS IS
------------
A working build of Use Case 05 from the discovery deck: a manager types a
question in plain English (or Afrikaans), and gets a narrative answer with
a chart in seconds — backed by live pandas computation, not a canned
response.

Pipeline (matches the deck's architecture slide exactly):
  1. Ask      -> question typed in English or Afrikaans
  2. Plan     -> qwen2.5 reads table metadata only (never row data) and
                 writes pandas code, grounded by tables retrieved from
                 ChromaDB
  3. Execute  -> code runs in a sandboxed, read-only, timeout-protected
                 environment against in-memory dataframes
  4. Answer   -> qwen2.5 turns the result into a plain-English narrative;
                 the app auto-builds a matching bar/line chart


REQUIREMENTS
------------
- Python 3.10+
- Ollama installed and running (https://ollama.com)
- ~4GB free RAM for qwen2.5:latest


SETUP
-----
    cd momentum_qna_copilot
    ollama pull qwen2.5:latest
    ollama pull nomic-embed-text
    bash setup.sh

setup.sh does all of the following for you:
    pip install -r requirements.txt
    python3 data/generate_data.py
    streamlit run app.py --server.port 8505

Or run each step manually if you prefer. The app opens at
http://localhost:8501


FOLDER STRUCTURE
-----------------
app.py                          Streamlit dashboard (entry point)
core/
  data_loader.py                 Loads the 16 CSVs into cached dataframes
  retriever.py                   ChromaDB schema retrieval (+ keyword
                                  fallback if Ollama embeddings unavailable)
  llm_engine.py                  All Ollama calls: code generation +
                                  narrative synthesis
  sandbox.py                     Safe pandas execution (AST check,
                                  restricted builtins, timeout)
  chart_engine.py                Auto-builds Momentum-branded Plotly charts
  pipeline.py                    Orchestrates steps 1-4 + role-based access
data/
  generate_data.py               Synthetic data generator — produces all
                                  16 tables fresh, anchored to "today"
  *.csv                          Generated tables (gitignore-able, can be
                                  regenerated any time)
  meta.json                      Canonical "as of" date used by the LLM
metadata/
  schema_catalogue.json          Table/column descriptions fed to the LLM
                                  and indexed into ChromaDB
.streamlit/config.toml          Streamlit theme (Momentum brand colors)
requirements.txt
setup.sh


DATA MODEL (16 tables, 4 domains — matches the deck's data layer slide)
-------------------------------------------------------------------------
Claims Data       claims_master, claims_by_cell, fraud_flags, triage_outcomes
Policy Data       policy_master, lapse_events, new_business, premium_history
Multiply Data     multiply_engagement, active_dayz, tier_history, reward_redemptions
Distribution Data advisor_performance, lead_pipeline, conversion_rates, territory_data

All synthetic, regenerated fresh each time you run generate_data.py,
anchored to the current date so "last 90 days" / "last quarter" always
resolve correctly. A few storylines are deliberately seeded so the deck's
headline example questions return a clean, presentable answer (e.g.
Botswana funeral lapse is highest on the Direct Digital channel; Motor
Cell has an elevated fraud rate).


ROLE-BASED ACCESS (sidebar "Signed in as")
-------------------------------------------
Executive (CEO / President)   all domains
Actuarial & Risk              Claims + Policy
Claims Manager                Claims only
Distribution Manager          Distribution + Policy
Marketing / Multiply          Multiply + Policy

Switch roles and ask a question outside that role's access to see the
governance behaviour live — it explains why and what's blocked rather
than just failing.


TROUBLESHOOTING
----------------
"Local LLM not reachable"
    Run `ollama serve` in a terminal, confirm `ollama pull qwen2.5:latest`
    completed, then refresh the page.

Sidebar shows "keyword fallback" instead of "vector search"
    Run `ollama pull nomic-embed-text`. The app still works either way —
    keyword search is just a slightly less precise fallback so a live
    demo never hard-fails.

Want different / larger data
    Edit the constants at the top of data/generate_data.py (N_POLICIES,
    N_CLAIMS, N_LEADS, markets, products, channels) and re-run it.

Reset the conversation
    Use the "Reset conversation" button in the sidebar.


SECURITY NOTES
---------------
This sandbox blocks imports, dunder access, and known dangerous builtins
via a static AST check before any code runs, and enforces a 15s timeout.
It is appropriate for a trusted internal tool running LLM-generated code
against your own data, not a claim of adversarial-proof isolation — a
production rollout would run step 3 in a dedicated isolated process or
container, as noted on the deck's security architecture slide.