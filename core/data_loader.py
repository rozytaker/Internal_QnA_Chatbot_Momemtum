"""Loads the 16 synthetic tables into memory as read-only pandas DataFrames."""

import json
from pathlib import Path
import pandas as pd
import streamlit as st

DATA_DIR = Path(__file__).parent.parent / "data"

DATE_COLUMNS = {
    "claims_master": ["loss_date", "submission_date"],
    "claims_by_cell": ["month"],
    "fraud_flags": ["flagged_date"],
    "policy_master": ["inception_date"],
    "lapse_events": ["lapse_date"],
    "new_business": ["issue_date"],
    "premium_history": ["period"],
    "multiply_engagement": ["month"],
    "active_dayz": ["month"],
    "tier_history": ["change_date"],
    "reward_redemptions": ["redemption_date"],
    "advisor_performance": ["month"],
    "lead_pipeline": ["created_date", "last_update"],
    "conversion_rates": ["month"],
}

TABLE_NAMES = [
    "claims_master", "claims_by_cell", "fraud_flags", "triage_outcomes",
    "policy_master", "lapse_events", "new_business", "premium_history",
    "multiply_engagement", "active_dayz", "tier_history", "reward_redemptions",
    "advisor_performance", "lead_pipeline", "conversion_rates", "territory_data",
]


@st.cache_resource(show_spinner=False)
def load_tables() -> dict:
    """Load every CSV once per app session. Returned frames are treated as
    read-only by the sandbox executor — nothing in this app ever writes
    back to disk."""
    tables = {}
    for name in TABLE_NAMES:
        path = DATA_DIR / f"{name}.csv"
        df = pd.read_csv(path)
        for col in DATE_COLUMNS.get(name, []):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        df.attrs["table_name"] = name
        tables[name] = df
    return tables


def data_window(tables: dict) -> tuple:
    """Return (min_date, max_date) across all date-bearing tables, used to
    tell the LLM what 'today' / 'last quarter' / 'last 90 days' means."""
    all_dates = []
    for name, cols in DATE_COLUMNS.items():
        df = tables.get(name)
        if df is None:
            continue
        for col in cols:
            if col in df.columns:
                all_dates.append(df[col].min())
                all_dates.append(df[col].max())
    return min(all_dates), max(all_dates)


@st.cache_resource(show_spinner=False)
def load_meta() -> dict:
    """Canonical 'as of' date written by the generator — the single source
    of truth for what 'today' means to the LLM, independent of any one
    table's particular max date."""
    path = DATA_DIR / "meta.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"as_of_date": pd.Timestamp.now().date().isoformat(), "window_start": None}
