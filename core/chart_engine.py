"""Turns the `chart_df` produced by the sandbox into a Momentum-branded
Plotly chart. Picks bar vs. line automatically based on the category column."""

import pandas as pd
import plotly.graph_objects as go

NAVY = "#0B1F3D"
ACCENT = "#C0392B"
PALETTE = ["#1B6F7E", "#C0392B", "#6B3FA0", "#D2691E", "#1B7F5C", "#1B3A6B"]
GRID = "#E7E9EE"
TEXT = "#3A4255"


def _is_time_like(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    try:
        pd.to_datetime(series.astype(str), errors="raise")
        return True
    except Exception:
        return False


def build_chart(chart_df: pd.DataFrame):
    if chart_df is None or chart_df.empty or chart_df.shape[1] < 2:
        return None

    cat_col, val_col = chart_df.columns[0], chart_df.columns[1]
    df = chart_df.copy()

    fig = go.Figure()

    if _is_time_like(df[cat_col]) and len(df) > 2:
        df[cat_col] = pd.to_datetime(df[cat_col])
        df = df.sort_values(cat_col)
        fig.add_trace(go.Scatter(
            x=df[cat_col], y=df[val_col],
            mode="lines+markers",
            line=dict(color=NAVY, width=3),
            marker=dict(size=7, color=ACCENT),
            fill="tozeroy", fillcolor="rgba(11,31,61,0.06)",
        ))
    else:
        df = df.sort_values(val_col, ascending=True)
        colors = [PALETTE[i % len(PALETTE)] for i in range(len(df))]
        # highlight the top (last, since ascending) bar in the brand accent
        if colors:
            colors[-1] = ACCENT
        fig.add_trace(go.Bar(
            x=df[val_col], y=df[cat_col].astype(str),
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=df[val_col].round(1).astype(str),
            textposition="outside",
        ))

    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Helvetica, Arial, sans-serif", color=TEXT, size=13),
        margin=dict(l=10, r=30, t=10, b=10),
        height=max(220, 46 * len(df) + 60) if not _is_time_like(chart_df[cat_col]) else 320,
        showlegend=False,
        xaxis=dict(showgrid=True, gridcolor=GRID, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
    )
    return fig
