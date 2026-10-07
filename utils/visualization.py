"""
Information Visualization Utilities
Reusable styling, themes, and interactive chart generators adhering to an
elegant, minimalist dark grey/black aesthetic without emojis.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, Any

# Palette constants
COLOR_BG_DARK = "#0d1117"
COLOR_BG_CARD = "#161b22"
COLOR_BORDER = "#282e38"
COLOR_GRID = "#21262d"
COLOR_TEXT = "#c9d1d9"
COLOR_TEXT_BRIGHT = "#f0f6fc"
COLOR_TEXT_MUTED = "#8b949e"

COLOR_LEGIT = "#388bfd"
COLOR_FRAUD = "#f85149"

COLOR_TABNET = "#58a6ff"
COLOR_MLP = "#e3b341"
COLOR_FT_TRANSFORMER = "#bc8cff"

MODEL_COLORS = {
    "TabNet": COLOR_TABNET,
    "MLP": COLOR_MLP,
    "FT-Transformer": COLOR_FT_TRANSFORMER,
}

CLASS_COLORS = {
    "Legitimate": COLOR_LEGIT,
    "Fraud": COLOR_FRAUD,
    0: COLOR_LEGIT,
    1: COLOR_FRAUD,
}


def apply_dark_theme(fig: go.Figure, height: int = 420, title: str = "") -> go.Figure:
    """Apply consistent dark-mode styling with grey grid lines, tooltip styling, and ample margins."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLOR_BG_CARD,
        plot_bgcolor=COLOR_BG_DARK,
        font=dict(
            family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
            color=COLOR_TEXT,
            size=12,
        ),
        margin=dict(l=55, r=45, t=60 if title else 35, b=55),
        height=height,
        hoverlabel=dict(
            bgcolor="#1f242c",
            bordercolor=COLOR_BORDER,
            font=dict(
                family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
                color=COLOR_TEXT_BRIGHT,
                size=12,
            ),
        ),
    )
    if title:
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(color=COLOR_TEXT_BRIGHT, size=14, weight=600),
                x=0.03,
                y=0.95,
            )
        )
    fig.update_xaxes(
        gridcolor=COLOR_GRID,
        zerolinecolor=COLOR_BORDER,
        linecolor=COLOR_BORDER,
        tickfont=dict(color=COLOR_TEXT_MUTED, size=11),
        title_font=dict(color=COLOR_TEXT, size=12),
    )
    fig.update_yaxes(
        gridcolor=COLOR_GRID,
        zerolinecolor=COLOR_BORDER,
        linecolor=COLOR_BORDER,
        tickfont=dict(color=COLOR_TEXT_MUTED, size=11),
        title_font=dict(color=COLOR_TEXT, size=12),
    )
    return fig
