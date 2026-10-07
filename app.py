"""
Explainable Credit Card Fraud Detection and Interactive Visualization
====================================================================
Main application entry point.
Clean, minimal, dark aesthetic.
"""

import streamlit as st

# ── Page configuration ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Explainable Credit Card Fraud Detection and Interactive Visualization",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS: Spacious, Clean, Dark & Grey Minimalist Theme ─────────────
st.markdown("""
<style>
    /* Global Base */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0d1117 !important;
        color: #c9d1d9 !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
        line-height: 1.65 !important;
    }

    /* Main Container Padding - Spacious & Breathable */
    .block-container {
        padding-top: 2.5rem !important;
        padding-bottom: 5rem !important;
        padding-left: 3.5rem !important;
        padding-right: 3.5rem !important;
        max-width: 1400px !important;
        margin: 0 auto !important;
    }

    /* Sidebar Background - Minimal, Clean, Dark Grey */
    [data-testid="stSidebar"] {
        background-color: #13171f !important;
        border-right: 1px solid #21262d !important;
        padding-top: 1.5rem !important;
    }
    [data-testid="stSidebar"] * {
        color: #c9d1d9 !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] {
        padding-top: 0.5rem !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        padding: 8px 12px !important;
        border-radius: 6px !important;
        transition: background 0.15s ease !important;
        font-size: 0.95rem !important;
        font-weight: 500 !important;
        cursor: pointer !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background-color: #1c212c !important;
    }

    /* App Header */
    .app-title {
        font-size: 2.1rem !important;
        font-weight: 700 !important;
        color: #f0f6fc !important;
        letter-spacing: -0.03em !important;
        margin-bottom: 2rem !important;
        padding-bottom: 1rem !important;
        border-bottom: 1px solid #21262d !important;
    }

    /* Section Spacing */
    h2 {
        font-size: 1.35rem !important;
        font-weight: 600 !important;
        color: #f0f6fc !important;
        margin-top: 2.5rem !important;
        margin-bottom: 1.25rem !important;
        letter-spacing: -0.02em !important;
    }
    h3 {
        font-size: 1.15rem !important;
        font-weight: 600 !important;
        color: #e6edf3 !important;
        margin-top: 2rem !important;
        margin-bottom: 1rem !important;
    }

    /* Metric Cards - Generous Internal Space */
    [data-testid="stMetric"] {
        background-color: #161b22 !important;
        border: 1px solid #282e38 !important;
        border-radius: 8px !important;
        padding: 20px 24px !important;
        min-height: 110px !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
        transition: border-color 0.2s ease !important;
    }
    [data-testid="stMetric"]:hover {
        border-color: #388bfd !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.8rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
        color: #8b949e !important;
        margin-bottom: 6px !important;
    }
    [data-testid="stMetricValue"] {
        color: #f0f6fc !important;
        font-size: 1.65rem !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em !important;
    }

    /* Visual Analytics Chart Containers */
    .stPlotlyChart {
        margin-top: 1rem !important;
        margin-bottom: 1.75rem !important;
        border-radius: 8px !important;
        overflow: hidden !important;
        border: 1px solid #21262d !important;
        background-color: #161b22 !important;
    }

    /* Dividers - Spacious Margin */
    hr {
        border: none !important;
        border-top: 1px solid #21262d !important;
        margin: 3rem 0 !important;
    }

    /* Tab navigation - Spacious & Clean */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #13171f !important;
        border-radius: 8px !important;
        padding: 6px !important;
        gap: 8px !important;
        border: 1px solid #21262d !important;
        margin-bottom: 1.5rem !important;
    }
    .stTabs [data-baseweb="tab"] {
        color: #8b949e !important;
        border-radius: 6px !important;
        padding: 8px 20px !important;
        font-weight: 500 !important;
        font-size: 0.92rem !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #21262d !important;
        color: #f0f6fc !important;
    }

    /* Expanders */
    [data-testid="stExpander"] {
        background-color: #161b22 !important;
        border: 1px solid #282e38 !important;
        border-radius: 8px !important;
        margin-top: 1.5rem !important;
        margin-bottom: 1.5rem !important;
    }

    /* Alert / Callout Boxes */
    [data-testid="stAlert"] {
        background-color: #161b22 !important;
        border: 1px solid #282e38 !important;
        border-radius: 8px !important;
        color: #c9d1d9 !important;
        padding: 18px 22px !important;
        margin-top: 1.5rem !important;
        margin-bottom: 1.5rem !important;
    }

    /* Status Badges */
    .status-badge-fraud {
        background-color: rgba(248, 81, 73, 0.12);
        color: #f85149;
        border: 1px solid rgba(248, 81, 73, 0.4);
        padding: 4px 12px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        display: inline-block;
    }
    .status-badge-legit {
        background-color: rgba(56, 139, 253, 0.12);
        color: #58a6ff;
        border: 1px solid rgba(56, 139, 253, 0.4);
        padding: 4px 12px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# ── Import view modules from views/ ───────────────────────────────────────
from views import data_exploration, model_analytics, transaction_investigation, explainability

# ── Minimal, Professional Sidebar in Title Case ───────────────────────────
st.sidebar.markdown(
    """
    <div style="padding-bottom: 1rem; border-bottom: 1px solid #21262d; margin-bottom: 1.5rem;">
        <div style="font-size: 1.15rem; font-weight: 600; color: #f0f6fc; letter-spacing: -0.01em;">Interactive Visualization</div>
    </div>
    """,
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Data Exploration",
        "Model Analytics",
        "Transaction Investigation",
        "Explainability",
    ],
    label_visibility="collapsed",
    help="Select a view to explore data distributions, compare models, investigate individual transactions, or inspect SHAP and LIME explanations.",
)

# ── Global App Header ─────────────────────────────────────────────────────
st.markdown(
    '<div class="app-title">Explainable Credit Card Fraud Detection and Interactive Visualization</div>',
    unsafe_allow_html=True,
)

# ── Routing ───────────────────────────────────────────────────────────────
if page == "Data Exploration":
    data_exploration.show()
elif page == "Model Analytics":
    model_analytics.show()
elif page == "Transaction Investigation":
    transaction_investigation.show()
elif page == "Explainability":
    explainability.show()
