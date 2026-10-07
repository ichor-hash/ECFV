"""
Page 1: Data Exploration
Information Visualization and exploratory visual analytics of credit card transactions.
Dark minimalist aesthetic with spacious layout and comprehensive tooltips.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from models.model_loader import load_test_data, load_feature_names
from utils.visualization import (
    apply_dark_theme,
    COLOR_LEGIT,
    COLOR_FRAUD,
    COLOR_GRID,
    COLOR_BORDER,
    COLOR_BG_DARK,
    COLOR_BG_CARD,
    COLOR_TEXT,
    COLOR_TEXT_BRIGHT,
    COLOR_TEXT_MUTED,
)


def show():
    st.subheader("Data Exploration & Benchmark Visual Analytics")
    st.markdown(
        "Multidimensional analysis of transaction distributions, class imbalance, "
        "and feature separability across the Credit Card Fraud Detection benchmark."
    )

    try:
        df = load_test_data()
        feature_names = load_feature_names()
    except Exception as e:
        st.error(f"Error loading test dataset: {str(e)}")
        return

    if "Class_Label" not in df.columns:
        df["Class_Label"] = df["Actual_Class"].map({0: "Legitimate", 1: "Fraud"})

    color_map = {"Legitimate": COLOR_LEGIT, "Fraud": COLOR_FRAUD}

    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────────────────────
    # Section 1: Dataset Partitioning & Benchmark Scope
    # ──────────────────────────────────────────────────────────────────────────
    st.markdown("### 1. Dataset Overview & Imbalance Summary")

    scope = st.radio(
        "Select Data View (Observation Scope):",
        ["Entire Dataset (All 112,534 Transactions)", "Final Test Set (16,881 Evaluated Transactions)"],
        horizontal=True,
        help="Observation Scope simply asks: 'Which slice of the data are you looking at right now?' Toggle between the Entire Dataset (all 112,534 transactions in the project) and the Final Test Set (the separate 15% portion used to grade the models).",
    )

    st.markdown("<div style='margin-top: 0.75rem;'></div>", unsafe_allow_html=True)

    if "Entire Dataset" in scope:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "Total Transactions",
            "112,534",
            help="Total transactions in the full dataset after cleaning bad or duplicate rows.",
        )
        c2.metric(
            "Legitimate Cases",
            "112,298 (99.79%)",
            help="Normal card transactions. Represents 99.79% of all activity.",
        )
        c3.metric(
            "Fraud Cases",
            "236 (0.21%)",
            help="Confirmed fraudulent transactions. Extremely rare (only 0.21% of total).",
        )
        c4.metric(
            "Imbalance Ratio",
            "1 : 476",
            help="Severe class imbalance: for every 1 fraud transaction, there are 476 legitimate ones.",
        )
        st.caption(
            "Divided using stratified sampling (Random Seed 42): 70% was used to train the models (78,773 rows), "
            "15% was used to tune decision thresholds (16,880 rows), and 15% was locked away to test them (16,881 rows)."
        )
    else:
        test_total = len(df)
        test_fraud = int(df["Actual_Class"].sum())
        test_legit = test_total - test_fraud
        fraud_pct = (test_fraud / test_total) * 100

        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "Test Transactions",
            f"{test_total:,}",
            help="The 15% held-out test partition used to evaluate model accuracy on unseen data.",
        )
        c2.metric(
            "Legitimate Cases",
            f"{test_legit:,} ({100 - fraud_pct:.2f}%)",
            help="Actual legitimate transactions in the test partition.",
        )
        c3.metric(
            "Fraud Cases",
            f"{test_fraud:,} ({fraud_pct:.2f}%)",
            help="Actual fraudulent transactions in the test partition.",
        )
        c4.metric(
            "Positive Loss Weight",
            "476.41x",
            help="Training penalty multiplier: forces neural networks to care 476 times more about catching rare fraud instead of ignoring it.",
        )
        st.caption(
            "These 16,881 transactions were kept completely separate from the models during training. "
            "They serve as an unseen final exam to fairly grade how well the models detect real-world fraud."
        )



    st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
    st.markdown("---")

    # ──────────────────────────────────────────────────────────────────────────
    # Section 2: Multidimensional Parallel Coordinates
    # ──────────────────────────────────────────────────────────────────────────
    st.markdown("### 2. Multidimensional Parallel Coordinates")
    st.markdown(
        "Parallel coordinates mapping high-dimensional feature profiles across classes. "
        "Drag vertically along any axis to brush and isolate multivariate subsets."
    )

    top_parallel_feats = ["V14", "V12", "V10", "V4", "V11", "Amount"]

    fraud_sub = df[df["Actual_Class"] == 1]
    legit_sub = df[df["Actual_Class"] == 0].sample(n=600, random_state=42)
    parallel_df = pd.concat([fraud_sub, legit_sub]).sample(frac=1, random_state=42)

    dimensions = [
        dict(
            range=[float(parallel_df[col].min()), float(parallel_df[col].max())],
            label=col,
            values=parallel_df[col],
        )
        for col in top_parallel_feats
    ]

    fig_parallel = go.Figure(
        data=go.Parcoords(
            line=dict(
                color=parallel_df["Actual_Class"],
                colorscale=[[0, COLOR_LEGIT], [1, COLOR_FRAUD]],
                showscale=False,
            ),
            dimensions=dimensions,
        )
    )
    fig_parallel.update_layout(
        paper_bgcolor=COLOR_BG_CARD,
        plot_bgcolor=COLOR_BG_DARK,
        font=dict(
            family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
            color=COLOR_TEXT,
            size=11,
        ),
        height=400,
        margin=dict(l=60, r=60, t=55, b=35),
    )
    st.plotly_chart(fig_parallel, use_container_width=True)
    st.caption(
        "Parallel coordinates profile showing all 36 test fraud transactions (crimson) against 600 sampled legitimate cases (blue). "
        "Notice the coordinated negative plunge on V14, V12, and V10 accompanied by positive spikes on V4 and V11 in fraud attacks."
    )

    st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
    st.markdown("---")

    # ──────────────────────────────────────────────────────────────────────────
    # Section 3: Interactive Subspace Projections
    # ──────────────────────────────────────────────────────────────────────────
    st.markdown("### 3. Latent Subspace Projections & Separability")

    tab_3d, tab_2d, tab_corr = st.tabs(
        ["3D Subspace Projection", "2D Bivariate Density & Separation", "Feature Correlation Matrix"]
    )

    # Subsampled points for interactive responsiveness
    scatter_sample = pd.concat([
        df[df["Actual_Class"] == 1],
        df[df["Actual_Class"] == 0].sample(n=1200, random_state=42)
    ])

    with tab_3d:
        st.markdown("Explore multivariate clustering of fraud instances in arbitrary 3-feature subspaces.")
        c_x, c_y, c_z = st.columns(3)
        with c_x:
            feat_x = st.selectbox(
                "X-Axis Feature:",
                feature_names,
                index=feature_names.index("V14"),
                help="Select the feature to project along the horizontal X-axis.",
            )
        with c_y:
            feat_y = st.selectbox(
                "Y-Axis Feature:",
                feature_names,
                index=feature_names.index("V4"),
                help="Select the feature to project along the depth Y-axis.",
            )
        with c_z:
            feat_z = st.selectbox(
                "Z-Axis Feature:",
                feature_names,
                index=feature_names.index("V12"),
                help="Select the feature to project along the vertical Z-axis.",
            )

        fig_3d = px.scatter_3d(
            scatter_sample,
            x=feat_x,
            y=feat_y,
            z=feat_z,
            color="Class_Label",
            color_discrete_map=color_map,
            opacity=0.8,
            size=scatter_sample["Actual_Class"].map({0: 3, 1: 7}),
            labels={"Class_Label": "Class"},
        )
        fig_3d.update_layout(
            template="plotly_dark",
            paper_bgcolor=COLOR_BG_CARD,
            scene=dict(
                xaxis=dict(backgroundcolor=COLOR_BG_DARK, gridcolor=COLOR_GRID, linecolor=COLOR_BORDER),
                yaxis=dict(backgroundcolor=COLOR_BG_DARK, gridcolor=COLOR_GRID, linecolor=COLOR_BORDER),
                zaxis=dict(backgroundcolor=COLOR_BG_DARK, gridcolor=COLOR_GRID, linecolor=COLOR_BORDER),
            ),
            height=520,
            margin=dict(l=15, r=15, t=15, b=15),
            legend=dict(yanchor="top", y=0.95, xanchor="left", x=0.05),
        )
        fig_3d.update_traces(
            hovertemplate="<b>%{data.name}</b><br>" + f"{feat_x}: %{{x:.3f}}<br>{feat_y}: %{{y:.3f}}<br>{feat_z}: %{{z:.3f}}<extra></extra>"
        )
        st.plotly_chart(fig_3d, use_container_width=True)

    with tab_2d:
        c_2dx, c_2dy = st.columns(2)
        with c_2dx:
            feat_2dx = st.selectbox(
                "Horizontal Feature (X):",
                feature_names,
                index=feature_names.index("V14"),
                key="2dx",
                help="Feature displayed on the horizontal axis with marginal box plot.",
            )
        with c_2dy:
            feat_2dy = st.selectbox(
                "Vertical Feature (Y):",
                feature_names,
                index=feature_names.index("V12"),
                key="2dy",
                help="Feature displayed on the vertical axis with marginal box plot.",
            )

        fig_2d = px.scatter(
            scatter_sample,
            x=feat_2dx,
            y=feat_2dy,
            color="Class_Label",
            color_discrete_map=color_map,
            marginal_x="box",
            marginal_y="box",
            opacity=0.75,
            labels={"Class_Label": "Class"},
        )
        apply_dark_theme(fig_2d, height=480, title=f"Bivariate Scatter: {feat_2dx} vs {feat_2dy}")
        fig_2d.update_traces(
            hovertemplate="<b>%{data.name}</b><br>" + f"{feat_2dx}: %{{x:.3f}}<br>{feat_2dy}: %{{y:.3f}}<extra></extra>"
        )
        st.plotly_chart(fig_2d, use_container_width=True)

    with tab_corr:
        st.markdown("Pairwise Pearson correlation structure among top discriminative features, Amount, and Time.")
        key_corr_cols = ["Time", "V4", "V10", "V11", "V12", "V14", "V16", "V17", "Amount", "Actual_Class"]
        corr_matrix = df[key_corr_cols].corr().round(3)

        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1.0,
            zmax=1.0,
            labels=dict(color="Pearson r"),
        )
        apply_dark_theme(fig_corr, height=500, title="Pairwise Pearson Correlation Matrix")
        fig_corr.update_traces(
            hovertemplate="X: %{x}<br>Y: %{y}<br>Pearson Correlation: %{z:.3f}<extra></extra>"
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
    st.markdown("---")

    # ──────────────────────────────────────────────────────────────────────────
    # Section 4: Amount & Temporal Distributions
    # ──────────────────────────────────────────────────────────────────────────
    st.markdown("### 4. Monetary Amount & Temporal Dynamics")

    c_amt, c_time = st.columns(2)

    with c_amt:
        use_log = st.checkbox(
            "Plot log(Amount + 1)",
            value=True,
            help="Applies a natural logarithmic transformation log(1 + Amount) to visualize wide amount ranges ($0 to $25,000+).",
        )
        plot_df = df.copy()
        if use_log:
            plot_df["Amt_Val"] = np.log1p(plot_df["Amount"])
            x_title = "log(Amount + 1)"
        else:
            plot_df["Amt_Val"] = plot_df["Amount"]
            x_title = "Amount ($)"

        fig_amt = px.histogram(
            plot_df,
            x="Amt_Val",
            color="Class_Label",
            color_discrete_map=color_map,
            barmode="overlay",
            nbins=50,
            opacity=0.65,
            labels={"Amt_Val": x_title, "Class_Label": "Class"},
        )
        apply_dark_theme(fig_amt, height=360, title="Transaction Amount Distribution")
        fig_amt.update_traces(hovertemplate="%{data.name}<br>Range: %{x:.2f}<br>Count: %{y}<extra></extra>")
        st.plotly_chart(fig_amt, use_container_width=True)

    with c_time:
        st.markdown("<div style='margin-top: 1.7rem;'></div>", unsafe_allow_html=True)
        fig_time = px.histogram(
            df,
            x="Time",
            color="Class_Label",
            color_discrete_map=color_map,
            barmode="overlay",
            nbins=50,
            opacity=0.65,
            labels={"Time": "Elapsed Seconds", "Class_Label": "Class"},
        )
        apply_dark_theme(fig_time, height=360, title="48-Hour Temporal Distribution")
        fig_time.update_traces(hovertemplate="%{data.name}<br>Seconds: %{x:.0f}s<br>Count: %{y}<extra></extra>")
        st.plotly_chart(fig_time, use_container_width=True)

    st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
    st.markdown("---")

    # ──────────────────────────────────────────────────────────────────────────
    # Section 5: Feature Distribution Profile
    # ──────────────────────────────────────────────────────────────────────────
    st.markdown("### 5. Individual Feature Distribution Inspector")

    selected_feat = st.selectbox(
        "Select Feature for Distribution Inspection:",
        feature_names,
        index=feature_names.index("V14"),
        key="feat_inspect_sel",
        help="Choose any of the 30 input features (Time, V1-V28, Amount) to inspect its distribution separated by class.",
    )

    c_p1, c_p2 = st.columns([2, 1])
    with c_p1:
        fig_feat = px.histogram(
            df,
            x=selected_feat,
            color="Class_Label",
            color_discrete_map=color_map,
            barmode="overlay",
            marginal="violin",
            nbins=70,
            opacity=0.65,
            labels={"Class_Label": "Class"},
        )
        apply_dark_theme(fig_feat, height=380, title=f"Distribution of {selected_feat} by Class")
        fig_feat.update_traces(hovertemplate="%{data.name}<br>Value: %{x:.3f}<br>Count: %{y}<extra></extra>")
        st.plotly_chart(fig_feat, use_container_width=True)

    with c_p2:
        st.markdown(f"**Descriptive Summary: {selected_feat}**")
        desc_df = df.groupby("Class_Label")[selected_feat].describe().round(4).T
        st.dataframe(desc_df, use_container_width=True)
        st.caption(
            "V1 through V28 are anonymized PCA components. No secondary PCA was applied. "
            "Separation between class distributions indicates strong predictive signal."
        )
