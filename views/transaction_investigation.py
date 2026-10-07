"""
Page 3: Transaction Investigation
Interactive forensic transaction inspector with tri-model consensus,
latent subspace positioning, parallel coordinate profiling, and population anomaly analysis.
Spacious, dark minimalist aesthetic with comprehensive tooltips.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from models.model_loader import (
    load_test_data,
    load_feature_names,
    load_thresholds,
    load_tabnet,
    load_mlp,
    load_ft_transformer,
    load_scaler,
)
from utils.preprocessing import predict_single_transaction, get_device
from utils.visualization import (
    apply_dark_theme,
    MODEL_COLORS,
    COLOR_LEGIT,
    COLOR_FRAUD,
    COLOR_BORDER,
    COLOR_BG_CARD,
    COLOR_BG_DARK,
    COLOR_TEXT,
    COLOR_TEXT_BRIGHT,
    COLOR_TEXT_MUTED,
)


def show():
    st.subheader("Transaction Investigation & Forensic Analytics")
    st.markdown(
        "Individual transaction diagnosis. Cross-reference tri-model ensemble decisions, "
        "locate transactions within discriminative feature subspaces, and inspect standardized deviations."
    )

    try:
        test_data = load_test_data()
        feature_names = load_feature_names()
        thresholds = load_thresholds()
    except Exception as e:
        st.error(f"Error loading transaction investigation data: {str(e)}")
        return

    test_data["Consensus_Fraud_Count"] = (
        (test_data["TabNet_Prediction"] == 1).astype(int)
        + (test_data["MLP_Prediction"] == 1).astype(int)
        + (test_data["FT_Transformer_Prediction"] == 1).astype(int)
    )

    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────────────────────
    # Section 1: Transaction Selection Controls
    # ──────────────────────────────────────────────────────────────────────────
    st.markdown("### 1. Case Selector")

    filter_mode = st.radio(
        "Selection Filter:",
        [
            "Primary Explainability Case (Row 12507)",
            "Verified Actual Fraud Cases (36 Test Cases)",
            "Model Disagreement Cases (1/3 or 2/3 Fraud Votes)",
            "Manual Row Index",
        ],
        horizontal=True,
        help="Filter candidate transactions by explainability baseline, ground-truth fraud status, ensemble consensus split, or direct row index.",
    )

    st.markdown("<div style='margin-top: 0.5rem;'></div>", unsafe_allow_html=True)

    if filter_mode == "Primary Explainability Case (Row 12507)":
        selected_idx = 12507
        st.caption("Active Case: Row 12507 (Benchmark focal case used in local SHAP and LIME studies).")

    elif filter_mode == "Verified Actual Fraud Cases (36 Test Cases)":
        fraud_indices = test_data[test_data["Actual_Class"] == 1].index.tolist()
        selected_idx = st.selectbox(
            "Select Verified Fraud Transaction:",
            fraud_indices,
            format_func=lambda i: f"Row {i} | Amount: ${test_data.loc[i, 'Amount']:.2f} | Time: {test_data.loc[i, 'Time']:.0f}s | Fraud Votes: {test_data.loc[i, 'Consensus_Fraud_Count']}/3",
            help="Select one of the 36 confirmed ground-truth fraud cases in the test partition.",
        )

    elif filter_mode == "Model Disagreement Cases (1/3 or 2/3 Fraud Votes)":
        disagree_indices = test_data[test_data["Consensus_Fraud_Count"].isin([1, 2])].index.tolist()
        selected_idx = st.selectbox(
            "Select Edge-Case Transaction with Split Decisions:",
            disagree_indices,
            format_func=lambda i: f"Row {i} | Actual: {'Fraud' if test_data.loc[i, 'Actual_Class']==1 else 'Legit'} | Fraud Votes: {test_data.loc[i, 'Consensus_Fraud_Count']}/3 | Amount: ${test_data.loc[i, 'Amount']:.2f}",
            help="Select an ambiguous transaction where the three architectures split their decisions.",
        )

    else:
        c1, c2 = st.columns([2, 1])
        with c1:
            selected_idx = st.number_input(
                f"Enter Test Row Index (0 to {len(test_data)-1}):",
                min_value=0,
                max_value=len(test_data)-1,
                value=12507,
                step=1,
                help=f"Index of any test transaction between 0 and {len(test_data)-1:,}.",
            )
        with c2:
            st.caption(f"Total available test cases: {len(test_data):,}")

    transaction = test_data.iloc[selected_idx]
    actual_label = "Fraud" if transaction["Actual_Class"] == 1 else "Legitimate"

    st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
    st.markdown("---")

    # ──────────────────────────────────────────────────────────────────────────
    # Section 2: Case Overview & Ensemble Agreement
    # ──────────────────────────────────────────────────────────────────────────
    st.markdown(f"### 2. Multi-Model Consensus: Row #{selected_idx}")

    c_ov1, c_ov2, c_ov3 = st.columns([1.2, 1.2, 2])
    with c_ov1:
        if actual_label == "Fraud":
            st.markdown('<span class="status-badge-fraud">GROUND TRUTH: FRAUD</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-badge-legit">GROUND TRUTH: LEGITIMATE</span>', unsafe_allow_html=True)
        st.write(f"Amount: **${transaction['Amount']:.2f}**")

    with c_ov2:
        st.write(f"Elapsed Time: **{transaction['Time']:.0f}s**")
        st.write(f"Row Position: **{selected_idx:,}**")

    with c_ov3:
        fraud_votes = int(transaction["Consensus_Fraud_Count"])
        if fraud_votes == 3:
            st.markdown("**Consensus: Unanimous Fraud (3/3 Models Vote FRAUD)**")
        elif fraud_votes == 0:
            st.markdown("**Consensus: Unanimous Legitimate (3/3 Models Vote LEGITIMATE)**")
        elif fraud_votes == 2:
            st.markdown("**Consensus: Majority Fraud (2/3 Models Vote FRAUD)**")
        else:
            st.markdown("**Consensus: Disagreement (1/3 Models Votes FRAUD)**")

    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

    # Probability Gauges
    c_m1, c_m2, c_m3 = st.columns(3)
    models_meta = [
        {"name": "TabNet", "prefix": "TabNet", "key": "TabNet"},
        {"name": "MLP", "prefix": "MLP", "key": "MLP"},
        {"name": "FT-Transformer", "prefix": "FT_Transformer", "key": "FT-Transformer"},
    ]

    for i, meta in enumerate(models_meta):
        prob = float(transaction[f"{meta['prefix']}_Probability"])
        pred_int = int(transaction[f"{meta['prefix']}_Prediction"])
        thresh = float(thresholds.get(meta["key"], 0.5))
        is_fraud = pred_int == 1

        with [c_m1, c_m2, c_m3][i]:
            st.metric(
                label=f"{meta['name']} Probability",
                value=f"{prob:.6f}",
                delta="FRAUD" if is_fraud else "LEGITIMATE",
                delta_color="inverse" if is_fraud else "normal",
                help=f"Model output probability for {meta['name']} against decision threshold {thresh:.6f}.",
            )
            st.caption(f"Decision Threshold: {thresh:.6f}")

            fig_g = go.Figure(
                go.Indicator(
                    mode="gauge",
                    value=prob,
                    domain={"x": [0, 1], "y": [0, 1]},
                    gauge={
                        "axis": {"range": [0, 1], "tickcolor": COLOR_TEXT_MUTED},
                        "bar": {"color": COLOR_FRAUD if is_fraud else COLOR_LEGIT},
                        "bgcolor": COLOR_BG_DARK,
                        "borderwidth": 1,
                        "bordercolor": COLOR_BORDER,
                        "threshold": {
                            "line": {"color": "#f0f6fc", "width": 3},
                            "thickness": 0.8,
                            "value": thresh,
                        },
                    },
                )
            )
            fig_g.update_layout(
                paper_bgcolor=COLOR_BG_CARD,
                height=120,
                margin=dict(l=15, r=15, t=10, b=10),
            )
            st.plotly_chart(fig_g, use_container_width=True)

    st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
    st.markdown("---")

    # ──────────────────────────────────────────────────────────────────────────
    # Section 3: Subspace Positioning & Parallel Profile
    # ──────────────────────────────────────────────────────────────────────────
    st.markdown("### 3. Subspace Positioning & Population Signature")

    tab_locator, tab_parallel_profile, tab_zscore = st.tabs(
        ["Latent Subspace Locator", "Forensic Parallel Profile", "Standardized Z-Score Anomaly Spectrum"]
    )

    with tab_locator:
        st.markdown(
            "Locate this specific transaction (target reticle) within the population feature distribution. "
            "Select projection axes to view neighborhood context."
        )

        c_lx, c_ly = st.columns(2)
        with c_lx:
            loc_x = st.selectbox(
                "Subspace Axis 1 (X):",
                feature_names,
                index=feature_names.index("V14"),
                key="loc_x",
                help="Horizontal feature coordinate for transaction locator scatter.",
            )
        with c_ly:
            loc_y = st.selectbox(
                "Subspace Axis 2 (Y):",
                feature_names,
                index=feature_names.index("V4"),
                key="loc_y",
                help="Vertical feature coordinate for transaction locator scatter.",
            )

        pop_sample = test_data.sample(n=1000, random_state=42)

        fig_loc = go.Figure()

        legit_pts = pop_sample[pop_sample["Actual_Class"] == 0]
        fig_loc.add_trace(
            go.Scatter(
                x=legit_pts[loc_x],
                y=legit_pts[loc_y],
                mode="markers",
                name="Legitimate Background",
                marker=dict(color=COLOR_LEGIT, size=4, opacity=0.3),
                hovertemplate=f"Legitimate<br>{loc_x}: %{{x:.3f}}<br>{loc_y}: %{{y:.3f}}<extra></extra>",
            )
        )

        fraud_pts = test_data[test_data["Actual_Class"] == 1]
        fig_loc.add_trace(
            go.Scatter(
                x=fraud_pts[loc_x],
                y=fraud_pts[loc_y],
                mode="markers",
                name="Known Fraud Cases",
                marker=dict(color=COLOR_FRAUD, size=7, opacity=0.8),
                hovertemplate=f"Fraud Case<br>{loc_x}: %{{x:.3f}}<br>{loc_y}: %{{y:.3f}}<extra></extra>",
            )
        )

        fig_loc.add_trace(
            go.Scatter(
                x=[transaction[loc_x]],
                y=[transaction[loc_y]],
                mode="markers+text",
                name=f"Investigated Case (#{selected_idx})",
                text=[f"Row #{selected_idx}"],
                textposition="top right",
                marker=dict(
                    color="#56d364",
                    size=16,
                    symbol="cross",
                    line=dict(color="#f0f6fc", width=2),
                ),
                hovertemplate=f"<b>Current Case #{selected_idx}</b><br>{loc_x}: %{{x:.3f}}<br>{loc_y}: %{{y:.3f}}<extra></extra>",
            )
        )

        apply_dark_theme(fig_loc, height=460, title=f"Latent Projection: {loc_x} vs {loc_y} with Transaction Locator")
        fig_loc.update_xaxes(title=loc_x)
        fig_loc.update_yaxes(title=loc_y)
        st.plotly_chart(fig_loc, use_container_width=True)

    with tab_parallel_profile:
        st.markdown(
            "Forensic signature comparing the investigated transaction's trajectory (green line) "
            "against the 10th to 90th percentile baseline of legitimate transactions across top features."
        )

        key_forensic_feats = ["V14", "V12", "V10", "V4", "V11", "V17", "Amount"]

        legit_data = test_data[test_data["Actual_Class"] == 0][key_forensic_feats]
        p10 = legit_data.quantile(0.10)
        p50 = legit_data.quantile(0.50)
        p90 = legit_data.quantile(0.90)
        tx_vals = transaction[key_forensic_feats]

        fig_forensic = go.Figure()

        fig_forensic.add_trace(
            go.Scatter(
                x=key_forensic_feats,
                y=p90.values,
                mode="lines",
                line=dict(color=COLOR_BORDER, width=1),
                name="90th Percentile (Legit)",
            )
        )
        fig_forensic.add_trace(
            go.Scatter(
                x=key_forensic_feats,
                y=p10.values,
                mode="lines",
                fill="tonexty",
                fillcolor="rgba(56, 139, 253, 0.12)",
                line=dict(color=COLOR_BORDER, width=1),
                name="10th-90th Percentile Band (Legit)",
            )
        )
        fig_forensic.add_trace(
            go.Scatter(
                x=key_forensic_feats,
                y=p50.values,
                mode="lines+markers",
                line=dict(color=COLOR_LEGIT, dash="dash", width=1.5),
                name="Median Baseline (Legit)",
            )
        )

        fig_forensic.add_trace(
            go.Scatter(
                x=key_forensic_feats,
                y=tx_vals.values,
                mode="lines+markers",
                line=dict(color="#56d364", width=3),
                marker=dict(size=8, color="#56d364"),
                name=f"Transaction #{selected_idx}",
                hovertemplate="Feature: %{x}<br>Transaction Value: %{y:.3f}<extra></extra>",
            )
        )

        apply_dark_theme(fig_forensic, height=400, title="Forensic Signature vs Population Reference Envelope")
        st.plotly_chart(fig_forensic, use_container_width=True)

    with tab_zscore:
        feat_vals = transaction[feature_names]
        pop_mean = test_data[feature_names].mean()
        pop_std = test_data[feature_names].std()
        z_scores = (feat_vals - pop_mean) / pop_std

        z_df = pd.DataFrame({
            "Feature": feature_names,
            "Z_Score": z_scores.values,
            "Abs_Z": np.abs(z_scores.values),
        }).sort_values("Abs_Z", ascending=True)

        fig_z = go.Figure(
            go.Bar(
                x=z_df["Z_Score"],
                y=z_df["Feature"],
                orientation="h",
                marker_color=[
                    COLOR_FRAUD if abs(z) > 2.0 else COLOR_LEGIT for z in z_df["Z_Score"]
                ],
                hovertemplate="Feature: %{y}<br>Z-Score: %{x:+.3f} std dev<extra></extra>",
            )
        )
        fig_z.add_vline(x=2.0, line_dash="dash", line_color=COLOR_BORDER)
        fig_z.add_vline(x=-2.0, line_dash="dash", line_color=COLOR_BORDER)
        apply_dark_theme(fig_z, height=620, title="Standardized Feature Deviations (Z-Scores, +/-2 sigma cutoffs)")
        fig_z.update_xaxes(title="Z-Score (Standard Deviations)")
        st.plotly_chart(fig_z, use_container_width=True)

    # Live Inference Option
    with st.expander("Live Neural Network Re-Inference"):
        if st.button("Execute Live PyTorch Inference on Current Inputs", help="Performs forward pass inference through the in-memory PyTorch models."):
            with st.spinner("Executing forward pass on CPU/CUDA..."):
                try:
                    scaler = load_scaler()
                    tn = load_tabnet()
                    mlp = load_mlp()
                    ft = load_ft_transformer()
                    dev = get_device()

                    raw_inputs = transaction[feature_names].values.reshape(1, -1)
                    scaled_inputs = scaler.transform(raw_inputs)
                    live_res = predict_single_transaction(scaled_inputs, tn, mlp, ft, thresholds, dev)

                    live_table = pd.DataFrame([
                        {
                            "Model": m,
                            "Live Probability": f"{r['probability']:.8f}",
                            "Stored Test Probability": f"{transaction[f'FT_Transformer_Probability' if m=='FT-Transformer' else f'{m}_Probability']:.8f}",
                            "Live Classification": r["prediction"],
                            "Threshold": f"{r['threshold']:.6f}",
                        }
                        for m, r in live_res.items()
                    ])
                    st.dataframe(live_table, use_container_width=True, hide_index=True)
                except Exception as ex:
                    st.error(f"Inference error: {str(ex)}")
