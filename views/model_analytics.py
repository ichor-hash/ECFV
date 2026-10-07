"""
Page 2: Model Analytics
Comparative evaluation of TabNet, MLP, and FT-Transformer architectures.
Information Visualization focus: Radar charts, Sankey decision flows, and confusion matrices.
Spacious layout, clean typography, and comprehensive tooltips.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from models.model_loader import load_metrics, load_thresholds
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
    st.subheader("Model Analytics & Comparative Benchmarks")
    st.markdown(
        "Empirical evaluation of three deep-learning architectures for tabular fraud detection: "
        "**TabNet** (attentive selection), **MLP** (dense neural network), and **FT-Transformer** (tabular self-attention). "
        "Evaluated on the held-out test split ($N = 16,881$)."
    )

    try:
        metrics = load_metrics()
        thresholds = load_thresholds()
    except Exception as e:
        st.error(f"Error loading model metrics: {str(e)}")
        return

    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────────────────────
    # Section 1: Performance Benchmark Table
    # ──────────────────────────────────────────────────────────────────────────
    st.markdown("### 1. Test Set Performance Benchmark")

    rows = []
    for m in ["TabNet", "MLP", "FT-Transformer"]:
        data = metrics.get(m, {})
        thresh = thresholds.get(m, data.get("threshold", 0.0))
        rows.append(
            {
                "Model": m,
                "ROC-AUC": data.get("roc_auc", 0.0),
                "PR-AUC": data.get("pr_auc", 0.0),
                "Precision": data.get("precision", 0.0),
                "Recall": data.get("recall", 0.0),
                "F1-Score": data.get("f1", 0.0),
                "Decision Threshold": thresh,
            }
        )

    df_metrics = pd.DataFrame(rows)

    display_df = df_metrics.copy()
    for col in ["ROC-AUC", "PR-AUC", "Precision", "Recall", "F1-Score"]:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.6f}")
    display_df["Decision Threshold"] = display_df["Decision Threshold"].apply(lambda x: f"{x:.6f}")

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.caption(
        "Thresholds optimized on validation set to maximize F1, then frozen for testing. "
        "Accuracy is intentionally omitted because a naive zero classifier yields 99.79% accuracy without detecting any fraud."
    )

    st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
    st.markdown("---")

    # ──────────────────────────────────────────────────────────────────────────
    # Section 2: Multivariate Capability Profiling
    # ──────────────────────────────────────────────────────────────────────────
    st.markdown("### 2. Multi-Metric Capability Profiling")

    tab_radar, tab_sankey, tab_tradeoff, tab_cms = st.tabs(
        ["Radar Capability Profiles", "Sankey Classification Flow", "Precision vs Recall Trade-off", "Confusion Matrices"]
    )

    with tab_radar:
        st.markdown(
            "Radar chart comparing multi-metric capability envelopes across architectures. "
            "TabNet expands outward on PR-AUC, Precision, and F1, while MLP peaks on ROC-AUC, and FT-Transformer leads on Recall."
        )

        categories = ["ROC-AUC", "PR-AUC", "Precision", "Recall", "F1-Score"]

        fig_radar = go.Figure()
        for idx, row in df_metrics.iterrows():
            m_name = row["Model"]
            values = [row[c] for c in categories]
            values.append(values[0])

            fig_radar.add_trace(
                go.Scatterpolar(
                    r=values,
                    theta=categories + [categories[0]],
                    name=m_name,
                    line=dict(color=MODEL_COLORS[m_name], width=2.5),
                    fill="toself",
                    opacity=0.22,
                    hovertemplate=f"<b>{m_name}</b><br>%{{theta}}: %{{r:.4f}}<extra></extra>",
                )
            )

        fig_radar.update_layout(
            template="plotly_dark",
            paper_bgcolor=COLOR_BG_CARD,
            plot_bgcolor=COLOR_BG_DARK,
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0.55, 1.0],
                    gridcolor=COLOR_BORDER,
                    tickfont=dict(color=COLOR_TEXT_MUTED, size=9),
                ),
                angularaxis=dict(gridcolor=COLOR_BORDER, tickfont=dict(color=COLOR_TEXT, size=11)),
                bgcolor=COLOR_BG_DARK,
            ),
            height=460,
            margin=dict(l=50, r=50, t=35, b=35),
            legend=dict(yanchor="top", y=1.0, xanchor="left", x=0.0),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with tab_sankey:
        st.markdown(
            "Sankey diagram tracing the distribution of 16,881 test cases from Ground Truth "
            "into detection outcomes (True Negatives, False Positives, False Negatives, True Positives)."
        )

        selected_sankey_model = st.selectbox(
            "Select Model for Classification Flow:",
            ["TabNet", "MLP", "FT-Transformer"],
            help="Select which model's classification pathways to display in the Sankey flow diagram.",
        )
        cm = metrics[selected_sankey_model]["confusion_matrix"]
        tn, fp = cm[0]
        fn, tp = cm[1]

        node_labels = [
            f"Actual Legitimate ({tn+fp:,})",
            f"Actual Fraud ({fn+tp})",
            f"True Negative ({tn:,})",
            f"False Positive ({fp})",
            f"False Negative ({fn})",
            f"True Positive ({tp})",
        ]

        node_colors = [
            COLOR_LEGIT,
            COLOR_FRAUD,
            "#238636",
            "#d29922",
            "#da3633",
            "#2ea043",
        ]

        link_sources = [0, 0, 1, 1]
        link_targets = [2, 3, 4, 5]
        link_values = [tn, fp, fn, tp]
        link_colors = [
            "rgba(56, 139, 253, 0.3)",
            "rgba(210, 153, 34, 0.5)",
            "rgba(218, 54, 51, 0.6)",
            "rgba(46, 160, 67, 0.6)",
        ]

        fig_sankey = go.Figure(
            data=[
                go.Sankey(
                    node=dict(
                        pad=18,
                        thickness=22,
                        line=dict(color=COLOR_BORDER, width=0.5),
                        label=node_labels,
                        color=node_colors,
                    ),
                    link=dict(
                        source=link_sources,
                        target=link_targets,
                        value=link_values,
                        color=link_colors,
                    ),
                )
            ]
        )
        fig_sankey.update_layout(
            template="plotly_dark",
            paper_bgcolor=COLOR_BG_CARD,
            font=dict(color=COLOR_TEXT, size=11),
            height=400,
            margin=dict(l=35, r=35, t=35, b=35),
        )
        st.plotly_chart(fig_sankey, use_container_width=True)
        st.caption(
            f"Visual representation of {selected_sankey_model} test decisions. "
            f"Notice how TabNet restricts False Positives to only 2 cases out of 16,845 legitimate transactions."
        )

    with tab_tradeoff:
        fig_tradeoff = go.Figure()
        for idx, row in df_metrics.iterrows():
            m_name = row["Model"]
            fig_tradeoff.add_trace(
                go.Scatter(
                    x=[row["Recall"]],
                    y=[row["Precision"]],
                    mode="markers+text",
                    name=m_name,
                    text=[f"{m_name}<br>F1: {row['F1-Score']:.4f}"],
                    textposition="top center",
                    marker=dict(size=18, color=MODEL_COLORS[m_name], line=dict(color=COLOR_BORDER, width=1.5)),
                    hovertemplate=f"<b>{m_name}</b><br>Recall: %{{x:.4f}}<br>Precision: %{{y:.4f}}<br>F1: {row['F1-Score']:.4f}<extra></extra>",
                )
            )

        # F1 iso-lines
        r_grid = np.linspace(0.70, 0.85, 50)
        for target_f1 in [0.78, 0.80, 0.82]:
            p_curve = (target_f1 * r_grid) / (2 * r_grid - target_f1)
            valid = (p_curve >= 0.70) & (p_curve <= 0.98)
            if np.any(valid):
                fig_tradeoff.add_trace(
                    go.Scatter(
                        x=r_grid[valid],
                        y=p_curve[valid],
                        mode="lines",
                        line=dict(color=COLOR_BORDER, dash="dot", width=1),
                        showlegend=False,
                        hoverinfo="none",
                    )
                )

        apply_dark_theme(fig_tradeoff, height=420, title="Precision vs Recall Space with F1 Iso-Contours")
        fig_tradeoff.update_xaxes(title="Recall (Sensitivity)", range=[0.72, 0.84])
        fig_tradeoff.update_yaxes(title="Precision (Positive Predictive Value)", range=[0.74, 0.96])
        st.plotly_chart(fig_tradeoff, use_container_width=True)

    with tab_cms:
        st.markdown("Evaluation matrices at frozen validation-calibrated decision thresholds.")
        c_cm1, c_cm2, c_cm3 = st.columns(3)

        for i, m_name in enumerate(["TabNet", "MLP", "FT-Transformer"]):
            cm = metrics[m_name]["confusion_matrix"]
            tn, fp = cm[0]
            fn, tp = cm[1]

            labels = [
                [f"TN<br>{tn:,}", f"FP<br>{fp}"],
                [f"FN<br>{fn}", f"TP<br>{tp}"],
            ]

            fig_cm = go.Figure(
                data=go.Heatmap(
                    z=cm,
                    x=["Pred: Legit", "Pred: Fraud"],
                    y=["Actual: Legit", "Actual: Fraud"],
                    text=labels,
                    texttemplate="%{text}",
                    colorscale=[[0, COLOR_BG_DARK], [1, MODEL_COLORS[m_name]]],
                    showscale=False,
                    hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
                )
            )
            fig_cm.update_layout(
                template="plotly_dark",
                paper_bgcolor=COLOR_BG_CARD,
                plot_bgcolor=COLOR_BG_DARK,
                height=300,
                margin=dict(l=25, r=25, t=40, b=25),
                title=dict(text=f"<b>{m_name}</b>", font=dict(color=COLOR_TEXT_BRIGHT, size=13)),
            )

            with [c_cm1, c_cm2, c_cm3][i]:
                st.plotly_chart(fig_cm, use_container_width=True)
                st.caption(f"Precision: {metrics[m_name]['precision']:.4f} | Recall: {metrics[m_name]['recall']:.4f}")

    st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
    st.markdown("---")

    # ──────────────────────────────────────────────────────────────────────────
    # Section 3: Threshold Calibration Analysis
    # ──────────────────────────────────────────────────────────────────────────
    st.markdown("### 3. Threshold Calibration Analysis")
    st.markdown(
        "Standard 0.5 probability cutoffs fail under extreme imbalance. "
        "Because models were trained with positive class loss weighting (ratio 476.41), "
        "uncalibrated output probabilities skew toward high ranges. "
        "Thresholds were calibrated via validation F1-maximization and frozen before testing."
    )

    c_t1, c_t2, c_t3 = st.columns(3)
    for i, m in enumerate(["TabNet", "MLP", "FT-Transformer"]):
        th = thresholds.get(m, 0.5)
        with [c_t1, c_t2, c_t3][i]:
            st.metric(
                f"{m} Threshold",
                f"{th:.6f}",
                delta=f"{th - 0.5:+.4f} vs 0.5",
                help=f"Optimized probability cutoff for {m} chosen by maximizing validation F1 score.",
            )

    st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
    st.markdown("---")

    # ──────────────────────────────────────────────────────────────────────────
    # Section 4: Architectural Summary
    # ──────────────────────────────────────────────────────────────────────────
    st.markdown("### 4. Architectural Synthesis & Operational Objectives")

    c_s1, c_s2, c_s3 = st.columns(3)

    with c_s1:
        st.markdown(
            """
            **TabNet (Attentive Feature Selection)**
            - **Leading Metrics:** PR-AUC (0.7961), Precision (0.9310), F1 (0.8308)
            - **Design:** Sparse sequential attention masks (`entmax`).
            - **Operational Focus:** Best when customer friction from false blocks or manual investigation overhead must be minimized.
            """
        )

    with c_s2:
        st.markdown(
            """
            **MLP (Deep Multilayer Perceptron)**
            - **Leading Metrics:** ROC-AUC (0.9722)
            - **Design:** Dense 4-layer feedforward network with BatchNorm and Dropout.
            - **Operational Focus:** Best for broad transaction rank-ordering and baseline benchmarking across operational sensitivity bands.
            """
        )

    with c_s3:
        st.markdown(
            """
            **FT-Transformer (Tabular Attention)**
            - **Leading Metrics:** Recall (0.8056)
            - **Design:** Feature Tokenizer with multi-head self-attention (800,897 parameters).
            - **Operational Focus:** Best when financial loss or regulatory penalties for missing fraud heavily outweigh false positive alerts.
            """
        )
