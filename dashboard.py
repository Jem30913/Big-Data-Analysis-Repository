import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Albay Flood Risk Dashboard",
    page_icon="🌊",
    layout="wide",
)

OUTPUT_DIR = Path("flood_prediction_output")

RISK_COLORS = {
    "LOW": "#2ecc71",
    "MODERATE": "#f1c40f",
    "HIGH": "#e67e22",
    "EXTREME": "#e74c3c",
}
RISK_ORDER = ["LOW", "MODERATE", "HIGH", "EXTREME"]


@st.cache_data
def load_data():
    df = pd.read_csv(OUTPUT_DIR / "albay_flood_predictions.csv")
    df["datetime"] = pd.to_datetime(df["datetime"])
    extreme = pd.read_csv(OUTPUT_DIR / "extreme_events.csv")
    extreme["datetime"] = pd.to_datetime(extreme["datetime"])
    metrics = pd.read_csv(OUTPUT_DIR / "model_metrics.csv")
    return df, extreme, metrics


@st.cache_resource
def load_model():
    d = joblib.load(OUTPUT_DIR / "flood_model.joblib")
    return d["feature_names"], d["model"].named_steps["clf"].feature_importances_


df, extreme_df, model_metrics = load_data()
feature_names, feature_importances = load_model()

st.title("🌊 Albay Flood Risk Dashboard")
st.markdown(
    "Rainfall-based flood risk prediction from SYNOP station data "
    "(2000–2026)"
)

# ── Sidebar filters ────────────────────────────────────────
st.sidebar.header("Filters")
date_range = st.sidebar.date_input(
    "Date range",
    value=(df["datetime"].min(), df["datetime"].max()),
    min_value=df["datetime"].min(),
    max_value=df["datetime"].max(),
)

risk_filter = st.sidebar.multiselect(
    "Risk classes",
    options=RISK_ORDER,
    default=RISK_ORDER,
)

mask = (
    (df["datetime"].dt.date >= date_range[0])
    & (df["datetime"].dt.date <= date_range[1])
    & (df["flood_risk_label"].isin(risk_filter))
)
filtered = df[mask].copy()

# ── KPI row ────────────────────────────────────────────────
best_idx = model_metrics["f1_weighted"].idxmax()
best = model_metrics.loc[best_idx]

k1, k2, k3, k4 = st.columns(4)
k1.metric("Best Model", best["model"])
k2.metric("Accuracy", f"{best['accuracy']:.2%}")
k3.metric("F1 Score", f"{best['f1_weighted']:.2%}")
k4.metric("AUC (OvR)", f"{best['auc_ovr']:.4f}")

# ── Tabs ───────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📈 Overview", "📊 Model Performance", "⚠️ Extreme Events"])

# ═══════════════════════════════════════════════════════════
# TAB 1: Overview
# ═══════════════════════════════════════════════════════════
with tab1:
    c1, c2 = st.columns([2, 1])

    with c1:
        st.subheader("Rainfall Timeline")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=filtered["datetime"],
            y=filtered["rain_3h"],
            mode="lines",
            name="Rain 3h (mm)",
            line=dict(color="#58a6ff", width=1),
            fill="tozeroy",
            opacity=0.5,
        ))

        r24 = filtered["rain_24h"]
        fig.add_trace(go.Scatter(
            x=filtered["datetime"],
            y=r24,
            mode="lines",
            name="Rain 24h (mm)",
            line=dict(color="#bc8cff", width=1),
            yaxis="y2",
        ))

        for thresh, color, label in [
            (1.5, "#f1c40f", "Moderate"),
            (5.0, "#e67e22", "High"),
            (10.0, "#e74c3c", "Extreme"),
        ]:
            fig.add_hline(
                y=thresh,
                line_dash="dash",
                line_color=color,
                opacity=0.5,
                annotation_text=label,
            )

        fig.update_layout(
            height=400,
            margin=dict(l=0, r=0, t=10, b=0),
            hovermode="x unified",
            yaxis=dict(title="Rain 3h (mm)"),
            yaxis2=dict(
                title="Rain 24h (mm)",
                overlaying="y",
                side="right",
                showgrid=False,
            ),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Risk Distribution")
        counts = filtered["flood_risk_label"].value_counts()
        vals = [counts.get(l, 0) for l in RISK_ORDER]
        fig = go.Figure(data=[go.Pie(
            labels=RISK_ORDER,
            values=vals,
            marker=dict(colors=[RISK_COLORS[l] for l in RISK_ORDER]),
            hole=0.4,
        )])
        fig.update_layout(height=350, margin=dict(l=0, r=0, t=0, b=0), showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Monthly Risk Heatmap")
        filtered["month"] = filtered["datetime"].dt.month
        filtered["year"] = filtered["datetime"].dt.year
        pivot = filtered.groupby(["year", "month"])["flood_risk"].mean().unstack(fill_value=0)
        month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        fig = px.imshow(
            pivot.values,
            x=[month_labels[m - 1] for m in pivot.columns],
            y=pivot.index,
            color_continuous_scale="YlOrRd",
            aspect="auto",
            labels=dict(x="Month", y="Year", color="Mean Risk"),
        )
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════
# TAB 2: Model Performance
# ═══════════════════════════════════════════════════════════
with tab2:
    c1, c2 = st.columns([1, 1])

    with c1:
        st.subheader("Metric Comparison")

        metrics_long = model_metrics.melt(
            id_vars=["model"],
            value_vars=["accuracy", "f1_weighted", "precision_weighted", "recall_weighted", "auc_ovr"],
            var_name="metric",
            value_name="score",
        )
        fig = px.bar(
            metrics_long,
            x="model",
            y="score",
            color="metric",
            barmode="group",
            text_auto=".3f",
            color_discrete_sequence=["#58a6ff", "#bc8cff", "#3fb950", "#e67e22", "#f1c40f"],
        )
        fig.update_layout(
            height=400,
            margin=dict(l=0, r=0, t=10, b=0),
            yaxis=dict(range=[0, 1]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Feature Importance")

        pairs = sorted(zip(feature_names, feature_importances), key=lambda x: -x[1])
        top_n = 15
        names = [p[0] for p in pairs[:top_n]][::-1]
        vals = [p[1] for p in pairs[:top_n]][::-1]

        fig = go.Figure(go.Bar(
            x=vals,
            y=names,
            orientation="h",
            marker_color="#58a6ff",
            text=[f"{v:.4f}" for v in vals],
            textposition="outside",
        ))
        fig.update_layout(
            height=600,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(title="Importance"),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Model Metrics Table")
        styled = model_metrics.copy()
        styled.columns = [c.replace("_", " ").title() for c in styled.columns]
        st.dataframe(styled, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════
# TAB 3: Extreme Events
# ═══════════════════════════════════════════════════════════
with tab3:
    c1, c2 = st.columns([2, 1])

    with c1:
        st.subheader("Extreme Rainfall Events")

        ex_mask = filtered["extreme_event"]
        ex_filtered = filtered[ex_mask]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=filtered["datetime"],
            y=filtered["rain_3h"],
            mode="markers",
            name="Normal",
            marker=dict(color="#8b949e", size=3, opacity=0.3),
        ))
        fig.add_trace(go.Scatter(
            x=ex_filtered["datetime"],
            y=ex_filtered["rain_3h"],
            mode="markers",
            name="Extreme",
            marker=dict(
                color=ex_filtered["flood_risk_label"].map(RISK_COLORS),
                size=8,
                opacity=0.9,
                line=dict(color="white", width=0.5),
            ),
        ))

        fig.add_hline(y=4.6, line_dash="dash", line_color="#e74c3c", opacity=0.6,
                      annotation_text=f"P99 = 4.6 mm/3h")

        fig.update_layout(
            height=400,
            margin=dict(l=0, r=0, t=10, b=0),
            hovermode="closest",
            yaxis=dict(title="Rain 3h (mm)"),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Extreme Event Stats")
        total_extreme = ex_filtered.shape[0]
        total_rows = filtered.shape[0]
        pct = 100 * total_extreme / total_rows if total_rows else 0

        st.metric("Extreme Events", f"{total_extreme:,}", f"{pct:.1f}% of data")
        st.metric("P99 Threshold", "4.6 mm/3h")
        st.metric("P99.5 Threshold", "6.4 mm/3h")

        tier_counts = ex_filtered["extreme_tier"].value_counts()
        for tier in ["extreme", "catastrophic"]:
            cnt = tier_counts.get(tier, 0)
            emoji = "🔶" if tier == "extreme" else "🔴"
            st.markdown(f"{emoji} **{tier.title()}**: {cnt} events")

    st.subheader("Extreme Events Table")
    display_cols = ["datetime", "rain_3h", "rain_percentile", "flood_risk_label",
                    "extreme_tier", "temp", "pressure", "humidity"]
    avail_cols = [c for c in display_cols if c in ex_filtered.columns]
    st.dataframe(
        ex_filtered[avail_cols].sort_values("rain_3h", ascending=False),
        use_container_width=True,
        hide_index=True,
        column_config={
            "datetime": st.column_config.DatetimeColumn("Date/Time"),
            "rain_3h": st.column_config.NumberColumn("Rain 3h (mm)", format="%.2f"),
            "rain_percentile": st.column_config.NumberColumn("Percentile", format="%.4f"),
        },
    )

st.sidebar.markdown("---")
st.sidebar.caption(
    f"Data: {df['datetime'].min().year}–{df['datetime'].max().year}  "
    f"| {len(df):,} observations  "
    f"| Pipeline: Decoder → Feature Engineering → Model → Export"
)
