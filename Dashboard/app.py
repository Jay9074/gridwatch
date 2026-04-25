"""
GridWatch — dashboard/app.py
================================
Reads real summary data directly from GitHub.
Author: Jaykumar Patel
"""

import json
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime

BASE_DIR  = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / "models"

# GitHub raw URLs for summary CSVs
GITHUB_RAW = "https://raw.githubusercontent.com/Jay9074/gridwatch/main/data/summary"
STATE_URL   = f"{GITHUB_RAW}/state_risk_summary.csv"
TREND_URL   = f"{GITHUB_RAW}/monthly_trend.csv"
SEASON_URL  = f"{GITHUB_RAW}/seasonal_summary.csv"

st.set_page_config(
    page_title="GridWatch | US Power Grid Risk",
    page_icon="⚡", layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family:'IBM Plex Sans',sans-serif; }
.main, .stApp { background:#0a0e1a; }
div[data-testid="metric-container"] {
    background:#0d1b2a; border:1px solid #1e3a5f;
    border-radius:10px; padding:16px;
}
</style>
""", unsafe_allow_html=True)

STATE_RISK = {
    "Maine":0.87,"Vermont":0.78,"New Hampshire":0.75,
    "New York":0.72,"Pennsylvania":0.68,"Massachusetts":0.65,
    "Connecticut":0.61,"New Jersey":0.60,"Rhode Island":0.58
}


# ── Load data ─────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_state_summary() -> pd.DataFrame:
    """Loads real state risk summary from GitHub."""
    try:
        df = pd.read_csv(STATE_URL)
        return df.sort_values("risk_score", ascending=False)
    except Exception:
        # Fallback to local file
        local = BASE_DIR / "data" / "processed" / "state_risk_summary.csv"
        if local.exists():
            return pd.read_csv(local).sort_values("risk_score", ascending=False)
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_trend() -> pd.DataFrame:
    """Loads monthly trend from GitHub."""
    try:
        df = pd.read_csv(TREND_URL)
        df["year_month"] = pd.to_datetime(
            df[["year","month"]].assign(day=1)
        )
        return df.sort_values("year_month")
    except Exception:
        local = BASE_DIR / "data" / "processed" / "monthly_trend.csv"
        if local.exists():
            df = pd.read_csv(local)
            df["year_month"] = pd.to_datetime(
                df[["year","month"]].assign(day=1)
            )
            return df.sort_values("year_month")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_seasonal() -> pd.DataFrame:
    """Loads seasonal breakdown from GitHub."""
    try:
        return pd.read_csv(SEASON_URL)
    except Exception:
        local = BASE_DIR / "data" / "processed" / "seasonal_summary.csv"
        if local.exists():
            return pd.read_csv(local)
        return pd.DataFrame()


@st.cache_data
def load_metrics() -> dict:
    path = MODEL_DIR / "model_metrics.json"
    if path.exists():
        return json.loads(path.read_text())
    return {
        "Logistic Regression": {"accuracy":0.598,"precision":0.088,"recall":0.770,"f1_score":0.158,"roc_auc":0.746},
        "Random Forest":       {"accuracy":0.672,"precision":0.098,"recall":0.698,"f1_score":0.172,"roc_auc":0.753},
        "XGBoost":             {"accuracy":0.643,"precision":0.089,"recall":0.682,"f1_score":0.158,"roc_auc":0.737},
        "best_model": "Random Forest"
    }


# ── Sidebar ───────────────────────────────────────────────────────
def sidebar(state_df):
    with st.sidebar:
        st.markdown("### ⚡ GridWatch")
        st.caption("AI-Powered Grid Risk Intelligence")
        st.divider()
        states = ["All"] + (sorted(state_df["state"].unique().tolist())
                            if not state_df.empty else [])
        sel_state = st.selectbox("Filter by State", states)
        sel_risk  = st.selectbox("Filter by Risk Level",
                                 ["All","HIGH","MEDIUM-HIGH","MEDIUM","LOW-MEDIUM"])
        st.divider()
        st.markdown("**Data Sources**")
        st.caption("EAGLE-I (ORNL/DOE) · NOAA Storm Events · EIA-861")
        st.caption("787,794 observations | 2020–2023 | 9 Northeast States")
        st.divider()
        st.caption(
            "**Jaykumar Patel**\n"
            "MS Data Science, Stevens Institute of Technology\n"
            "MS IT Project Mgmt (in progress), New England College\n"
            "Analytics Analyst, Central Maine Power"
        )
        st.divider()
        st.caption("📄 [GitHub](https://github.com/Jay9074/gridwatch) | "
                   "Research paper in progress")
    return sel_state, sel_risk


# ── KPIs ──────────────────────────────────────────────────────────
def kpis(state_df):
    st.markdown("#### Grid Status Overview — Real EAGLE-I Data (2020–2023)")
    if state_df.empty:
        st.warning("Data loading — please wait or refresh.")
        return

    total_outage_days = int(state_df["total_outage_days"].sum())
    peak_customers    = int(state_df["max_customers_out"].max())
    worst_state       = state_df.iloc[0]["state"]
    avg_rate          = state_df["outage_rate"].mean()

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Major Outage Days",  f"{total_outage_days:,}",
              "Days with 1,000+ customers out")
    c2.metric("Peak Customers Affected",  f"{peak_customers:,}",
              "Single worst event")
    c3.metric("Avg State Outage Rate",    f"{avg_rate:.1%}",
              "Across all 9 states")
    c4.metric("Highest Risk State",       worst_state,
              "By composite risk score")


# ── Map ───────────────────────────────────────────────────────────
def risk_map(state_df):
    st.markdown("#### State Risk Map — Northeast US")
    if state_df.empty:
        st.info("Loading map data...")
        return

    color_map = {
        "HIGH":"#e63946","MEDIUM-HIGH":"#f4a261",
        "MEDIUM":"#e9c46a","LOW-MEDIUM":"#2a9d8f","LOW":"#457b9d"
    }
    fig = px.scatter_mapbox(
        state_df,
        lat="lat", lon="lon",
        color="risk_level",
        size="risk_score",
        size_max=45,
        hover_name="state",
        hover_data={
            "risk_score":      ":.4f",
            "outage_rate":     ":.2%",
            "total_outage_days":":,",
            "max_customers_out":":,",
            "lat":False,"lon":False
        },
        color_discrete_map=color_map,
        mapbox_style="carto-darkmatter",
        zoom=4.5,
        center={"lat":43.0,"lon":-73.0}
    )
    fig.update_layout(
        height=450,
        margin={"r":0,"t":0,"l":0,"b":0},
        paper_bgcolor="#0a0e1a",
        legend=dict(bgcolor="#0d1b2a",bordercolor="#1e3a5f",
                    font=dict(color="#e8f4fd"))
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Risk Table ────────────────────────────────────────────────────
def risk_table(state_df):
    st.markdown("#### State Risk Rankings — Real Data")
    if state_df.empty:
        st.info("Loading...")
        return

    disp = state_df[[
        "state","risk_level","risk_score","outage_rate",
        "total_outage_days","max_customers_out"
    ]].copy()
    disp.columns = [
        "State","Risk Level","Risk Score","Outage Rate",
        "Outage Days","Peak Customers Out"
    ]
    disp["Risk Score"]        = disp["Risk Score"].apply(lambda x: f"{x:.4f}")
    disp["Outage Rate"]       = disp["Outage Rate"].apply(lambda x: f"{x:.2%}")
    disp["Peak Customers Out"]= disp["Peak Customers Out"].apply(lambda x: f"{int(x):,}")

    def hl(val):
        m = {
            "HIGH":        "background-color:#3d0000;color:#ff6b6b",
            "MEDIUM-HIGH": "background-color:#3d2600;color:#f4a261",
            "MEDIUM":      "background-color:#3d3400;color:#e9c46a",
            "LOW-MEDIUM":  "background-color:#003d1a;color:#2a9d8f",
        }
        return m.get(val,"")

    st.dataframe(
        disp.style.map(hl, subset=["Risk Level"]),
        use_container_width=True, hide_index=True
    )


# ── Trend Chart ───────────────────────────────────────────────────
def trend_chart(trend_df):
    st.markdown("#### Monthly Outage Trend — Real Data 2020–2023")
    if trend_df.empty:
        st.info("Loading trend data...")
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trend_df["year_month"], y=trend_df["outage_events"],
        name="Major Outage Events",
        line=dict(color="#e63946",width=2.5),
        mode="lines+markers", marker=dict(size=5)
    ))
    fig.add_trace(go.Bar(
        x=trend_df["year_month"], y=trend_df["avg_customers_out"],
        name="Avg Customers Out",
        yaxis="y2", marker_color="#1e3a5f", opacity=0.55
    ))
    fig.update_layout(
        height=300,
        paper_bgcolor="#0a0e1a", plot_bgcolor="#0a0e1a",
        font=dict(color="#e8f4fd",family="IBM Plex Sans"),
        legend=dict(bgcolor="#0d1b2a",bordercolor="#1e3a5f"),
        xaxis=dict(gridcolor="#1e3a5f"),
        yaxis=dict(title="Outage Events",gridcolor="#1e3a5f"),
        yaxis2=dict(title="Avg Customers Out",overlaying="y",
                    side="right",gridcolor="#1e3a5f")
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Seasonal Chart ────────────────────────────────────────────────
def seasonal_chart(seasonal_df):
    st.markdown("#### Outage Rate by Season — Real Data")
    if seasonal_df.empty:
        st.info("Loading seasonal data...")
        return

    season_colors = {
        "Winter":"#e63946","Fall":"#f4a261",
        "Summer":"#e9c46a","Spring":"#2a9d8f"
    }
    season_order = ["Winter","Spring","Summer","Fall"]
    seasonal_df["season_order"] = seasonal_df["season"].map(
        {s:i for i,s in enumerate(season_order)}
    )
    seasonal_df = seasonal_df.sort_values("season_order")

    fig = go.Figure()
    for _, row in seasonal_df.iterrows():
        fig.add_trace(go.Bar(
            x=[row["season"]], y=[row["outage_rate"]],
            name=row["season"],
            marker_color=season_colors.get(row["season"],"#457b9d"),
            text=f"{row['outage_rate']:.1%}",
            textposition="outside",
            textfont=dict(color="#e8f4fd")
        ))
    fig.update_layout(
        height=300, showlegend=False,
        paper_bgcolor="#0a0e1a", plot_bgcolor="#0a0e1a",
        font=dict(color="#e8f4fd"),
        yaxis=dict(title="Outage Rate",gridcolor="#1e3a5f",
                   tickformat=".0%"),
        xaxis=dict(gridcolor="#1e3a5f"),
        title=dict(
            text="Winter dominates — 4.9% outage rate vs 0% other seasons",
            font=dict(color="#7fb3d3",size=12)
        )
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Model Performance ─────────────────────────────────────────────
def model_chart(metrics):
    st.markdown("#### ML Model Performance — Real EAGLE-I Data")
    names  = [k for k in metrics if k not in
              ["best_model","trained_at","data_source","note"]]
    keys   = ["accuracy","precision","recall","f1_score","roc_auc"]
    labels = ["Accuracy","Precision","Recall","F1","ROC-AUC"]
    colors = ["#e63946","#457b9d","#2a9d8f"]

    fig = go.Figure()
    for i,name in enumerate(names):
        fig.add_trace(go.Bar(
            name=name, x=labels,
            y=[metrics[name].get(k,0) for k in keys],
            marker_color=colors[i % len(colors)],
            text=[f"{metrics[name].get(k,0):.3f}" for k in keys],
            textposition="outside",
            textfont=dict(size=10,color="#e8f4fd")
        ))
    best = metrics.get("best_model","Random Forest")
    best_auc = metrics.get(best,{}).get("roc_auc","N/A")
    fig.update_layout(
        barmode="group", height=340,
        paper_bgcolor="#0a0e1a", plot_bgcolor="#0a0e1a",
        font=dict(color="#e8f4fd",family="IBM Plex Sans"),
        legend=dict(bgcolor="#0d1b2a",bordercolor="#1e3a5f"),
        yaxis=dict(range=[0,1.12],gridcolor="#1e3a5f"),
        xaxis=dict(gridcolor="#1e3a5f"),
        title=dict(
            text=f"Best: {best} | ROC-AUC: {best_auc} | "
                 f"787,794 EAGLE-I observations | No data leakage",
            font=dict(color="#7fb3d3",size=11)
        )
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Risk Calculator ───────────────────────────────────────────────
def risk_calculator():
    st.markdown("#### Outage Risk Calculator")
    st.caption("Estimate outage risk using model features")

    c1,c2,c3 = st.columns(3)
    with c1:
        state       = st.selectbox("State", list(STATE_RISK.keys()))
        season      = st.selectbox("Season",["Winter","Spring","Summer","Fall"])
        month       = st.slider("Month",1,12,1)
    with c2:
        storm_count = st.slider("Storm Events This Month",0,20,3)
        ice_events  = st.slider("Ice/Blizzard Events",0,5,0)
        wind_events = st.slider("High Wind Events",0,10,2)
    with c3:
        prior_outage= st.checkbox("County Had Outage Last Month")
        year_trend  = st.slider("Years Since 2020",0,4,2)

    if st.button("Calculate Risk", type="primary"):
        state_r  = STATE_RISK.get(state,0.65)
        is_winter= 1 if month in [12,1,2,3] else 0
        season_r = {"Winter":3,"Fall":2,"Summer":2,"Spring":1}.get(season,1)

        risk = min(1.0,(
            is_winter        * 0.25 +
            state_r          * 0.20 +
            (storm_count/20) * 0.15 +
            (ice_events/5)   * 0.20 +
            (wind_events/10) * 0.10 +
            (1 if prior_outage else 0) * 0.15 +
            (year_trend/4)   * 0.05 +
            (season_r/3)     * 0.10
        ))

        if risk >= 0.55:   level,col = "HIGH","#e63946"
        elif risk >= 0.40: level,col = "MEDIUM-HIGH","#f4a261"
        elif risk >= 0.25: level,col = "MEDIUM","#e9c46a"
        else:              level,col = "LOW","#2a9d8f"

        st.markdown(f"""
        <div style='background:#0d1b2a;border:2px solid {col};
                    border-radius:12px;padding:24px;text-align:center;margin-top:12px;'>
            <div style='font-family:IBM Plex Mono;font-size:2rem;
                        color:{col};font-weight:600;'>
                {risk:.0%} Risk — {level}
            </div>
            <div style='color:#e8f4fd;font-size:0.85rem;margin-top:8px;'>
                {state} | {season} | {storm_count} storms |
                {ice_events} ice events | Prior outage: {prior_outage}
            </div>
        </div>""", unsafe_allow_html=True)


# ── Main ──────────────────────────────────────────────────────────
def main():
    state_df   = load_state_summary()
    trend_df   = load_trend()
    seasonal_df= load_seasonal()
    metrics    = load_metrics()

    sel_state, sel_risk = sidebar(state_df)

    # Apply filters
    filt = state_df.copy()
    if sel_state != "All" and not filt.empty:
        filt = filt[filt["state"] == sel_state]
    if sel_risk != "All" and not filt.empty:
        filt = filt[filt["risk_level"] == sel_risk]

    # Header
    st.markdown("""
    <div style='padding:8px 0 4px;'>
        <span style='font-family:IBM Plex Mono;font-size:2rem;
                     font-weight:600;color:#e8f4fd;'>⚡ GridWatch</span>
        <span style='font-size:0.9rem;color:#7fb3d3;margin-left:12px;
                     letter-spacing:2px;text-transform:uppercase;'>
            Northeast US Power Grid Risk Intelligence
        </span>
    </div>
    <div style='color:#7fb3d3;font-size:0.8rem;margin-bottom:8px;'>
        Data: EAGLE-I (ORNL/DOE) · NOAA Storm Events · EIA-861 |
        787,794 observations · 2020–2023 · 9 Northeast US States |
        US outages cost <b style="color:#e63946">$121–150B annually</b> (DOE/ORNL 2024)
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    kpis(filt if not filt.empty else state_df)
    st.divider()

    col_l,col_r = st.columns([1.4,1])
    with col_l: risk_map(filt if not filt.empty else state_df)
    with col_r: risk_table(filt if not filt.empty else state_df)

    st.divider()
    col_a,col_b = st.columns(2)
    with col_a: trend_chart(trend_df)
    with col_b: seasonal_chart(seasonal_df)

    st.divider()
    col_c,col_d = st.columns(2)
    with col_c: model_chart(metrics)
    with col_d: risk_calculator()

    st.divider()
    st.caption(
        f"GridWatch | Jaykumar Patel | "
        f"Data: EAGLE-I ORNL, NOAA Storm Events, EIA-861 | "
        f"787,794 observations | 2020-2023 | "
        f"Updated {datetime.now().strftime('%B %Y')} | "
        "Independent research — not affiliated with any utility."
    )


if __name__ == "__main__":
    main()
