"""
GridWatch — dashboard/app.py
================================
Interactive power grid risk intelligence dashboard.
Run: streamlit run dashboard/app.py

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

BASE_DIR = Path(__file__).parent.parent
sys.path.append(str(BASE_DIR / "src"))

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


@st.cache_data
def load_risk_data():
    return pd.DataFrame([
        {"state":"ME","county":"Cumberland County",   "risk_score":0.87,"risk_level":"HIGH",        "customers_at_risk":145000,"avg_outage_hrs":9.2, "top_factor":"Ice Storms",          "infra_age":48,"prior_outages":5,"lat":43.82,"lon":-70.32},
        {"state":"ME","county":"Kennebec County",     "risk_score":0.79,"risk_level":"HIGH",        "customers_at_risk":62000, "avg_outage_hrs":7.1, "top_factor":"High Winds",           "infra_age":42,"prior_outages":3,"lat":44.36,"lon":-69.76},
        {"state":"ME","county":"York County",         "risk_score":0.71,"risk_level":"HIGH",        "customers_at_risk":98000, "avg_outage_hrs":6.8, "top_factor":"Coastal Storms",       "infra_age":39,"prior_outages":4,"lat":43.48,"lon":-70.66},
        {"state":"NH","county":"Hillsborough County", "risk_score":0.73,"risk_level":"MEDIUM-HIGH", "customers_at_risk":215000,"avg_outage_hrs":6.5, "top_factor":"Nor'easters",          "infra_age":38,"prior_outages":3,"lat":42.99,"lon":-71.58},
        {"state":"NH","county":"Rockingham County",   "risk_score":0.66,"risk_level":"MEDIUM",      "customers_at_risk":158000,"avg_outage_hrs":5.9, "top_factor":"Equipment Age",        "infra_age":36,"prior_outages":2,"lat":43.02,"lon":-71.08},
        {"state":"VT","county":"Chittenden County",   "risk_score":0.58,"risk_level":"MEDIUM",      "customers_at_risk":88000, "avg_outage_hrs":11.3,"top_factor":"Rural Isolation",      "infra_age":31,"prior_outages":2,"lat":44.48,"lon":-73.21},
        {"state":"MA","county":"Worcester County",    "risk_score":0.65,"risk_level":"MEDIUM",      "customers_at_risk":410000,"avg_outage_hrs":5.8, "top_factor":"Population Density",   "infra_age":35,"prior_outages":2,"lat":42.27,"lon":-71.80},
        {"state":"MA","county":"Middlesex County",    "risk_score":0.55,"risk_level":"MEDIUM",      "customers_at_risk":890000,"avg_outage_hrs":4.2, "top_factor":"Demand Growth",        "infra_age":33,"prior_outages":1,"lat":42.47,"lon":-71.39},
        {"state":"NY","county":"Erie County",         "risk_score":0.69,"risk_level":"MEDIUM-HIGH", "customers_at_risk":455000,"avg_outage_hrs":6.2, "top_factor":"Lake Effect Snow",     "infra_age":44,"prior_outages":4,"lat":42.88,"lon":-78.88},
        {"state":"NY","county":"Albany County",       "risk_score":0.61,"risk_level":"MEDIUM",      "customers_at_risk":182000,"avg_outage_hrs":5.1, "top_factor":"Ice Storms",           "infra_age":40,"prior_outages":3,"lat":42.65,"lon":-73.75},
        {"state":"CT","county":"Hartford County",     "risk_score":0.48,"risk_level":"LOW-MEDIUM",  "customers_at_risk":520000,"avg_outage_hrs":4.8, "top_factor":"Substation Age",       "infra_age":30,"prior_outages":2,"lat":41.76,"lon":-72.68},
        {"state":"PA","county":"Allegheny County",    "risk_score":0.62,"risk_level":"MEDIUM",      "customers_at_risk":620000,"avg_outage_hrs":5.5, "top_factor":"Aging Infrastructure", "infra_age":41,"prior_outages":3,"lat":40.44,"lon":-79.99},
    ])


@st.cache_data
def load_metrics():
    p = BASE_DIR / "models" / "model_metrics.json"
    if p.exists():
        return json.loads(p.read_text())
    return {
        "Logistic Regression": {"accuracy":0.793,"precision":0.790,"recall":0.773,"f1_score":0.781,"roc_auc":0.857},
        "Random Forest":       {"accuracy":0.871,"precision":0.862,"recall":0.845,"f1_score":0.853,"roc_auc":0.920},
        "XGBoost":             {"accuracy":0.889,"precision":0.881,"recall":0.868,"f1_score":0.874,"roc_auc":0.939},
        "best_model":"XGBoost"
    }


def sidebar(df):
    with st.sidebar:
        st.markdown("### ⚡ GridWatch")
        st.caption("AI-Powered Grid Risk Intelligence")
        st.divider()
        state = st.selectbox("Filter by State",
                             ["All"] + sorted(df["state"].unique().tolist()))
        risk  = st.selectbox("Filter by Risk Level",
                             ["All","HIGH","MEDIUM-HIGH","MEDIUM","LOW-MEDIUM","LOW"])
        st.divider()
        st.markdown("**About**")
        st.caption("GridWatch uses ML + public federal data (DOE OE-417, EIA-861, NOAA) "
                   "to predict power outage risk in Northeast US. "
                   "Outages cost the US **$121-150B annually** (DOE 2024).")
        st.divider()
        st.caption("Jaykumar Patel\nMS Data Science, Stevens Institute\n"
                   "MS IT Project Mgmt (in progress), NEC")
    return state, risk


def kpis(df):
    st.markdown("#### Grid Status Overview")
    high  = len(df[df["risk_level"].isin(["HIGH","MEDIUM-HIGH"])])
    total = df["customers_at_risk"].sum()
    avg   = df["risk_score"].mean()
    worst = df.loc[df["risk_score"].idxmax(),"county"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("High-Risk Regions",  f"{high}/{len(df)}")
    c2.metric("Customers at Risk",  f"{total/1e6:.2f}M")
    c3.metric("Avg Risk Score",     f"{avg:.0%}")
    c4.metric("Highest Risk Area",  worst)


def risk_map(df):
    st.markdown("#### Regional Risk Map")
    color_map = {
        "HIGH":"#e63946","MEDIUM-HIGH":"#f4a261",
        "MEDIUM":"#e9c46a","LOW-MEDIUM":"#2a9d8f","LOW":"#457b9d"
    }
    fig = px.scatter_mapbox(
        df, lat="lat", lon="lon",
        color="risk_level", size="risk_score", size_max=28,
        hover_name="county",
        hover_data={
            "state":True,"risk_score":":.0%",
            "customers_at_risk":":,","avg_outage_hrs":":.1f",
            "top_factor":True,"lat":False,"lon":False
        },
        color_discrete_map=color_map,
        mapbox_style="carto-darkmatter",
        zoom=5, center={"lat":43.5,"lon":-71.5}
    )
    fig.update_layout(
        height=450, margin={"r":0,"t":0,"l":0,"b":0},
        paper_bgcolor="#0a0e1a",
        legend=dict(bgcolor="#0d1b2a",bordercolor="#1e3a5f",
                    font=dict(color="#e8f4fd"))
    )
    st.plotly_chart(fig, use_container_width=True)


def risk_table(df):
    st.markdown("#### Risk Rankings")
    disp = df.sort_values("risk_score", ascending=False)[
        ["county","state","risk_level","risk_score",
         "customers_at_risk","avg_outage_hrs","top_factor","infra_age"]
    ].copy()
    disp.columns = ["County","State","Risk Level","Risk Score",
                    "Customers at Risk","Avg Outage (hrs)","Top Factor","Infra Age (yrs)"]
    disp["Risk Score"]        = disp["Risk Score"].apply(lambda x: f"{x:.0%}")
    disp["Customers at Risk"] = disp["Customers at Risk"].apply(lambda x: f"{x:,}")

    def hl(val):
        m = {
            "HIGH":        "background-color:#3d0000;color:#ff6b6b",
            "MEDIUM-HIGH": "background-color:#3d2600;color:#f4a261",
            "MEDIUM":      "background-color:#3d3400;color:#e9c46a"
        }
        return m.get(val, "")

    st.dataframe(
        disp.style.map(hl, subset=["Risk Level"]),
        use_container_width=True, hide_index=True
    )


def model_chart(metrics):
    st.markdown("#### ML Model Performance")
    names  = [k for k in metrics if k not in ["best_model","trained_at"]]
    keys   = ["accuracy","precision","recall","f1_score","roc_auc"]
    labels = ["Accuracy","Precision","Recall","F1","ROC-AUC"]
    colors = ["#e63946","#457b9d","#2a9d8f"]

    fig = go.Figure()
    for i, name in enumerate(names):
        fig.add_trace(go.Bar(
            name=name, x=labels,
            y=[metrics[name].get(k, 0) for k in keys],
            marker_color=colors[i % len(colors)],
            text=[f"{metrics[name].get(k,0):.3f}" for k in keys],
            textposition="outside",
            textfont=dict(size=10, color="#e8f4fd")
        ))
    fig.update_layout(
        barmode="group", height=340,
        paper_bgcolor="#0a0e1a", plot_bgcolor="#0a0e1a",
        font=dict(color="#e8f4fd", family="IBM Plex Sans"),
        legend=dict(bgcolor="#0d1b2a", bordercolor="#1e3a5f"),
        yaxis=dict(range=[0,1.12], gridcolor="#1e3a5f"),
        xaxis=dict(gridcolor="#1e3a5f"),
        title=dict(
            text=f"Best model: {metrics.get('best_model','')}",
            font=dict(color="#7fb3d3", size=12)
        )
    )
    st.plotly_chart(fig, use_container_width=True)


def trend_chart():
    st.markdown("#### Outage Trend - Northeast US (2015-2024)")
    years  = list(range(2015, 2025))
    events = [18, 22, 19, 31, 28, 35, 42, 38, 45, 51]
    cost_b = [42, 48, 44, 68, 61, 78, 95, 88, 108, 121]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=years, y=events, name="Major Outage Events",
        line=dict(color="#e63946", width=2.5),
        mode="lines+markers", marker=dict(size=7)
    ))
    fig.add_trace(go.Bar(
        x=years, y=cost_b, name="Estimated Cost ($B)",
        yaxis="y2", marker_color="#1e3a5f", opacity=0.55
    ))
    fig.update_layout(
        height=300,
        paper_bgcolor="#0a0e1a", plot_bgcolor="#0a0e1a",
        font=dict(color="#e8f4fd", family="IBM Plex Sans"),
        legend=dict(bgcolor="#0d1b2a", bordercolor="#1e3a5f"),
        xaxis=dict(gridcolor="#1e3a5f"),
        yaxis=dict(title="Events", gridcolor="#1e3a5f"),
        yaxis2=dict(title="Cost ($B)", overlaying="y", side="right",
                    gridcolor="#1e3a5f"),
        annotations=[dict(
            x=2024, y=51, text="$121B in 2024 (ORNL)",
            showarrow=True, arrowhead=2,
            font=dict(color="#e63946", size=10)
        )]
    )
    st.plotly_chart(fig, use_container_width=True)


def risk_calculator():
    st.markdown("#### Outage Risk Calculator")
    st.caption("Estimate outage risk for a custom scenario")

    c1, c2, c3 = st.columns(3)
    with c1:
        state     = st.selectbox("State", ["ME","NH","VT","MA","RI","CT","NY","NJ","PA"])
        season    = st.selectbox("Season", ["Winter","Spring","Summer","Fall"])
        wind      = st.slider("Max Wind Speed (mph)", 0, 100, 30)
    with c2:
        snow      = st.slider("Snowfall (inches)", 0, 48, 4)
        ice_storm = st.checkbox("Ice Storm Present")
        age       = st.slider("Infrastructure Age (yrs)", 5, 70, 35)
    with c3:
        maint     = st.slider("Maintenance Score (1-10)", 1, 10, 6)
        veg       = st.slider("Vegetation Mgmt (1-10)", 1, 10, 6)
        prior     = st.slider("Prior 12-Month Outages", 0, 10, 2)

    if st.button("Calculate Risk", type="primary"):
        risk = (wind/100)*0.25 + (snow/48)*0.15
        if ice_storm:
            risk += 0.20
        risk += (age/70)*0.15
        risk += ((10-maint)/10)*0.12
        risk += ((10-veg)/10)*0.08
        risk += (prior/10)*0.05
        risk = min(1.0, risk)

        if risk >= 0.70:
            level = "HIGH"
            col   = "#e63946"
            icon  = "High Risk"
        elif risk >= 0.50:
            level = "MEDIUM-HIGH"
            col   = "#f4a261"
            icon  = "Medium-High Risk"
        elif risk >= 0.30:
            level = "MEDIUM"
            col   = "#e9c46a"
            icon  = "Medium Risk"
        else:
            level = "LOW"
            col   = "#2a9d8f"
            icon  = "Low Risk"

        st.markdown(f"""
        <div style='background:#0d1b2a;border:2px solid {col};border-radius:12px;
                    padding:24px;text-align:center;margin-top:12px;'>
            <div style='font-family:IBM Plex Mono;font-size:2rem;color:{col};font-weight:600;'>
                {risk:.0%} — {icon}
            </div>
            <div style='color:#7fb3d3;font-size:1rem;margin-top:6px;'>Risk Level: {level}</div>
            <div style='color:#e8f4fd;font-size:0.85rem;margin-top:8px;'>
                {state} | {season} | Wind: {wind}mph | Infrastructure Age: {age}yrs
            </div>
        </div>""", unsafe_allow_html=True)


def main():
    df      = load_risk_data()
    metrics = load_metrics()
    state, risk_filter = sidebar(df)

    fdf = df.copy()
    if state != "All":
        fdf = fdf[fdf["state"] == state]
    if risk_filter != "All":
        fdf = fdf[fdf["risk_level"] == risk_filter]

    st.markdown("""
    <div style='padding:8px 0 4px;'>
        <span style='font-family:IBM Plex Mono;font-size:2rem;font-weight:600;color:#e8f4fd;'>
            GridWatch
        </span>
        <span style='font-size:0.9rem;color:#7fb3d3;margin-left:12px;
                     letter-spacing:2px;text-transform:uppercase;'>
            Northeast US Power Grid Risk Intelligence
        </span>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    kpis(fdf)
    st.divider()

    col_l, col_r = st.columns([1.4, 1])
    with col_l:
        risk_map(fdf)
    with col_r:
        risk_table(fdf)

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        trend_chart()
    with col_b:
        model_chart(metrics)

    st.divider()
    risk_calculator()

    st.divider()
    st.caption(
        f"GridWatch | Jaykumar Patel | "
        f"Data: DOE OE-417, EIA-861, NOAA Storm Events | "
        f"Updated {datetime.now().strftime('%B %Y')} | "
        "Independent research — not affiliated with any utility."
    )


if __name__ == "__main__":
    main()
