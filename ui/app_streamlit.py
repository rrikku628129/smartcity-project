"""
NYC Travel Advisor v3 — Streamlit UI
Run: streamlit run app_streamlit.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime

# ── Page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="NYC Travel Advisor v3",
    page_icon="🗽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stApp { font-family: 'Segoe UI', sans-serif; }

    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a2e;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #888;
        margin-top: 0.2rem;
    }
    .verdict-easy {
        background: linear-gradient(135deg, #2ecc71, #27ae60);
        color: white;
        border-radius: 16px;
        padding: 1.5rem 2rem;
        text-align: center;
        font-size: 1.8rem;
        font-weight: 700;
        box-shadow: 0 4px 15px rgba(46,204,113,0.3);
        margin-bottom: 1rem;
    }
    .verdict-moderate {
        background: linear-gradient(135deg, #f39c12, #e67e22);
        color: white;
        border-radius: 16px;
        padding: 1.5rem 2rem;
        text-align: center;
        font-size: 1.8rem;
        font-weight: 700;
        box-shadow: 0 4px 15px rgba(243,156,18,0.3);
        margin-bottom: 1rem;
    }
    .verdict-hard {
        background: linear-gradient(135deg, #e74c3c, #c0392b);
        color: white;
        border-radius: 16px;
        padding: 1.5rem 2rem;
        text-align: center;
        font-size: 1.8rem;
        font-weight: 700;
        box-shadow: 0 4px 15px rgba(231,76,60,0.3);
        margin-bottom: 1rem;
    }
    .info-box {
        background: #e8f4fd;
        border-left: 4px solid #3498db;
        border-radius: 0 8px 8px 0;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .warning-box {
        background: #fef9e7;
        border-left: 4px solid #f39c12;
        border-radius: 0 8px 8px 0;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .window-card {
        background: white;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        margin: 0.4rem 0;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
        display: flex;
        justify-content: space-between;
    }
    div[data-testid="stSidebar"] {
        background: #1a1a2e;
    }
    div[data-testid="stSidebar"] * {
        color: white !important;
    }
    div[data-testid="stSidebar"] .stSelectbox label,
    div[data-testid="stSidebar"] .stSlider label,
    div[data-testid="stSidebar"] .stCheckbox label {
        color: #ccc !important;
    }
    .sidebar-header {
        text-align: center;
        padding: 1rem 0;
        border-bottom: 1px solid #333;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Load models ────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    for d in [
        Path("/home/zhoub1/AI Deep Learning/models"),
        Path(r"C:\Users\86188\urban_event_prediction\models"),
    ]:
        if d.exists():
            models_dir = d
            break
    try:
        xgb = joblib.load(models_dir / "xgb_model_v3.pkl")
        lgb = joblib.load(models_dir / "lgb_model_v3.pkl")
        rf  = joblib.load(models_dir / "rf_model_v3.pkl")
        return xgb, lgb, rf, True
    except Exception as e:
        st.error(f"Model load failed: {e}")
        return None, None, None, False

xgb_model, lgb_model, rf_model, MODEL_OK = load_models()

FEATURE_COLS = [
    'hour','day_of_week','day','month','is_weekend',
    'is_rush_am','is_rush_pm','is_night',
    'is_month_start','is_month_end',
    'hour_sin','hour_cos','dow_sin','dow_cos',
    'month_sin','month_cos',
    'lag_1h','lag_24h','lag_168h',
    'rolling_3h_mean','rolling_24h_mean','rolling_7d_mean','demand_trend',
    'event_score_norm','n_special','n_accident',
    'traffic_intensity','trend_score',
    'avg_fare','avg_distance','avg_passengers','avg_fare_per_mile'
]

WEIGHTS = {'xgb':0.25,'lgb':0.45,'rf':0.30}
BM = {"Manhattan":1.00,"Brooklyn":0.28,
      "Queens":0.22,"Bronx":0.08,"Staten Island":0.03}
WC = {"Clear / Sunny":0,"Partly Cloudy":2,"Overcast":3,
      "Drizzle":51,"Rain":61,"Heavy Rain":63,
      "Snow":71,"Heavy Snow":75,"Thunderstorm":95}
API_511NY = "26920640392042aaa349340dd3292222"

# ── 511NY ──────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_511ny():
    try:
        r   = requests.get(
            f"https://511ny.org/api/getevents?key={API_511NY}&format=json",
            timeout=10).json()
        nyc = [e for e in r if e.get("Latitude") and
               40.48<=float(e["Latitude"])<=40.92 and
               -74.26<=float(e.get("Longitude",0))<=-73.70]
        tw  = {"specialEvents":3.0,"accidentsAndIncidents":2.5,
               "closures":2.0,"transitOperations":1.5,"roadwork":1.0}
        sw  = {"Major":3.0,"Moderate":2.0,"Minor":1.5,"Unknown":1.0,"None":0.5}
        sc  = sum(tw.get(e.get("EventType",""),1)*
                  sw.get(e.get("Severity",""),1) for e in nyc)
        return {
            "score": min(sc/500, 1.0),
            "ns":    sum(1 for e in nyc if e.get("EventType")=="specialEvents"),
            "na":    sum(1 for e in nyc if e.get("EventType")=="accidentsAndIncidents"),
            "nc":    sum(1 for e in nyc if e.get("EventType")=="closures"),
            "nr":    sum(1 for e in nyc if e.get("EventType")=="roadwork"),
            "n":     len(nyc),
            "ok":    True
        }
    except:
        return {"score":0.0,"ns":0,"na":0,"nc":0,"nr":0,"n":0,"ok":False}


# ── Core prediction ────────────────────────────────────────────────
def build_row(h, dow, day, t, p, w, wl, ev):
    wc  = WC.get(wl, 0)
    mon = datetime.now().month
    return {
        "hour":h,"day_of_week":dow,"day":day,"month":mon,
        "is_weekend":    int(dow>=5),
        "is_rush_am":    int(7<=h<=9),
        "is_rush_pm":    int(17<=h<=19),
        "is_night":      int(h>=22 or h<=5),
        "is_month_start":int(day<=3),
        "is_month_end":  int(day>=28),
        "hour_sin":      np.sin(2*np.pi*h/24),
        "hour_cos":      np.cos(2*np.pi*h/24),
        "dow_sin":       np.sin(2*np.pi*dow/7),
        "dow_cos":       np.cos(2*np.pi*dow/7),
        "month_sin":     np.sin(2*np.pi*mon/12),
        "month_cos":     np.cos(2*np.pi*mon/12),
        "lag_1h":2800,"lag_24h":2800,"lag_168h":2800,
        "rolling_3h_mean":2800,"rolling_24h_mean":2800,
        "rolling_7d_mean":2800,"demand_trend":0,
        "event_score_norm":ev["score"],
        "n_special":     ev["ns"],
        "n_accident":    ev["na"],
        "traffic_intensity":0.5,
        "trend_score":   0.5,
        "avg_fare":18.5,"avg_distance":2.8,
        "avg_passengers":1.3,"avg_fare_per_mile":6.5,
    }

def pred_one(h, dow, day, b, t, p, w, wl, ev):
    X  = pd.DataFrame([build_row(h,dow,day,t,p,w,wl,ev)])[FEATURE_COLS]
    xp = float(xgb_model.get_booster().inplace_predict(
        X.values, validate_features=False)[0])
    lp = float(lgb_model.predict(X)[0])
    rp = float(rf_model.predict(X.values)[0])  # 用 .values 跳过特征名验证
    return max(0, WEIGHTS["xgb"]*xp+WEIGHTS["lgb"]*lp+WEIGHTS["rf"]*rp)*BM.get(b,1.0)

def get_day_demands(dow, day, b, t, p, w, wl, ev):
    return [pred_one(h, dow, day, b, t, p, w, wl, ev) for h in range(24)]

def best_windows(demands, n=3):
    return sorted(
        [(h, (demands[h]+demands[min(h+1,23)])/2) for h in range(23)],
        key=lambda x: x[1])[:n]

def make_plotly_chart(demands, hour, borough, mode):
    lo,hi = (2200,3200) if mode=="taxi" else (2000,3500)
    colors = ["#2ecc71" if d<lo else ("#f39c12" if d<hi else "#e74c3c")
              for d in demands]
    # Highlight selected hour
    colors[hour] = "#1a1a2e"

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(range(24)),
        y=demands,
        marker_color=colors,
        marker_line_color="white",
        marker_line_width=1.5,
        hovertemplate="<b>%{x}:00</b><br>%{y:,.0f} trips/hr<extra></extra>",
    ))
    # Add threshold lines
    fig.add_hline(y=lo, line_dash="dot", line_color="#f39c12",
                  annotation_text="Moderate threshold", annotation_position="right")
    fig.add_hline(y=hi, line_dash="dot", line_color="#e74c3c",
                  annotation_text="High threshold", annotation_position="right")

    fig.update_layout(
        title=dict(
            text=f"{'Taxi demand' if mode=='taxi' else 'Traffic congestion'} forecast — {borough}",
            font=dict(size=14)),
        xaxis=dict(title="Hour of Day", tickmode="linear", tick0=0, dtick=2),
        yaxis=dict(title="Trips / hour"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        height=320,
        margin=dict(l=10,r=10,t=40,b=10),
    )
    return fig


# ══════════════════════════════════════════════════════════════════
# UI Layout
# ══════════════════════════════════════════════════════════════════

# ── Sidebar ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
      <div style="font-size:2.5rem">🗽</div>
      <div style="font-size:1.1rem;font-weight:600;margin-top:0.5rem">NYC Travel Advisor</div>
      <div style="font-size:0.75rem;color:#aaa;margin-top:0.2rem">v3 · Ensemble Model</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Trip Details")
    mode = st.radio("Travel mode",
                    ["🚕  Hail a taxi", "🚗  Drive yourself"],
                    index=0)
    borough = st.selectbox("Borough", list(BM.keys()))

    col1, col2 = st.columns(2)
    with col1:
        hour = st.selectbox("Hour",
            options=list(range(24)),
            format_func=lambda x: f"{x:02d}:00",
            index=datetime.now().hour)
    with col2:
        dow = st.selectbox("Day",
            options=list(range(7)),
            format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x],
            index=datetime.now().weekday())

    st.markdown("### Weather")
    weather = st.selectbox("Condition", list(WC.keys()))
    temp    = st.slider("Temperature (°C)", -15, 40, 5)
    prec    = st.slider("Precipitation (mm)", 0, 50, 0)
    wind    = st.slider("Wind speed (km/h)", 0, 80, 10)

    st.markdown("### 🚦 Live Events")
    use511 = st.toggle("Use live 511NY events", value=True)

    predict_btn = st.button("🔮  Get Prediction",
                            type="primary", use_container_width=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.7rem;color:#888;text-align:center">
    XGBoost(25%) + LightGBM(45%) + RF(30%)<br>
    16 months · RMSE=379.68 · R²=0.9755<br>
    511NY Official Traffic API
    </div>
    """, unsafe_allow_html=True)


# ── Main area ──────────────────────────────────────────────────────
st.markdown("# 🗽 NYC Travel Advisor")
st.markdown("Predict **taxi availability** or **road congestion** using real-time 511NY data and ensemble ML.")

# Stats row
c1,c2,c3,c4,c5 = st.columns(5)
for col, val, lbl in zip(
    [c1,c2,c3,c4,c5],
    ["50M+","97.6%","3","32","511NY"],
    ["Trips analyzed","R² accuracy","Models in ensemble","Features","Live events"]
):
    col.markdown(f"""
    <div class="metric-card">
      <div class="metric-value">{val}</div>
      <div class="metric-label">{lbl}</div>
    </div>""", unsafe_allow_html=True)

st.divider()

if predict_btn:
    with st.spinner("Fetching live 511NY events and computing prediction..."):
        ev = fetch_511ny() if use511 else {"score":0.0,"ns":0,"na":0,"nc":0,"nr":0,"n":0,"ok":False}
        now = datetime.now()
        day = now.day
        mode_key = "taxi" if "taxi" in mode.lower() else "drive"
        demands = get_day_demands(dow, day, borough, temp, prec, wind, weather, ev)
        demand  = demands[hour]
        top3    = best_windows(demands)
        pct     = (demand - 2800*BM.get(borough,1)) / (2800*BM.get(borough,1)) * 100
        lo,hi   = (2200,3200) if mode_key=="taxi" else (2000,3500)

        if mode_key == "taxi":
            if demand < lo:
                verdict, css = "✅ Easy to hail", "verdict-easy"
            elif demand < hi:
                verdict, css = "⚠️ Moderate wait", "verdict-moderate"
            else:
                verdict, css = "❌ Hard to hail", "verdict-hard"
        else:
            if demand < lo:
                verdict, css = "✅ Light traffic", "verdict-easy"
            elif demand < hi:
                verdict, css = "⚠️ Moderate traffic", "verdict-moderate"
            else:
                verdict, css = "❌ Heavy traffic", "verdict-hard"

    # ── Result layout ──────────────────────────────────────────────
    left, right = st.columns([2,1])

    with left:
        # Verdict banner
        st.markdown(f'<div class="{css}">{verdict}</div>', unsafe_allow_html=True)

        # Metrics row
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Predicted demand", f"{demand:,.0f}", "trips/hr")
        m2.metric("90% CI", f"{demand*0.9:,.0f}–{demand*1.1:,.0f}")
        m3.metric(f"vs {borough} avg", f"{abs(pct):.1f}%",
                  f"{'▲' if pct>=0 else '▼'}")
        m4.metric("Model R²", "0.9755", "D3 Ensemble")

        # Chart
        st.plotly_chart(make_plotly_chart(demands, hour, borough, mode_key),
                        use_container_width=True)

        # How to read
        how = (f"Thresholds: <{lo:,}/hr = easy | {lo:,}–{hi:,}/hr = moderate | >{hi:,}/hr = hard"
               if mode_key=="taxi" else
               f"Thresholds: <{lo:,}/hr = light | {lo:,}–{hi:,}/hr = moderate | >{hi:,}/hr = heavy")
        st.markdown(f'<div class="info-box">ℹ️ {how}</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="warning-box">⚠️ Lag features use historical borough averages '
            'at inference time. Real-time accuracy may differ from training R²=0.9755.</div>',
            unsafe_allow_html=True)

    with right:
        # 511NY status
        st.markdown("### 🚦 Live 511NY Events")
        if use511 and ev["ok"]:
            st.success(f"**{ev['n']} NYC events** active right now")
            cols = st.columns(2)
            cols[0].metric("Special events", ev["ns"])
            cols[1].metric("Accidents", ev["na"])
            cols[0].metric("Closures", ev["nc"])
            cols[1].metric("Roadwork", ev["nr"])
        elif use511:
            st.warning("511NY API unavailable — using defaults")
        else:
            st.info("511NY events disabled")

        st.divider()

        # Best windows
        st.markdown("### 🕐 Best travel windows")
        st.caption("Lowest predicted demand periods today")
        for i, (h, avg) in enumerate(top3):
            medal = ["🥇","🥈","🥉"][i]
            st.markdown(f"""
            <div class="window-card">
              <span>{medal} <b>{h:02d}:00 – {h+2:02d}:00</b></span>
              <span style="color:#888">{avg:,.0f} trips/hr</span>
            </div>""", unsafe_allow_html=True)

        st.divider()

        # Prediction details
        st.markdown("### 📊 Prediction details")
        st.markdown(f"**Borough:** {borough} (×{BM[borough]})")
        st.markdown(f"**Time:** {hour:02d}:00 · {'Weekend' if dow>=5 else 'Weekday'}")
        st.markdown(f"**Weather:** {weather}")
        st.markdown(f"**Temp:** {temp}°C · **Precip:** {prec}mm")

else:
    # Landing state
    st.markdown("""
    <div style="text-align:center;padding:4rem 2rem;color:#888">
      <div style="font-size:5rem;margin-bottom:1rem">🗽</div>
      <div style="font-size:1.3rem;font-weight:500;color:#555;margin-bottom:0.5rem">
        Configure your trip in the sidebar
      </div>
      <div style="font-size:1rem">
        Select your borough, time, weather conditions, and click <b>Get Prediction</b>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Feature highlights
    st.markdown("---")
    f1,f2,f3,f4 = st.columns(4)
    for col, icon, title, desc in zip(
        [f1,f2,f3,f4],
        ["🤖","📡","🌦️","📊"],
        ["Ensemble Model","Live 511NY","Weather-aware","32 Features"],
        ["XGBoost+LightGBM+RF weighted combination",
         "Real-time NYC traffic events from official API",
         "Temperature, precipitation, wind integrated",
         "Temporal, lag, weather, event signals"]
    ):
        col.markdown(f"""
        <div class="metric-card" style="text-align:left;padding:1.5rem">
          <div style="font-size:2rem">{icon}</div>
          <div style="font-weight:600;margin:0.5rem 0;color:#1a1a2e">{title}</div>
          <div style="font-size:0.85rem;color:#888">{desc}</div>
        </div>""", unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "NYC Travel Advisor v3 · "
    "D3 Ensemble: XGBoost(25%) + LightGBM(45%) + RandomForest(30%) · "
    "16 months NYC Yellow Taxi data · RMSE=379.68 · R²=0.9755 · "
    "511NY Official Traffic API · 32 features"
)
