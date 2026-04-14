"""
ui/app.py — NYC Travel Advisor
Conversational UI: user chooses taxi or drive, inputs trip details,
gets prediction on taxi availability OR traffic congestion.
"""

import gradio as gr
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────
MODELS_DIR = Path(r"C:\Users\86188\urban_event_prediction\models")
try:
    model        = joblib.load(MODELS_DIR / "xgb_demand_model.pkl")
    feature_cols = joblib.load(MODELS_DIR / "feature_cols.pkl")
    MODEL_LOADED = True
    FEATURE_IMPORTANCES = {f: float(i) for f, i in
                           zip(feature_cols, model.feature_importances_)}
    print(f"✅ Model loaded — {len(feature_cols)} features")
except Exception as e:
    MODEL_LOADED = False
    model = None
    feature_cols = []
    FEATURE_IMPORTANCES = {}
    print(f"⚠️  Model not found: {e}")

# ── Constants ──────────────────────────────────────────────────────
LAG_DEFAULTS = {
    "lag_1h": 2800, "lag_24h": 2800, "lag_168h": 2800,
    "rolling_3h_mean": 2800, "rolling_24h_mean": 2800,
    "avg_fare": 18.5, "avg_distance": 2.8, "avg_passengers": 1.3,
}
WEATHER_CODE_MAP = {
    "Clear / Sunny": 0, "Partly Cloudy": 2, "Overcast": 3,
    "Drizzle": 51, "Rain": 61, "Heavy Rain": 63,
    "Snow": 71, "Heavy Snow": 75, "Thunderstorm": 95,
}
BOROUGH_MULTIPLIER = {
    "Manhattan": 1.00, "Brooklyn": 0.28,
    "Queens": 0.22, "Bronx": 0.08, "Staten Island": 0.03,
}

# Taxi availability thresholds (trips/hour for full Manhattan)
# Scaled by borough multiplier when used
TAXI_THRESHOLDS = {
    "Easy to hail"   : (0,    2200),
    "Moderate wait"  : (2200, 3200),
    "Hard to hail"   : (3200, float("inf")),
}
# Congestion thresholds (same demand proxy, different labels)
CONGESTION_THRESHOLDS = {
    "Light traffic"  : (0,    2000),
    "Moderate traffic": (2000, 3500),
    "Heavy traffic"  : (3500, float("inf")),
}

def build_row(hour, dow, temperature, precipitation, windspeed,
              weather_label, has_event, sentiment=0.0):
    wc = WEATHER_CODE_MAP.get(weather_label, 0)
    return {
        "hour": hour, "day_of_week": dow, "day": 15,
        "is_weekend": int(dow >= 5),
        "is_rush_am": int(7 <= hour <= 9),
        "is_rush_pm": int(17 <= hour <= 19),
        "is_night": int(hour >= 22 or hour <= 5),
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "dow_sin": np.sin(2 * np.pi * dow / 7),
        "dow_cos": np.cos(2 * np.pi * dow / 7),
        "temperature_c": temperature,
        "precipitation_mm": precipitation,
        "windspeed_kmh": windspeed,
        "is_raining": int(precipitation > 0.5),
        "is_snowing": int(wc in [71, 73, 75, 77]),
        "is_bad_weather": int(precipitation > 0.5 or wc in [71,73,75,77]),
        "has_event": int(has_event),
        "sentiment_score": sentiment,
        **LAG_DEFAULTS,
    }

def get_prediction(hour, dow, borough, temperature,
                   precipitation, windspeed, weather_label, has_event):
    """Return raw predicted trips/hour for given conditions."""
    row   = build_row(hour, dow, temperature, precipitation,
                      windspeed, weather_label, has_event)
    X_in  = pd.DataFrame([row])
    avail = [c for c in feature_cols if c in X_in.columns]
    base  = float(model.predict(X_in[avail])[0])
    mult  = BOROUGH_MULTIPLIER.get(borough, 1.0)
    return max(0, base * mult), mult

def classify_taxi(demand):
    for label, (lo, hi) in TAXI_THRESHOLDS.items():
        if lo <= demand < hi:
            return label
    return "Hard to hail"

def classify_congestion(demand):
    for label, (lo, hi) in CONGESTION_THRESHOLDS.items():
        if lo <= demand < hi:
            return label
    return "Heavy traffic"

def make_forecast_chart(hour, dow, borough, temperature,
                        precipitation, windspeed, weather_label,
                        has_event, mode):
    """Generate 24-hour bar chart coloured by severity."""
    hours, demands = [], []
    mult = BOROUGH_MULTIPLIER.get(borough, 1.0)
    for h in range(24):
        row  = build_row(h, dow, temperature, precipitation,
                         windspeed, weather_label, has_event)
        X_in = pd.DataFrame([row])
        avail= [c for c in feature_cols if c in X_in.columns]
        d    = max(0, float(model.predict(X_in[avail])[0]) * mult)
        hours.append(h)
        demands.append(d)

    if mode == "taxi":
        def color(d):
            if d < 2200: return "#378ADD"
            if d < 3200: return "#EF9F27"
            return "#E24B4A"
        title  = f"Hourly taxi demand — {borough}"
        ylabel = "Predicted trips/hour"
        legend = ["Easy (<2200)", "Moderate (2200–3200)", "Hard (>3200)"]
        lcolors= ["#378ADD", "#EF9F27", "#E24B4A"]
    else:
        def color(d):
            if d < 2000: return "#378ADD"
            if d < 3500: return "#EF9F27"
            return "#E24B4A"
        title  = f"Traffic congestion proxy — {borough}"
        ylabel = "Taxi volume proxy (trips/hour)"
        legend = ["Light (<2000)", "Moderate (2000–3500)", "Heavy (>3500)"]
        lcolors= ["#378ADD", "#EF9F27", "#E24B4A"]

    fig, ax = plt.subplots(figsize=(10, 3.5))
    bars = ax.bar(hours, demands,
                  color=[color(d) for d in demands],
                  edgecolor="white", width=0.75)
    ax.axvline(hour - 0.5 + 0.375, color="#2C2C2A",
               linestyle="--", linewidth=1.5, label=f"Selected: {hour:02d}:00")

    # Legend patches
    from matplotlib.patches import Patch
    handles = [Patch(color=c, label=l) for c, l in zip(lcolors, legend)]
    ax.legend(handles=handles, fontsize=8, loc="upper left")

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(0, 24, 2))
    ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    return fig

def find_best_window(demands, mode):
    """Find the 2-hour window with lowest demand."""
    min_avg = float("inf")
    best_h  = 0
    for h in range(22):
        avg = (demands[h] + demands[h+1]) / 2
        if avg < min_avg:
            min_avg = avg
            best_h  = h
    return best_h, min_avg


def predict(mode, borough, hour, dow,
            temperature, precipitation, windspeed,
            weather_condition, has_event):

    if not MODEL_LOADED:
        return "⚠️ Model not loaded. Run model_training.ipynb first.", None

    demand, mult = get_prediction(hour, dow, borough, temperature,
                                  precipitation, windspeed,
                                  weather_condition, has_event)
    ci_low  = demand * 0.92
    ci_high = demand * 1.08
    avg_demand = 2800 * mult
    pct = (demand - avg_demand) / avg_demand * 100

    # Pre-compute full day for best-window tip
    day_demands = []
    for h in range(24):
        d, _ = get_prediction(h, dow, borough, temperature,
                              precipitation, windspeed,
                              weather_condition, has_event)
        day_demands.append(d)
    best_h, best_d = find_best_window(day_demands, mode)

    if mode == "Hail a taxi":
        verdict = classify_taxi(demand)
        verdict_emoji = {"Easy to hail": "✅", "Moderate wait": "⚠️",
                         "Hard to hail": "❌"}[verdict]
        color_word = {"Easy to hail": "green", "Moderate wait": "orange",
                      "Hard to hail": "red"}[verdict]

        tip = (f"Best window to hail: **{best_h:02d}:00–{best_h+2:02d}:00** "
               f"(demand drops to ~{best_d:,.0f} trips/hr)"
               if verdict != "Easy to hail" else
               f"Right now is a good time — demand is below average.")

        result = f"""
## {verdict_emoji} {verdict}

**Predicted taxi demand:** {demand:,.0f} trips/hour
**95% CI:** {ci_low:,.0f} – {ci_high:,.0f} &nbsp;|&nbsp; {'▲' if pct>=0 else '▼'} {abs(pct):.1f}% vs {borough} average

---
**How to read this:**
High demand = more people competing for taxis = longer wait.
Low demand = taxis available = easy to hail.

**Tip:** {tip}

📍 {borough} &nbsp;|&nbsp; 🕐 {hour:02d}:00 &nbsp;|&nbsp; {'🗓️ Weekend' if dow >= 5 else '💼 Weekday'}
"""
        chart_mode = "taxi"

    else:  # Drive yourself
        verdict = classify_congestion(demand)
        verdict_emoji = {"Light traffic": "✅", "Moderate traffic": "⚠️",
                         "Heavy traffic": "❌"}[verdict]

        tip = (f"Suggested departure: **{best_h:02d}:00–{best_h+2:02d}:00** "
               f"(traffic proxy drops to ~{best_d:,.0f})"
               if verdict != "Light traffic" else
               f"Roads should be relatively clear at this time.")

        result = f"""
## {verdict_emoji} {verdict}

**Traffic proxy (taxi volume):** {demand:,.0f} trips/hour
**95% CI:** {ci_low:,.0f} – {ci_high:,.0f} &nbsp;|&nbsp; {'▲' if pct>=0 else '▼'} {abs(pct):.1f}% vs {borough} average

---
**How to read this:**
Taxi volume is used as a proxy for road traffic — when many people hail taxis,
overall vehicle volume on roads tends to be high too.

**Thresholds:** &lt;2,000/hr = light &nbsp;|&nbsp; 2,000–3,500/hr = moderate &nbsp;|&nbsp; &gt;3,500/hr = heavy

**Tip:** {tip}

📍 {borough} &nbsp;|&nbsp; 🕐 {hour:02d}:00 &nbsp;|&nbsp; {'🗓️ Weekend' if dow >= 5 else '💼 Weekday'}
"""
        chart_mode = "drive"

    chart = make_forecast_chart(hour, dow, borough, temperature,
                                precipitation, windspeed,
                                weather_condition, has_event, chart_mode)
    return result, chart


# ── Gradio UI ──────────────────────────────────────────────────────
with gr.Blocks(title="NYC Travel Advisor", theme=gr.themes.Soft()) as demo:

    gr.Markdown(
        "# 🗽 NYC Travel Advisor\n"
        "Tell us how you want to travel — we'll predict taxi availability or road congestion."
    )

    # Step 1 — mode
    gr.Markdown("### Step 1 — How are you traveling?")
    mode = gr.Radio(
        choices=["Hail a taxi", "Drive yourself"],
        value="Hail a taxi",
        label="Travel mode"
    )

    # Step 2 — trip details
    gr.Markdown("### Step 2 — Trip details")
    with gr.Row():
        borough = gr.Dropdown(
            choices=list(BOROUGH_MULTIPLIER.keys()),
            value="Manhattan", label="📍 Borough / Area"
        )
        hour = gr.Slider(0, 23, value=18, step=1, label="🕐 Hour of departure")
        dow  = gr.Slider(0, 6,  value=2,  step=1, label="📅 Day of week (0=Mon · 6=Sun)")

    gr.Markdown("#### Weather conditions")
    with gr.Row():
        weather_condition = gr.Dropdown(
            choices=list(WEATHER_CODE_MAP.keys()),
            value="Clear / Sunny", label="Weather"
        )
        temperature   = gr.Slider(-15, 40, value=5,  step=0.5, label="Temperature (°C)")
        precipitation = gr.Slider(0,   50, value=0,  step=0.5, label="Precipitation (mm)")
        windspeed     = gr.Slider(0,   80, value=10, step=1,   label="Wind speed (km/h)")

    has_event = gr.Checkbox(label="Major public event today (concert, parade, etc.)", value=False)

    predict_btn = gr.Button("🔮  Get prediction", variant="primary", size="lg")

    # Step 3 — results
    gr.Markdown("### Step 3 — Prediction")
    result_text = gr.Markdown("*Fill in the details above and click **Get prediction**.*")
    result_chart = gr.Plot(label="24-hour forecast")

    predict_btn.click(
        fn=predict,
        inputs=[mode, borough, hour, dow,
                temperature, precipitation, windspeed,
                weather_condition, has_event],
        outputs=[result_text, result_chart]
    )

    gr.Markdown(
        "---\n"
        "*Model: XGBoost trained on NYC Yellow Taxi data (Jan 2026) + "
        "weather + social sentiment + city events. "
        "Congestion is inferred from taxi demand volume as a proxy.*"
    )

if __name__ == "__main__":
    demo.launch(share=False, server_port=7860)
