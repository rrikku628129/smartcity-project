"""
ui/app.py  —  Multimodal Urban Event Prediction
Gradio interface for interactive taxi demand forecasting.

Run:
    python ui/app.py
Then open http://localhost:7860 in your browser.
"""

import gradio as gr
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── Paths (absolute — safe on Windows) ────────────────────────────
MODELS_DIR = Path(r"C:\Users\86188\urban_event_prediction\models")
MODEL_PATH = MODELS_DIR / "xgb_demand_model.pkl"
FEAT_PATH  = MODELS_DIR / "feature_cols.pkl"

# ── Load model ─────────────────────────────────────────────────────
try:
    model        = joblib.load(MODEL_PATH)
    feature_cols = joblib.load(FEAT_PATH)
    MODEL_LOADED = True
    print(f"✅ Model loaded — {len(feature_cols)} features")
except Exception as e:
    MODEL_LOADED = False
    model        = None
    feature_cols = []
    print(f"⚠️  Model not found: {e}\n   Run model_training.ipynb first.")

# ── Constants ──────────────────────────────────────────────────────
LAG_DEFAULTS = {
    "lag_1h"          : 2800,
    "lag_24h"         : 2800,
    "lag_168h"        : 2800,
    "rolling_3h_mean" : 2800,
    "rolling_24h_mean": 2800,
    "avg_fare"        : 18.5,
    "avg_distance"    : 2.8,
    "avg_passengers"  : 1.3,
}

WEATHER_CODE_MAP = {
    "Clear / Sunny" : 0,
    "Partly Cloudy" : 2,
    "Overcast"      : 3,
    "Drizzle"       : 51,
    "Rain"          : 61,
    "Heavy Rain"    : 63,
    "Snow"          : 71,
    "Heavy Snow"    : 75,
    "Thunderstorm"  : 95,
}

BOROUGH_MULTIPLIER = {
    "Manhattan"    : 1.00,
    "Brooklyn"     : 0.28,
    "Queens"       : 0.22,
    "Bronx"        : 0.08,
    "Staten Island": 0.03,
}

# Pre-compute real feature importances from model
FEATURE_IMPORTANCES = {}
if MODEL_LOADED:
    importances = model.feature_importances_
    for feat, imp in zip(feature_cols, importances):
        FEATURE_IMPORTANCES[feat] = float(imp)


def build_row(hour, day_of_week, day, temperature,
              precipitation, windspeed, weather_label,
              has_event, sentiment):
    """Build one feature row matching the training schema."""
    weather_code = WEATHER_CODE_MAP.get(weather_label, 0)
    is_weekend   = int(day_of_week >= 5)
    row = {
        "hour"             : hour,
        "day_of_week"      : day_of_week,
        "day"              : day,
        "is_weekend"       : is_weekend,
        "is_rush_am"       : int(7  <= hour <= 9),
        "is_rush_pm"       : int(17 <= hour <= 19),
        "is_night"         : int(hour >= 22 or hour <= 5),
        "hour_sin"         : np.sin(2 * np.pi * hour / 24),
        "hour_cos"         : np.cos(2 * np.pi * hour / 24),
        "dow_sin"          : np.sin(2 * np.pi * day_of_week / 7),
        "dow_cos"          : np.cos(2 * np.pi * day_of_week / 7),
        "temperature_c"    : temperature,
        "precipitation_mm" : precipitation,
        "windspeed_kmh"    : windspeed,
        "is_raining"       : int(precipitation > 0.5),
        "is_snowing"       : int(weather_code in [71, 73, 75, 77]),
        "is_bad_weather"   : int(precipitation > 0.5 or weather_code in [71,73,75,77]),
        "has_event"        : int(has_event),
        "sentiment_score"  : sentiment,
        **LAG_DEFAULTS,
    }
    return row


def predict_demand(hour, day_of_week, borough,
                   temperature, precipitation, windspeed,
                   weather_condition, has_event, sentiment_score):
    """Main prediction function called by Gradio."""

    if not MODEL_LOADED:
        return "⚠️  Model not loaded. Run model_training.ipynb first.", None, None

    day        = 15
    is_weekend = int(day_of_week >= 5)
    multiplier = BOROUGH_MULTIPLIER.get(borough, 1.0)

    # ── Single prediction ──────────────────────────────────────────
    row  = build_row(hour, day_of_week, day, temperature,
                     precipitation, windspeed, weather_condition,
                     has_event, sentiment_score)
    X_in = pd.DataFrame([row])
    avail = [c for c in feature_cols if c in X_in.columns]
    X_in  = X_in[avail]

    base_pred  = float(model.predict(X_in)[0])
    prediction = max(0, base_pred * multiplier)
    ci_low     = max(0, prediction * 0.92)
    ci_high    = prediction * 1.08

    avg      = (2800 if is_weekend == 0 else 2200) * multiplier
    pct_diff = (prediction - avg) / avg * 100
    arrow    = "▲" if pct_diff >= 0 else "▼"

    summary = (
        f"### Predicted Demand: {prediction:,.0f} trips/hour\n\n"
        f"**95% CI:** {ci_low:,.0f} – {ci_high:,.0f}\n\n"
        f"**{arrow} {abs(pct_diff):.1f}%** vs. {borough} average\n\n"
        f"📍 {borough} &nbsp;|&nbsp; "
        f"🕐 {hour:02d}:00 &nbsp;|&nbsp; "
        f"{'🗓️ Weekend' if is_weekend else '💼 Weekday'}"
    )

    # ── Plot 1: 24-hour forecast ────────────────────────────────────
    hours     = list(range(24))
    day_preds = []
    for h in hours:
        r  = build_row(h, day_of_week, day, temperature,
                       precipitation, windspeed, weather_condition,
                       has_event, sentiment_score)
        xr = pd.DataFrame([r])
        av = [c for c in feature_cols if c in xr.columns]
        p  = float(model.predict(xr[av])[0]) * multiplier
        day_preds.append(max(0, p))

    fig1, ax1 = plt.subplots(figsize=(9, 3.8))
    ax1.fill_between(hours,
                     [p * 0.92 for p in day_preds],
                     [p * 1.08 for p in day_preds],
                     alpha=0.18, color="steelblue")
    ax1.plot(hours, day_preds, color="steelblue", lw=2,
             marker="o", markersize=4, label="Predicted demand")
    ax1.axvline(hour, color="coral", ls="--", lw=1.8,
                label=f"Selected: {hour:02d}:00")
    ax1.scatter([hour], [prediction], color="coral", zorder=5, s=90)
    ax1.set_title(f"24-Hour Demand Forecast — {borough}", fontsize=12)
    ax1.set_xlabel("Hour of Day")
    ax1.set_ylabel("Estimated Trips")
    ax1.set_xticks(range(0, 24, 2))
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.25)
    plt.tight_layout()

    # ── Plot 2: Real feature importance from model ──────────────────
    # Group features into interpretable categories
    group_map = {
        "hour"            : "Time of Day",
        "hour_sin"        : "Time of Day",
        "hour_cos"        : "Time of Day",
        "is_rush_am"      : "Time of Day",
        "is_rush_pm"      : "Time of Day",
        "is_night"        : "Time of Day",
        "day_of_week"     : "Day of Week",
        "dow_sin"         : "Day of Week",
        "dow_cos"         : "Day of Week",
        "is_weekend"      : "Day of Week",
        "day"             : "Day of Week",
        "lag_1h"          : "Lag Features",
        "lag_24h"         : "Lag Features",
        "lag_168h"        : "Lag Features",
        "rolling_3h_mean" : "Lag Features",
        "rolling_24h_mean": "Lag Features",
        "temperature_c"   : "Weather",
        "precipitation_mm": "Weather",
        "windspeed_kmh"   : "Weather",
        "is_raining"      : "Weather",
        "is_snowing"      : "Weather",
        "is_bad_weather"  : "Weather",
        "has_event"       : "City Events",
        "sentiment_score" : "Social Sentiment",
        "avg_fare"        : "Trip Statistics",
        "avg_distance"    : "Trip Statistics",
        "avg_passengers"  : "Trip Statistics",
    }

    group_importance = {}
    for feat, imp in FEATURE_IMPORTANCES.items():
        group = group_map.get(feat, feat)
        group_importance[group] = group_importance.get(group, 0) + imp

    # Sort and normalize
    total = sum(group_importance.values()) or 1
    group_importance = {
        k: v / total * 100
        for k, v in sorted(group_importance.items(), key=lambda x: x[1])
    }

    fig2, ax2 = plt.subplots(figsize=(7, 3.8))
    bar_colors = {
        "Lag Features"    : "#e74c3c",
        "Time of Day"     : "#3498db",
        "Day of Week"     : "#2ecc71",
        "Weather"         : "#9b59b6",
        "Trip Statistics" : "#f39c12",
        "City Events"     : "#1abc9c",
        "Social Sentiment": "#e67e22",
    }
    colors_list = [bar_colors.get(k, "#95a5a6") for k in group_importance]
    bars = ax2.barh(list(group_importance.keys()),
                    list(group_importance.values()),
                    color=colors_list, edgecolor="white", height=0.55)
    for bar, v in zip(bars, group_importance.values()):
        ax2.text(v + 0.3, bar.get_y() + bar.get_height()/2,
                 f"{v:.1f}%", va="center", fontsize=9)
    ax2.set_title("Feature Group Importance (from XGBoost)", fontsize=11)
    ax2.set_xlabel("Importance (%)")
    ax2.set_xlim(0, max(group_importance.values()) * 1.2)
    plt.tight_layout()

    return summary, fig1, fig2


# ── Gradio Layout ──────────────────────────────────────────────────
with gr.Blocks(title="Urban Event Predictor", theme=gr.themes.Soft()) as demo:

    gr.Markdown(
        "# 🏙️ Multimodal Urban Event Prediction\n"
        "Forecast NYC taxi demand using taxi data, weather, social sentiment, and city events."
    )

    with gr.Row():

        # ── Left: Inputs ───────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### 📥 Input Parameters")

            borough = gr.Dropdown(
                choices=list(BOROUGH_MULTIPLIER.keys()),
                value="Manhattan", label="🗺️ Borough"
            )
            with gr.Row():
                hour        = gr.Slider(0, 23, value=18, step=1,
                                        label="Hour of Day")
                day_of_week = gr.Slider(0, 6, value=2, step=1,
                                        label="Day of Week  (0=Mon · 6=Sun)")

            gr.Markdown("#### 🌦️ Weather Conditions")
            weather_condition = gr.Dropdown(
                choices=list(WEATHER_CODE_MAP.keys()),
                value="Clear / Sunny", label="Weather"
            )
            with gr.Row():
                temperature   = gr.Slider(-15, 40, value=5,  step=0.5,
                                          label="Temperature (°C)")
                precipitation = gr.Slider(0,   50, value=0,  step=0.5,
                                          label="Precipitation (mm)")
            windspeed = gr.Slider(0, 80, value=10, step=1,
                                  label="Wind Speed (km/h)")

            gr.Markdown("#### 🎭 City Context")
            has_event = gr.Checkbox(
                label="Major Public Event Today", value=False)
            sentiment_score = gr.Slider(
                -1.0, 1.0, value=0.0, step=0.05,
                label="Social Media Sentiment  (−1 Negative · +1 Positive)")

            predict_btn = gr.Button(
                "🔮  Predict Demand", variant="primary", size="lg")

        # ── Right: Outputs ─────────────────────────────────────────
        with gr.Column(scale=2):
            gr.Markdown("### 📊 Prediction Results")
            prediction_text = gr.Markdown(
                "*Adjust inputs and click **Predict Demand** to see results.*")
            with gr.Row():
                forecast_plot   = gr.Plot(label="24-Hour Demand Forecast")
                importance_plot = gr.Plot(label="Feature Group Importance")

    predict_btn.click(
        fn=predict_demand,
        inputs=[hour, day_of_week, borough,
                temperature, precipitation, windspeed,
                weather_condition, has_event, sentiment_score],
        outputs=[prediction_text, forecast_plot, importance_plot]
    )

    gr.Markdown(
        "---\n"
        "**Model:** XGBoost · NYC Yellow Taxi (Jan 2026) · "
        "Open-Meteo Weather · Reddit/Twitter Sentiment · NYC Special Events"
    )

if __name__ == "__main__":
    demo.launch(share=False, server_port=7860)
