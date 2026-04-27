# 🏙Multimodal Urban Event Prediction

Multimodal Urban Event Prediction Using Social Media Sentiment and City Data
Predict taxi availability and road congestion in New York City by fusing taxi trip records, weather data, social media sentiment, and city events into a single XGBoost model — served through a website-style Gradio interface.

---

## Project Structure

urban_event_prediction/
├── data/                              # Processed outputs (auto-created at runtime)
│   ├── merged_features_2026_01.csv    # Final merged feature dataset
│   └── weather_nyc_2026_01.csv        # Cached weather data
├── notebooks/
│   ├── setup.ipynb                    # ← Deliverable 1: EDA and data loading
│   ├── model_training.ipynb           # ← Deliverable 2: Model training and evaluation
│   └── model_training_complete.ipynb  # ← Deliverable 3: Refined training + evaluation
├── src/
│   ├── data_loader.py                 # Data loading utilities
│   ├── feature_engineering.py         # Feature pipeline
│   ├── sentiment.py                   # Sentiment scoring module
│   └── ensemble.py                    # Ensemble model logic (XGB + LGB + RF)
├── ui/
│   ├── app.py                         # Gradio web interface (4-page flow)
│   └── app_streamlit.py               # Streamlit interface (Deliverable 3 improved UI)
├── models/                            # Saved model files (auto-created)
│   ├── xgb_demand_model.pkl           # Trained XGBoost model
│   ├── lgb_model_v3.pkl               # LightGBM model
│   ├── rf_model_v3.pkl                # Random Forest model
│   ├── ensemble_weights.pkl           # Ensemble weights
│   └── feature_cols.pkl               # Feature column list
├── results/                           # All plots and evaluation outputs
│   ├── model_comparison.png
│   ├── ablation_study.png
│   ├── cv_results.png
│   ├── error_analysis.png
│   ├── predicted_vs_actual.png
│   ├── training_loss_curve.png
│   ├── shap_summary.png
│   └── feature_importance.png         # Additional feature importance visualization
├── docs/                              # Architecture diagrams, UI screenshots
│   ├── ui1.jpeg                       # UI screenshot (main interface)
│   ├── ui2.jpeg                       # UI screenshot (sidebar / interaction)
│   └── screenshot-ui.png              # Additional UI preview
├── REPORT.pdf                         # ← Deliverable 3 IEEE report (placed at root)
├── requirements.txt
├── environment.yml
├── .gitignore                         # Ignore data/models files
└── README.md

> **Note:** Raw data files are stored locally and are not committed to the repository.
> See the Dataset section below for download links.

---

## Installation

**Requirements:** Python 3.10+

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/urban_event_prediction.git
cd urban_event_prediction

# 2. Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows

# 3. Install dependencies
pip install -r requirements.txt
```


---

## 🚀 How to Run

### Step 1 — Data exploration (`setup.ipynb`)

Open `notebooks/setup.ipynb` in Jupyter and update the data paths in Section 2:

```python
TAXI_PATH       = Path(r"C:\YOUR_PATH\yellow_tripdata_2026-01.parquet")
TAXI_ZONES_PATH = Path(r"C:\YOUR_PATH\taxi_zones\taxi_zones.shp")
EVENTS_PATH     = Path(r"C:\YOUR_PATH\NYCHA_Citywide_Special_Events_20260413.csv")
REDDIT_PATH     = Path(r"C:\YOUR_PATH\Reddit_Data.csv")
TWITTER_PATH    = Path(r"C:\YOUR_PATH\Twitter_Data.csv")
```

Click **Kernel → Restart & Run All**. This will verify your environment, load all datasets, fetch weather data, and save EDA plots to `results/`.

### Step 2 — Model training (`model_training.ipynb`)

Open `notebooks/model_training.ipynb` and run all cells. This will:
- Engineer 26 features across 5 modalities
- Compare Ridge / Random Forest / LightGBM / XGBoost
- Run 5-fold time-series cross-validation
- Run Optuna hyperparameter tuning (50 trials)
- Run ablation study across all data modalities
- Save the trained model to `models/xgb_demand_model.pkl`

### Step 3 — Launch the UI

```bash
python ui/app.py
```

Then open `http://127.0.0.1:7862` in your browser. The interface walks you through:

1. **Landing page** — project overview
2. **Mode selection** — hail a taxi or drive yourself
3. **Trip inputs** — borough, time, weather, event flag
4. **Prediction result** — verdict, metrics, 24-hour forecast chart, and travel tip

---

## 📊 Datasets

| Dataset | Source | Size | Description |
|---------|--------|------|-------------|
| NYC Yellow Taxi Jan 2026 | [NYC TLC](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) | ~3M rows | Core demand signal |
| Taxi Zone Shapefile | [NYC TLC](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) | 263 zones | Geographic mapping |
| NYC Special Events | [NYC Open Data](https://data.cityofnewyork.us/) | ~28 MB | City-wide event calendar |
| Reddit Sentiment | [Kaggle](https://www.kaggle.com/datasets/cosmos98/twitter-and-reddit-sentimental-analysis-dataset) | ~37K posts | Social media sentiment labels |
| Twitter Sentiment | [Kaggle](https://www.kaggle.com/datasets/cosmos98/twitter-and-reddit-sentimental-analysis-dataset) | ~163K tweets | Social media sentiment labels |
| Weather | [Open-Meteo](https://open-meteo.com/) | 744 hours | Hourly NYC weather — free, no API key |

---

## 🧠 Model Architecture

```
NYC Taxi (tabular)  ──┐
Weather (API)       ──┤                              ┌─→ Taxi availability verdict
Social Sentiment    ──┼─→ Feature Engineering (26) ──┤    (Easy / Moderate / Hard)
City Events (flag)  ──┤       XGBoost Model         │
Lag Features        ──┘                              └─→ Congestion verdict
                                                          (Light / Moderate / Heavy)
                                                               ↓
                                                         Gradio UI (4-page flow)
```

**Model:** XGBoost with Optuna-tuned hyperparameters, trained on 80% chronological split.

---

## Current Results

| Metric | Value |
|--------|-------|
| RMSE | ~180 trips/hour |
| MAE | ~130 trips/hour |
| R² | ~0.93 |
| MAPE | ~7.2% |

See `results/` for full evaluation plots including model comparison, ablation study, cross-validation curves, SHAP summary, and error analysis.

---

## 🗂️ Feature Summary

| Category | Features | Count |
|----------|----------|-------|
| Temporal | hour, day_of_week, is_weekend, rush flags, sin/cos encoding | 11 |
| Lag | lag_1h, lag_24h, lag_168h, rolling means | 5 |
| Weather | temperature, precipitation, windspeed, condition flags | 6 |
| City context | has_event, sentiment_score | 2 |
| Trip stats | avg_fare, avg_distance, avg_passengers | 3 |

---

## Known Issues

- Sentiment data (Reddit/Twitter) is not time-stamped to January 2026 — currently used as a macro-level proxy feature rather than a time-aligned signal.
- Borough-level demand uses fixed multipliers at inference time rather than zone-specific models.
- Lag features use historical averages at UI inference time (cold-start approximation).
---

## Author

**BINGQING ZHOU**  
[zhoub1@ufl.edu]  
Course: [AI Deeping Learning], [UF], Spring 2026
