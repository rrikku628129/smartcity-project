# 🏙Multimodal Urban Event Prediction

Predicting NYC urban demand and city events by fusing **taxi trip records**,
**social media sentiment** (Reddit + Twitter), **weather**, and **special events** data.

---

## Project Structure

```
urban_event_prediction/
├── data/                          # Processed outputs (auto-created)
│   └── merged_features_2026_01.csv
├── notebooks/
│   └── setup.ipynb                # ← START HERE
├── src/
│   ├── data_loader.py             # Data loading utilities
│   ├── feature_engineering.py     # Feature pipeline
│   └── sentiment.py               # Sentiment scoring module
├── ui/
│   └── app.py                     # Gradio interface
├── results/                       # Plots and outputs (auto-created)
│   ├── taxi_demand_overview.png
│   ├── demand_heatmap.png
│   ├── taxi_zone_map.png
│   ├── sentiment_distribution.png
│   └── correlation_matrix.png
├── docs/                          # Architecture diagrams
├── requirements.txt
└── README.md
```

> **Note:** Raw data files are stored locally in `C:\Users\86188\Downloads\` and are
> not committed to the repository. See Dataset section below for download links.

---

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/rrikku628129/smartcity-project.git
cd urban_event_prediction

# 2. Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Step 1 – Open the setup notebook
```bash
cd notebooks
jupyter lab setup.ipynb
```

### Step 2 – Update data paths in Section 2
Edit the paths at the top of Section 2 to match your local machine:
```python
TAXI_PATH       = Path(r"C:\YOUR_PATH\yellow_tripdata_2026-01.parquet")
TAXI_ZONES_PATH = Path(r"C:\YOUR_PATH\taxi_zones\taxi_zones.shp")
EVENTS_PATH     = Path(r"C:\YOUR_PATH\NYCHA_Citywide_Special_Events_20260413.csv")
REDDIT_PATH     = Path(r"C:\YOUR_PATH\Reddit_Data.csv")
TWITTER_PATH    = Path(r"C:\YOUR_PATH\Twitter_Data.csv")
```

### Step 3 – Run all cells
Click **Kernel → Restart & Run All**. The notebook will:
- Verify your environment and all dependencies
- Load all 5 datasets and print summaries
- Fetch hourly weather data from Open-Meteo API (or generate mock data if offline)
- Generate 5 exploratory plots saved to `results/`
- Merge all features into `data/merged_features_2026_01.csv`

### Step 4 – Launch UI (coming soon)
```bash
python ui/app.py
```

---

## 📊 Datasets

| Dataset | Source | Description |
|---------|--------|-------------|
| NYC Yellow Taxi Jan 2026 | [NYC TLC](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) | Hourly trip demand signal |
| Taxi Zone Shapefile | [NYC TLC](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) | Geographic zone mapping |
| NYC Special Events | [NYC Open Data](https://data.cityofnewyork.us/) | City-wide public events |
| Reddit Sentiment | [Kaggle](https://www.kaggle.com/datasets/cosmos98/twitter-and-reddit-sentimental-analysis-dataset) | Social media sentiment labels |
| Twitter Sentiment | [Kaggle](https://www.kaggle.com/datasets/cosmos98/twitter-and-reddit-sentimental-analysis-dataset) | Social media sentiment labels |
| Weather (auto-fetched) | [Open-Meteo](https://open-meteo.com/) | Hourly NYC weather — free, no API key needed |

---

## 🧠 Model Architecture

```
Taxi Demand (tabular) ──┐
Weather (time-series)  ──┤─→ Feature Fusion → XGBoost / LSTM → Prediction
Social Sentiment (NLP) ──┤                         ↓
Special Events (flag)  ──┘                   Gradio UI
```

**Planned models:**
- Baseline: XGBoost / LightGBM on tabular features
- Sentiment: fine-tuned RoBERTa for Reddit/Twitter scoring
- Final: hybrid model combining tabular + NLP features

---

## 🗂️ Feature Summary

| Feature | Source | Type |
|---------|--------|------|
| `trip_count` | Taxi data | Target variable (hourly) |
| `hour` | Taxi data | Temporal |
| `day_of_week` | Taxi data | Temporal |
| `is_weekend` | Taxi data | Temporal |
| `temperature_c` | Weather API | Meteorological |
| `precipitation_mm` | Weather API | Meteorological |
| `windspeed_kmh` | Weather API | Meteorological |
| `sentiment_score` | Reddit/Twitter | NLP *(next sprint)* |

---

## Author

**BINGQING ZHOU**  
[zhoub1@ufl.edu]  
Course: [AI Deeping Learning], [UF], Spring 2026
