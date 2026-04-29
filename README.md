# 🗽 NYC Travel Advisor v3

> A multimodal urban demand prediction system that tells you whether it's easy to hail a taxi or how congested the roads will be in New York City — powered by a weighted ensemble of XGBoost, LightGBM, and Random Forest, with live 511NY traffic event integration.

---

## 📊 Project Overview

**NYC Travel Advisor** fuses five heterogeneous data modalities to predict hourly taxi demand and road congestion:

| Modality | Source | Signal |
|----------|--------|--------|
| NYC Yellow Taxi | NYC TLC Open Data | 50M+ trips, 16 months |
| Weather | Open-Meteo API | Hourly temp, precip, wind |
| Live Events | 511NY Official API | 800+ real-time NYC events |
| Social Interest | Google Trends | NYC taxi + traffic search volume |
| Traffic Pattern | Historical junction data | Hourly flow proxy |

**Model:** Weighted ensemble — XGBoost (25%) + LightGBM (45%) + Random Forest (30%)  
**R² = 0.9755 · RMSE = 379.68 · 32 features · Trained on 16 months of data**

---

## 🗂️ Repository Structure

```
urban_event_prediction/
├── data/                          # Processed datasets & API caches
│   ├── merged_final.csv           # Final merged feature dataset
│   ├── weather_nyc_16months.csv   # Cached weather data
│   └── 511ny_events_hourly.csv    # Cached 511NY event patterns
│
├── models/                        # Saved model artifacts
│   ├── xgb_model_v3.pkl           # XGBoost model
│   ├── lgb_model_v3.pkl           # LightGBM model
│   ├── rf_model_v3.pkl            # Random Forest model
│   ├── feature_cols_v3.pkl        # Feature column list (32 features)
│   ├── ensemble_weights.pkl       # Ensemble weights {xgb, lgb, rf}
│   ├── gat_transformer_final.pth  # GAT+Transformer weights (PyTorch)
│   └── gat_graph_data.pt          # Zone graph (edge_index, node_features)
│
├── notebooks/
│   ├── setup.ipynb                # D1: Data loading & EDA (5 modalities)
│   ├── model_training_final.ipynb # D2/D3: Ensemble training pipeline
│   ├── model_training_complete.ipynb  # D3 Final: Complete integrated pipeline
│   ├── gcn_lstm_training.ipynb    # Deep learning: GCN-LSTM prototype
│   └── gat_transformer_training.ipynb # Deep learning: GAT+Transformer (GPU)
│
├── results/                       # Plots, metrics, evaluation outputs
│   ├── ablation_study.png         # Modality contribution chart
│   ├── model_comparison.png       # Model comparison bar chart
│   ├── predicted_vs_actual.png    # Actual vs predicted time series
│   ├── d2_vs_d3_comparison.png    # D2 vs D3 improvement chart
│   ├── all_models_comparison.png  # Ensemble vs deep learning
│   ├── shap_summary.png           # SHAP feature importance
│   ├── residual_analysis.png      # Error distribution & by-hour MAE
│   ├── feature_importance_v3.png  # LightGBM feature importance
│   ├── correlation_matrix.png     # Feature correlation heatmap
│   └── cv_results.png             # Cross-validation RMSE per fold
│
├── ui/
│   ├── app_streamlit.py           # Production Streamlit interface (v3 Final)
│   └── app_v3_final.py            # Gradio interface (v3, fallback)
│
├── docs/
│   ├── ui_screenshot_result.png   # Streamlit UI — prediction result
│   └── ui_screenshot_inputs.png   # Streamlit UI — input & best windows
│
├── requirements.txt               # Python dependencies
├── environment.yml                # Conda environment
└── README.md                      # This file
```

---

## 🚀 Quick Start

### 1. Clone & Setup Environment

```bash
git clone https://github.com/yourusername/urban_event_prediction.git
cd urban_event_prediction

# Option A: Conda (recommended)
conda env create -f environment.yml
conda activate nyc_travel

# Option B: pip
pip install -r requirements.txt
```

### 2. Download Data

Download NYC Yellow Taxi parquet files from the [NYC TLC website](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page):

```python
# Run in notebook or script
import urllib.request, requests
from pathlib import Path

DL_DIR = Path("data/raw")
DL_DIR.mkdir(parents=True, exist_ok=True)
BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"

months = [f"2024-{m:02d}" for m in range(1,13)] + \
         ["2025-10","2025-11","2025-12","2026-01"]

for ym in months:
    fname = f"yellow_tripdata_{ym}.parquet"
    dst = DL_DIR / fname
    if not dst.exists():
        resp = requests.get(f"{BASE_URL}/{fname}", stream=True, timeout=300)
        with open(dst, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024*1024):
                f.write(chunk)
        print(f"✅ {fname}")
```

### 3. Train Models

```bash
# Run the complete D3 pipeline
jupyter notebook notebooks/model_training_complete.ipynb
```

Set `HIPERGATOR = False` for local CPU, `True` for HiperGator GPU.

### 4. Launch Interface

**Streamlit (recommended):**
```bash
streamlit run ui/app_streamlit.py --server.port 8501
# Open http://localhost:8501
```

**Gradio (alternative):**
```python
# In Jupyter notebook
%run "ui/app_v3_final.py"
# Open http://127.0.0.1:7863
```

---

## 📈 Model Performance

### Final Results (D3 Ensemble)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| RMSE | 379.68 trips/hr | Avg prediction error |
| MAE | 227.13 trips/hr | Median absolute error |
| R² | 0.9755 | 97.6% variance explained |
| MAPE | 12.61% | Mean abs % error (demand > 100) |

### Model Comparison

| Model | RMSE | R² | MAPE | Role |
|-------|------|-----|------|------|
| Ridge (Baseline) | 314.27 | 0.9778 | 16.59% | Baseline |
| Random Forest | 294.42 | 0.9805 | 8.16% | 30% ensemble |
| LightGBM | 327.29 | 0.9759 | 10.73% | 45% ensemble |
| XGBoost | 382.80 | 0.9671 | 9.74% | 25% + SHAP |
| **D3 Ensemble** | **379.68** | **0.9755** | **12.61%** | **Final** |
| GAT-Transformer | 742.32 | 0.9065 | 23.82% | Research |

### Multimodal Ablation

| Configuration | RMSE | Delta |
|---------------|------|-------|
| Temporal only | 812.4 | baseline |
| + Lag features | 398.2 | ↓ 51.0% |
| + Weather | 385.1 | ↓ 3.3% |
| + 511NY Events | 383.6 | ↓ 0.4% |
| + Traffic pattern | 383.1 | ↓ 0.1% |
| Full + Google Trends | 382.8 | ↓ 0.08% |

---

## ⚠️ Known Limitations

1. **Training-Inference Lag Gap:** Lag features (which contribute 51% of predictive signal) use borough-scaled historical averages at inference time, introducing ~8-12% RMSE degradation. Displayed as a disclaimer in the UI.

2. **Seasonal scope:** Model trained primarily on Oct 2025 – Jan 2026 data. Summer/holiday demand patterns may differ.

3. **Borough-level granularity:** Fixed multipliers replace zone-level models. Outer boroughs carry 18-22% higher relative MAE than Manhattan.

4. **Google Trends granularity:** Weekly signal used as daily proxy — coarse temporal alignment.

---

## 🔑 511NY API Setup

1. Register at [511ny.org/developers/help](https://511ny.org/developers/help)
2. Set your API key in `ui/app_streamlit.py`:
   ```python
   API_511NY = "your_api_key_here"
   ```

---

## 🧠 Deep Learning Extension (HiperGator)

The GAT + Transformer model requires GPU training:

```bash
# On HiperGator
pip install torch torch-geometric plotly streamlit
cd "/home/zhoub1/AI Deep Learning"
jupyter notebook gat_transformer_training.ipynb  # Set HIPERGATOR=True
```

**Architecture:** GAT (2 layers, 4 heads) + Transformer Encoder (3 layers, 4 heads, d=128)  
**Parameters:** 660,353  
**Hardware:** NVIDIA L4 GPU, CUDA 12.8, PyTorch 2.9.1

---

## 📁 Data Sources

| Source | URL | License |
|--------|-----|---------|
| NYC Yellow Taxi | [NYC TLC](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) | Public |
| Weather | [Open-Meteo](https://open-meteo.com) | CC BY 4.0 |
| Traffic Events | [511NY API](https://511ny.org/developers/help) | Developer Terms |
| Google Trends | [PyTrends](https://github.com/GeneralMills/pytrends) | Public |

---

## 👤 Contact

**Bingqing Zhou**  
University of Florida  
zhoub1@ufl.edu
