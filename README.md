# 🚴‍♂️ Best BG Performance

> **Optimize cycling power and glycemic fueling using Continuous Glucose Monitor (CGM) telemetry and Generalized Additive Models (GAM).**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg)](https://streamlit.io/)
[![pyGAM](https://img.shields.io/badge/pyGAM-0.9+-green.svg)](https://pygam.readthedocs.io/)
[![Plotly](https://img.shields.io/badge/Plotly-5.18+-blueviolet.svg)](https://plotly.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📖 Overview

**Best BG Performance** is an endurance sports data science application designed to uncover the non-linear relationship between **Blood Glucose (CGM)**, **Heart Rate (HR)**, and **Cycling Power (Watts)**.

CGM sensors measure interstitial fluid glucose, which experiences a physiological sensor delay of **5 to 15 minutes** behind active systemic circulation and muscular energy utilization. This application:

1. **Ingests continuous activity streams** from the [Intervals.icu](https://intervals.icu) API.
2. **Estimates pooled CGM sensor lag** ($\tau^*$) across all workouts by maximizing cross-correlation with the **Aerobic Efficiency Ratio** ($\text{Watts} / \text{HR}$).
3. **Fits a Generalized Additive Model (GAM)** with tensor-product splines to isolate non-linear glycemic impact on power across cardiovascular intensity tiers.
4. **Delivers interactive Plotly dashboards** in a reactive Streamlit UI for data-driven race-day fueling and pacing strategies.

---

## 📐 Mathematical Formulation

The modeling engine uses non-parametric Generalized Additive Models (`pygam.LinearGAM`):

$$\text{Power} \approx s(\text{Heart Rate}) + s(\text{Blood Glucose}) + te(\text{Heart Rate}, \text{Blood Glucose})$$

Where:
- $s(\text{Heart Rate})$: Smooth non-linear spline capturing cardiovascular cardiac-power progression.
- $s(\text{Blood Glucose})$: Smooth spline representing glycemic fuel availability.
- $te(\text{Heart Rate}, \text{Blood Glucose})$: 2D Tensor-product spline modeling the physiological interaction between metabolic demand and blood sugar concentration.

---

## ✨ Key Features

- ⌚ **Garmin Connect Cloud Integration**: Directly logs into Garmin Connect and downloads original activity FIT files.
- 📁 **Direct .FIT File Upload**: Drag-and-drop `.fit`, `.fit.gz`, or `.zip` files exported from your Garmin device or Connect without needing API credentials.
- 🩸 **Connect IQ Developer Field Extraction**: Decodes glucose telemetry written to FIT files by Connect IQ data fields (Dexcom, Libre, Supersapiens, xDrip+, etc.).
- 🔄 **Intervals.icu API Ingestion & Disk Caching**: Pulls activity streams (`watts`, `heartrate`, `icu_blood_glucose`, `time`) and caches parsed data locally in Apache Parquet format (`.cache/streams/`).
- ⏱️ **Pooled CGM Lag Optimization**: Solves for the athlete-specific interstitial lag across all workouts pooled together ($\tau \in [-30, 30]\text{ min}$).
- 🎛️ **Reactive Time-Shift Slider**: Adjust CGM lag dynamically in the UI and observe instant re-fitting and plot updates without re-fetching API data.
- 🗺️ **2D GAM Contour Surface**: Visualizes predicted watts across HR and BG with the **Optimal BG Ridge Line**.
- 📈 **Zone-by-Zone Glycemic Recommendations**: Computes target glucose concentrations (mg/dL) tailored for Zone 1 (Recovery) through Zone 5 (VO₂ Max).
- 🌈 **17-Level Multi-BG Power Curves**: Overlays power trajectories for glucose levels from $60\text{ to }220\text{ mg/dL}$ in $10\text{ mg/dL}$ increments.
- ⚡ **Offline Demo Generator**: Built-in physiological synthetic stream simulator to explore all application features without requiring credentials.

---

## 📁 Repository Structure

```
best-bg-performance/
├── AGENT.md                 # Agent development roadmap & instructions
├── README.md                # Project documentation
├── app.py                   # Streamlit dashboard & state manager
├── garmin_client.py         # Garmin Connect API client & session caching
├── fit_parser.py            # FIT file decoder & CIQ developer field extractor
├── intervals_client.py      # Intervals.icu API client, local caching & demo data
├── data_processor.py        # Stream cleaning & pooled CGM lag optimizer
├── gam_engine.py            # GAM modeling engine (pygam.LinearGAM)
├── visualizer.py            # Plotly dark-themed interactive visualizations
├── requirements.txt         # Project dependencies
└── .cache/                  # Local Parquet/Feather cache directory
```

---

## 🚀 Quickstart

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/best-bg-performance.git
cd best-bg-performance
```

### 2. Create and Activate a Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**On macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`.

---

## 🔑 Data Ingestion Options

### 1. Garmin Connect (Cloud Sync)
- Enter your Garmin Connect **Email** and **Password** in the sidebar.
- Sessions and tokens are cached securely in `.cache/garmin_tokens/` to minimize login prompts.
- Downloads original `.fit` files directly from Garmin and parses developer fields written by Connect IQ apps (Dexcom, Libre, Supersapiens, etc.).

### 2. Direct .FIT File Upload
- Export `.fit`, `.fit.gz`, or `.zip` files from your Garmin Edge / watch or Garmin Connect web and upload them directly via the sidebar.
- Works offline with zero credentials or network requests required.

### 3. Intervals.icu API
1. Log in to [Intervals.icu](https://intervals.icu).
2. Go to **Settings** $\rightarrow$ scroll down to **Developer / API Access**.
3. Copy your **Athlete ID** (e.g. `i12345` or `0` for self) and **API Key**.
4. Paste them into the sidebar under **Intervals.icu API**.

> **Privacy Note:** All credentials and tokens are stored locally on your machine and are never transmitted to any third party.

---

## 📊 Dashboard Walkthrough

| Tab | Feature | Description |
| :--- | :--- | :--- |
| **1. Data Summary & Diagnostics** | **Lag Optimization** | Displays the cross-correlation curve over tested minute shifts $\tau \in [-30, 30]$ and summary metrics of filtered streams. |
| **2. GAM 2D Heatmap** | **Power Landscape** | 2D contour map depicting predicted power output as a function of HR (X-axis) and BG (Y-axis), with an overlaid optimal BG ridge line. |
| **3. Optimal BG Curve** | **Glycemic Trajectory** | Displays optimal blood glucose concentration as intensity escalates, with an automatic breakdown across Zone 1 to Zone 5. |
| **4. 10 mg/dL Step Power Overlay** | **Comparative Power** | 17 overlaid curves ($60\text{–}220\text{ mg/dL}$) highlighting power deltas between hypoglycemia, normoglycemia, and hyperglycemia. |

---

## ⚠️ Medical & Athletic Disclaimer

This software is developed for **research, athletic data science, and training analysis purposes only**. It is **not** a medical device, nor does it provide clinical diagnosis or medical treatment advice. Athletes managing conditions such as Type 1 or Type 2 Diabetes should consult their physician or endocrinologist before making any changes to insulin administration or dietary management.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
