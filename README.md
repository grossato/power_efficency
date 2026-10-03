# 🚴‍♂️ Best BG Performance

> **Empirical modeling of endurance cycling power, cardiovascular intensity, and glycemic fueling using Continuous Glucose Monitor (CGM) telemetry and Generalized Additive Models (GAM).**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg)](https://streamlit.io/)
[![pyGAM](https://img.shields.io/badge/pyGAM-0.9+-green.svg)](https://pygam.readthedocs.io/)
[![Plotly](https://img.shields.io/badge/Plotly-5.18+-blueviolet.svg)](https://plotly.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📖 Overview

**Best BG Performance** is an advanced athletic data science and exercise physiology application. It investigates the non-linear relationship between **Blood Glucose (CGM)**, **Heart Rate (HR)**, and **Cycling Power (Watts)** during real-world training and racing.

By resolving the physiological **interstitial sensor lag** (5–15 minutes) and fitting non-parametric **Generalized Additive Models (GAM)** with tensor-product splines, the application decodes:
1. The **optimal blood glucose concentration (mg/dL)** required to maximize power at any given heart rate.
2. The **glycemic sensitivity landscape** across intensity tiers (from Zone 2 aerobic endurance to Zone 5 VO₂ max).
3. The power penalties associated with relative hypoglycemia and acute hyperglycemia under metabolic load.

---

## 🔬 Mathematical & Theoretical Foundations

### 1. Why Generalized Additive Models (GAMs)?

Traditional athletic performance analysis often relies on linear regression models:

$$\text{Power} = \beta_0 + \beta_1 \text{Heart Rate} + \beta_2 \text{Blood Glucose} + \varepsilon$$

However, linear models enforce a **constant marginal effect** ($\partial \text{Power} / \partial X_j = \beta_j$). In sports physiology, this assumption fails catastrophically:
- **Cardiovascular Response:** Heart rate displays non-linear threshold dynamics, baseline cardiac reserve, and high-end stroke volume ceilings.
- **Substrate Utilization:** Blood glucose exhibits a distinct non-monotonic response. Under-fueling ($< 70\text{ mg/dL}$) induces central nervous system fatigue and reduced motor unit recruitment, while acute hyperglycemia ($> 200\text{ mg/dL}$) can trigger hyperosmolar dehydration, delayed gastric emptying, and autonomic disruption.
- **Black-Box Alternatives (Neural Networks / Random Forests):** While neural networks can model non-linear surfaces, they act as unconstrained black boxes, risk severe overfitting on noisy sensor telemetry, and lack interpretability.

**Generalized Additive Models (Hastie & Tibshirani, 1986; Wood, 2017)** provide the ideal compromise: they relax the linearity constraint while maintaining complete additive interpretability:

$$g(\mathbb{E}[\text{Power}]) = \beta_0 + \sum_{j=1}^p f_j(X_j)$$

where each $f_j$ is a non-parametric smooth function estimated directly from empirical data.

---

### 2. Basis Expansions & Penalized B-Splines (P-Splines)

In our model, smooth functions $f_j(x)$ are represented as linear combinations of $K$ penalized B-spline basis functions $B_k(x)$:

$$f_j(x) = \sum_{k=1}^K \beta_{j,k} B_k(x)$$

To prevent overfitting and control the roughness of the estimated curve, the model minimizes the **Penalized Residual Sum of Squares (PRSS)**:

$$\mathcal{S}(\beta) = \|y - X\beta\|^2 + \sum_{j} \lambda_j \int \left[ f_j''(x) \right]^2 dx = \|y - X\beta\|^2 + \sum_{j} \lambda_j \beta_j^T S_j \beta_j$$

Where:
- $\int [f_j''(x)]^2 dx$ measures the **total curvature** (wiggliness) of the function based on its second derivative.
- $S_j$ is the discrete penalty matrix representing squared second-order differences of adjacent spline coefficients.
- $\lambda_j \ge 0$ is the **smoothing hyperparameter**:
  - As $\lambda \to \infty$, the penalty dominates, forcing $f_j''(x) = 0$ (collapsing the term into a simple linear slope).
  - As $\lambda \to 0$, the penalty vanishes, allowing the spline to interpolate empirical noise.

---

### 3. Tensor Product Interactions: $te(\text{Heart Rate}, \text{Blood Glucose})$

In human exercise metabolism, cardiovascular load and substrate availability do not act in isolation. According to the **Brooks Crossover Concept**, metabolic fuel selection shifts dramatically across intensities:
- **Low Intensity (Zone 1/2):** Fat oxidation supplies the predominant fraction of ATP; muscular glycogenolysis and circulating glucose flux are modest.
- **High Intensity (Zone 4/5):** Glycolytic flux accelerates exponentially; muscle glycogen and systemic glucose availability become the primary rate-limiting substrates.

A simple additive model $s(\text{HR}) + s(\text{BG})$ cannot capture this dependency. Furthermore, standard bivariate thin-plate splines $s(\text{HR}, \text{BG})$ assume isotropic variance and identical physical units. Because Heart Rate ($\text{bpm}$) and Blood Glucose ($\text{mg/dL}$) operate on fundamentally different scales, we construct a **Tensor Product Spline ($te$)**:

$$f_{1,2}(\text{HR}, \text{BG}) = \sum_{k=1}^{K_1} \sum_{l=1}^{K_2} \theta_{k,l} \left[ B_k^{(\text{HR})}(\text{HR}) \otimes B_l^{(\text{BG})}(\text{BG}) \right]$$

Tensor products possess **two independent penalty matrices** ($S_{\text{HR}} \otimes I$ and $I \otimes S_{\text{BG}}$) and separate smoothing penalties $\lambda_1, \lambda_2$. This makes the model **invariant to linear transformations of scale** and allows the data to dictate different degrees of smoothness along the cardiovascular and glycemic axes.

---

### 4. Full Model Specification

The complete mathematical engine implemented in `BGPerformanceGAM` is:

$$\text{Power} = \beta_0 + \underbrace{s(\text{Heart Rate})}_{\text{Cardiovascular Progression}} + \underbrace{s(\text{Blood Glucose})}_{\text{Glycemic Fuel Availability}} + \underbrace{te(\text{Heart Rate}, \text{Blood Glucose})}_{\text{Intensity-Dependent Metabolic Shift}} + \varepsilon$$

$$\varepsilon \sim \mathcal{N}(0, \sigma^2)$$

#### Optimal Glycemic Trajectory (Ridge Estimation)
From the dense fitted prediction surface $\widehat{\text{Power}}(\text{HR}, \text{BG})$, the algorithm traces the **Optimal BG Ridge Line** $\text{BG}^*(\text{HR})$ across the heart rate domain $[100, 190]\text{ bpm}$:

$$\text{BG}^*(\text{HR}) = \arg\max_{\text{BG} \in [60, 220]} \widehat{\text{Power}}(\text{HR}, \text{BG})$$

---

## ⏱️ Physiological Sensor Lag Optimization

### Capillary Blood vs. Interstitial Fluid (ISF)
Commercial Continuous Glucose Monitors (Dexcom, Abbott Freestyle Libre, Supersapiens, etc.) utilize a subcutaneous enzymatic sensor that measures glucose concentration in **interstitial fluid (ISF)**, not directly in arterial or venous blood.

Under resting conditions, ISF glucose equilibrates with systemic blood glucose. However, during cycling:
1. Muscle contractions trigger rapid insulin-independent glucose uptake via GLUT4 translocation.
2. Hepatic glucose output and ingested carbohydrates rapidly enter the circulation.
3. This creates a physiological gradient: arterial glucose changes precede subcutaneous ISF readings by **5 to 15 minutes** (sensor processing delay + physiological transit time).

### Pooled Cross-Correlation Algorithm
To align sensor readings with active muscular output, the application computes an empirical sensor lag offset across all workouts combined:
1. For each candidate lag $\tau \in [-30, +30]\text{ minutes}$, glucose timestamps are shifted by $\tau \times 60$ seconds within each activity.
2. The algorithm computes the cross-correlation between shifted glucose and the **Aerobic Efficiency Ratio** ($\text{Watts} / \text{Heart Rate}$):

$$\tau^* = \arg\max_{\tau \in [-30, +30]} \text{Corr}\left( \text{BG}(t + \tau), \frac{\text{Watts}(t)}{\text{HeartRate}(t)} \right)$$

3. The UI exposes a real-time reactive slider allowing instant adjustment of $\tau$, refitting the GAM and re-rendering all diagnostic figures on the fly.

---

## ✨ Key Features

- ⌚ **Garmin Connect Cloud Integration**: Directly logs into Garmin Connect (with automated session token caching in `.cache/garmin_tokens/`) and downloads original activity FIT files.
- 📁 **Direct .FIT File Upload**: Drag-and-drop `.fit`, `.fit.gz`, or `.zip` files exported from your Garmin Edge, Forerunner, or Connect web — completely offline with zero API credentials required.
- 🩸 **Connect IQ Developer Field Extraction**: Uses `fitdecode` to inspect record frames and decode glucose developer fields written by Connect IQ data fields (Dexcom, Libre, Supersapiens, xDrip+, etc.).
- 🔄 **Intervals.icu API Ingestion**: Pulls second-by-second activity streams (`watts`, `heartrate`, `icu_blood_glucose`, `time`) with local Apache Parquet caching (`.cache/streams/`).
- 🔄 **Automatic Unit Conversion**: Detects European CGM streams measured in $\text{mmol/L}$ and converts them to standard $\text{mg/dL}$ ($\times 18.0182$).
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
├── README.md                # Mathematical documentation & user guide
├── app.py                   # Streamlit dashboard & reactive state manager
├── garmin_client.py         # Garmin Connect API client & session caching
├── fit_parser.py            # FIT file decoder & CIQ developer field extractor
├── intervals_client.py      # Intervals.icu API client, local caching & demo data
├── data_processor.py        # Stream cleaning & pooled CGM lag optimizer
├── gam_engine.py            # GAM modeling engine (pygam.LinearGAM)
├── visualizer.py            # Plotly dark-themed interactive visualizations
├── requirements.txt         # Project dependencies
└── .cache/                  # Local Parquet & session token cache directory
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
- Sessions and tokens are cached securely in `.cache/garmin_tokens/` to avoid repeated logins.
- Downloads original `.fit` files directly from Garmin and parses developer fields written by Connect IQ apps (Dexcom, Libre, Supersapiens, etc.).

### 2. Direct .FIT File Upload
- Export `.fit`, `.fit.gz`, or `.zip` files from your Garmin Edge / watch or Garmin Connect web and upload them directly via the sidebar.
- Works offline with zero credentials or network requests required.

### 3. Intervals.icu API
1. Log in to [Intervals.icu](https://intervals.icu).
2. Go to **Settings** $\rightarrow$ scroll down to **Developer / API Access**.
3. Copy your **Athlete ID** (e.g. `i12345` or `0` for self) and **API Key**.
4. Paste them into the sidebar under **Intervals.icu API**.

### 4. Demo / Synthetic Data
- Click **"⚡ Generate Demo Data"** to simulate realistic multi-workout streams with ground-truth sensor delays and test the entire analysis pipeline instantly.

> **Privacy Note:** All credentials, tokens, and telemetry streams are stored locally on your machine and are never transmitted to any third party.

---

## 📊 Dashboard Walkthrough

| Tab | Feature | Physiological & Mathematical Description |
| :--- | :--- | :--- |
| **1. Data Summary & Diagnostics** | **Lag Optimization & Distributions** | Displays the cross-correlation curve over tested minute shifts $\tau \in [-30, 30]$, identifies $\tau^*$, and plots distribution statistics for power, heart rate, and blood glucose. Includes instant cache persistence status. |
| **2. GAM 2D Heatmap** | **Power Landscape Surface** | 2D `go.Contour` map depicting predicted power output as a joint function of HR (X-axis) and BG (Y-axis), with the overlaid red dashed **Optimal BG Ridge Line**. |
| **3. Optimal BG Curve** | **Zone Glycemic Targets** | Displays the optimal blood glucose trajectory as heart rate escalates, providing empirical target ranges for Zone 1 (Recovery) through Zone 5 (VO₂ Max). |
| **4. Power vs HR Overlay** | **Binned & Modeled Power Curves** | Dual-view panel featuring: (1) Empirical observed power curves binned by blood glucose (default 20 mg/dL bins) and HR intervals, and (2) 17 continuous GAM modeled power curves ($60\text{–}220\text{ mg/dL}$) with interactive wattage delta analysis. |
| **5. Power Curves vs Time by BG** | **Glycemic Power-Duration & Fatigue (No HR)** | Independent of cardiac response (no HR): (1) **Mean Maximal Power (MMP)** curves across standard durations (1s to 60+ min) per 20 mg/dL BG bin, (2) **Power vs Elapsed Workout Time** analyzing fatigue and pacing resilience, and (3) **Second-by-second Activity Timelines** with dual power/glucose axes. |

---

## ⚠️ Medical & Athletic Disclaimer

This software is developed for **research, athletic data science, and training analysis purposes only**. It is **not** a medical device, nor does it provide clinical diagnosis or medical treatment advice. Athletes managing conditions such as Type 1 or Type 2 Diabetes should consult their physician or endocrinologist before making any changes to insulin administration, medical monitoring, or dietary management.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
