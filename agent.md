# AGENT.md — Best BG Performance App Development Instructions

This document serves as the single source of truth for AI coding agents (Cursor, Claude Code, Windsurf, Aider, Copilot, etc.) building the **Best BG Performance** Streamlit application.

---

## 1. Project Overview & Objective

The goal is to build a Python Streamlit application that connects to the **Intervals.icu API**, fetches second-by-second activity streams (`watts`, `heartrate`, `icu_blood_glucose`), computes a **pooled CGM time-lag offset** across all workouts combined, fits a Generalized Additive Model (`pygam.LinearGAM`), and renders interactive **Plotly** visualizations.

### Core Mathematical Model
$$\text{Power} \approx s(\text{Heart Rate}) + s(\text{Blood Glucose}) + te(\text{Heart Rate}, \text{Blood Glucose})$$

---

## 2. Repository Architecture

```
best-bg-performance/
├── AGENT.md                 # Agent instructions & roadmap (this file)
├── app.py                   # Streamlit UI & reactive dashboard
├── intervals_client.py      # Intervals.icu API client & disk caching
├── data_processor.py        # Stream filtering & pooled CGM lag optimizer
├── gam_engine.py            # GAM modeling engine (pygam)
├── visualizer.py            # Plotly interactive charting functions
├── requirements.txt         # Project dependencies
└── .cache/                  # Local parquet/feather cache directory
```

---

## 3. Tech Stack & Dependencies

Write a `requirements.txt` containing:
```text
streamlit>=1.30.0
pygam>=0.9.0
plotly>=5.18.0
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
requests>=2.31.0
pyarrow>=14.0.0
```

---

## 4. Agent Coding Principles & Guidelines

1. **Modular Architecture:** Keep API logic, data manipulation, modeling, and visualization strictly separated into their dedicated modules.
2. **Caching First:** Never call live APIs repeatedly during development or UI interactions. Always inspect and utilize local cache (`.cache/streams/`).
3. **Defensive Processing:** Never assume complete streams. Safely handle missing columns, null values, or zero-power segments.
4. **Reactive UI:** Streamlit state must handle lag slider updates efficiently without re-querying the API.

---

## 5. Phased Execution Roadmap

---

### Phase 1: API Ingestion & Local Caching (`intervals_client.py`)

#### Module Goal
Fetch activity streams from Intervals.icu and persist them locally to `.cache/streams/`.

#### Implementation Checklist
- [ ] Create `IntervalsClient` class accepting `athlete_id` and `api_key`.
- [ ] Authenticate via HTTP Basic Auth (`username="API_KEY"`, `password=api_key`).
- [ ] `get_activities(start_date, end_date)`: Call `GET https://intervals.icu/api/v1/athlete/{athlete_id}/activities`.
- [ ] `get_activity_streams(activity_id)`: Call `GET https://intervals.icu/api/v1/athlete/{athlete_id}/activities/{activity_id}/streams?keys=watts,heartrate,icu_blood_glucose,time`.
- [ ] `fetch_all_streams(start_date, end_date)`: Loop through activities, retrieve streams, concatenate into a single DataFrame with `['timestamp', 'activity_id', 'watts', 'heartrate', 'icu_blood_glucose']`.
- [ ] Save concatenated DataFrame to `.cache/streams/{athlete_id}_{start}_{end}.parquet`.

#### Validation Criteria
Run `python -c "from intervals_client import IntervalsClient; ..."` to confirm that a non-empty DataFrame is returned and cached locally.

---

### Phase 2: Pooled CGM Lag & Data Preprocessing (`data_processor.py`)

#### Module Goal
Calculate physiological sensor lag across all workouts pooled together and clean signal noise.

#### Implementation Checklist
- [ ] `estimate_pooled_cgm_lag(df, lag_min=-30, lag_max=30)`:
  - Takes the concatenated DataFrame of **all pooled activities**.
  - Iterates over minute shifts $\tau \in [-30, 30]$.
  - For each shift, offsets `icu_blood_glucose` and computes cross-correlation with the aerobic efficiency ratio ($\text{Watts} / \text{Heart Rate}$).
  - Returns optimal integer offset $\tau^*$ maximizing correlation.
- [ ] `clean_and_shift_data(df, lag_minutes, min_power=20)`:
  - Shifts `icu_blood_glucose` timestamps by `lag_minutes`.
  - Filters coasting/zero-power rows (`watts <= min_power`).
  - Drops rows with missing HR or out-of-bounds glucose ($< 40$ or $> 350$ mg/dL).
  - Returns clean, modeling-ready DataFrame.

#### Validation Criteria
Verify that `estimate_pooled_cgm_lag()` returns an integer (typically between 5 and 15 minutes for CGMs).

---

### Phase 3: GAM Engine (`gam_engine.py`)

#### Module Goal
Fit `pygam.LinearGAM` and generate dense prediction grids for HR and BG states.

#### Implementation Checklist
- [ ] Create `BGPerformanceGAM` class wrapping `pygam.LinearGAM`.
- [ ] `fit(df)`: Fit model predicting `watts` from `heartrate` (term 0) and `icu_blood_glucose` (term 1) using `LinearGAM(s(0) + s(1) + te(0, 1))`.
- [ ] `generate_predictions(hr_range=(100, 190), bg_min=60, bg_max=220, bg_step=10)`:
  - Create grid across HR (1 bpm steps) and BG ($60, 70, 80, \dots, 220$ mg/dL; 17 levels).
  - Compute predicted `watts` for every grid point.
  - Extract:
    1. **2D Prediction Matrix**: Array of shape `(len(hr), len(bg))`.
    2. **Optimal BG Vector**: For each HR, identify $\arg\max_{\text{BG}} \text{Power}(\text{HR}, \text{BG})$.
    3. **17 Power Curves**: Vectors mapping HR to Power for each BG level.

#### Validation Criteria
Confirm model $R^2 > 0$ and predictions cover all 17 blood glucose steps without NaNs.

---

### Phase 4: Interactive Visualizations (`visualizer.py`)

#### Module Goal
Create Plotly chart constructors for the interactive dashboard.

#### Implementation Checklist
- [ ] `plot_gam_heatmap(prediction_matrix, hr_grid, bg_grid)`: Render 2D `go.Contour` plot of Power vs HR (X-axis) and BG (Y-axis).
- [ ] `plot_optimal_bg_curve(optimal_bg_df)`: Render `go.Scatter` line plot showing Optimal BG (Y-axis) per HR (X-axis).
- [ ] `plot_multi_bg_power_curves(curves_dict, hr_grid)`: Render 17 overlaid `go.Scatter` traces colored continuously from 60 to 220 mg/dL.
- [ ] Use consistent dark layout templates (`template="plotly_dark"`).

#### Validation Criteria
Function returns `plotly.graph_objects.Figure` instances that render cleanly without missing traces.

---

### Phase 5: Streamlit Application (`app.py`)

#### Module Goal
Assemble the complete user interface and state management.

#### Implementation Checklist
- [ ] **Sidebar Controls:** Athlete ID, API Key, Date Range, "Fetch Activities" button.
- [ ] **State Management:** Store raw DataFrame, optimal lag, clean DataFrame, and fitted GAM in `st.session_state`.
- [ ] **Dynamic CGM Lag Slider:**
  ```python
  cgm_lag = st.sidebar.slider(
      "CGM Time Shift / Lag (minutes)",
      min_value=-30, max_value=30,
      value=st.session_state.get('initial_lag', 0),
      step=1
  )
  ```
  Updating the slider triggers `clean_and_shift_data()`, refits GAM, and refreshes plots instantly.
- [ ] **Tabbed Dashboard Layout:**
  - Tab 1: **Data Summary & Diagnostics**
  - Tab 2: **GAM 2D Heatmap**
  - Tab 3: **Optimal BG Curve**
  - Tab 4: **10 mg/dL Step Power Overlay**

#### Validation Criteria
Execute `streamlit run app.py`, enter credentials, fetch data, adjust lag slider, and verify reactive chart updates.

---

## 6. Execution Command Summary

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run application
streamlit run app.py
```
