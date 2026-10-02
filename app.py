"""Streamlit Application for Best BG Performance Modeling & Visualization."""

from datetime import date, datetime, timedelta
import logging

import numpy as np
import pandas as pd
import streamlit as st

from data_processor import clean_and_shift_data, estimate_pooled_cgm_lag
from fit_parser import parse_fit_activity
from gam_engine import BGPerformanceGAM
from garmin_client import GarminClient
from intervals_client import IntervalsClient
from visualizer import (
    plot_gam_heatmap,
    plot_lag_correlation,
    plot_multi_bg_power_curves,
    plot_optimal_bg_curve,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit Page Config
st.set_page_config(
    page_title="Best BG Performance",
    page_icon="🚴‍♂️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #1E1E1E;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #333333;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_session_state():
    """Initialize Streamlit session state variables."""
    if "raw_df" not in st.session_state:
        st.session_state["raw_df"] = None
    if "corr_df" not in st.session_state:
        st.session_state["corr_df"] = pd.DataFrame()
    if "initial_lag" not in st.session_state:
        st.session_state["initial_lag"] = 10
    if "current_lag" not in st.session_state:
        st.session_state["current_lag"] = 10
    if "clean_df" not in st.session_state:
        st.session_state["clean_df"] = None
    if "gam_results" not in st.session_state:
        st.session_state["gam_results"] = None
    if "last_processed_lag" not in st.session_state:
        st.session_state["last_processed_lag"] = None
    if "data_source" not in st.session_state:
        st.session_state["data_source"] = "None"


init_session_state()


def run_pipeline(raw_df: pd.DataFrame, lag_minutes: int):
    """Clean data with specified lag and fit GAM model."""
    if raw_df is None or raw_df.empty:
        return

    with st.spinner(f"Cleaning data & fitting GAM at lag {lag_minutes} min..."):
        clean_df = clean_and_shift_data(raw_df, lag_minutes=lag_minutes)
        st.session_state["clean_df"] = clean_df

        if len(clean_df) < 50:
            st.error(
                f"Only {len(clean_df)} valid data points remaining after filtering. Unable to fit GAM."
            )
            st.session_state["gam_results"] = None
            return

        try:
            gam = BGPerformanceGAM()
            gam.fit(clean_df)
            results = gam.generate_predictions()
            st.session_state["gam_results"] = results
            st.session_state["last_processed_lag"] = lag_minutes
        except Exception as e:
            st.error(f"Failed to fit GAM model: {e}")
            logger.exception("GAM fitting failed")


# --- SIDEBAR CONTROLS ---
st.sidebar.title("🚴‍♂️ Best BG Performance")
st.sidebar.markdown(
    "Analyze physiological CGM lag and determine optimal blood glucose levels for cycling power."
)
st.sidebar.divider()

st.sidebar.subheader("1. Data Ingestion")
data_mode = st.sidebar.radio(
    "Data Source",
    options=[
        "Garmin Connect",
        "Upload .FIT Files (Direct)",
        "Intervals.icu API",
        "Demo / Synthetic Data",
    ],
    index=0,
)

if data_mode == "Garmin Connect (Cloud)":
    garmin_email = st.sidebar.text_input("Garmin Email", value="", placeholder="user@example.com")
    garmin_password = st.sidebar.text_input(
        "Garmin Password", value="", type="password", placeholder="Garmin Connect password"
    )

    col_d1, col_d2 = st.sidebar.columns(2)
    default_end = date.today()
    default_start = default_end - timedelta(days=60)
    start_date = col_d1.date_input("Start Date", value=default_start)
    end_date = col_d2.date_input("End Date", value=default_end)
    force_refresh = st.sidebar.checkbox("Force Cache Refresh", value=False)

    if st.sidebar.button("📥 Fetch from Garmin", use_container_width=True, type="primary"):
        if not garmin_email or not garmin_password:
            st.sidebar.error("Please provide both Garmin Email and Password.")
        else:
            try:
                progress_bar = st.sidebar.progress(0.0)
                status_text = st.sidebar.empty()

                def update_progress(pct: float, msg: str):
                    progress_bar.progress(min(1.0, max(0.0, pct)))
                    status_text.text(msg)

                client = GarminClient(garmin_email, garmin_password)
                df = client.fetch_all_streams(
                    str(start_date),
                    str(end_date),
                    force_refresh=force_refresh,
                    progress_callback=update_progress,
                )

                progress_bar.empty()
                status_text.empty()

                if df.empty:
                    st.sidebar.warning("No activities found in Garmin Connect for this date range.")
                else:
                    n_with_bg = df["icu_blood_glucose"].notna().sum()
                    if n_with_bg == 0:
                        st.sidebar.warning(
                            f"Downloaded {len(df):,} data points, but NO blood glucose developer fields were detected in these FIT files."
                        )
                    st.session_state["raw_df"] = df
                    st.session_state["data_source"] = f"Garmin Connect ({len(df):,} sec)"

                    with st.spinner("Calculating pooled CGM lag optimization..."):
                        opt_lag, corr_df = estimate_pooled_cgm_lag(df)
                        st.session_state["initial_lag"] = opt_lag
                        st.session_state["current_lag"] = opt_lag
                        st.session_state["corr_df"] = corr_df

                    run_pipeline(df, opt_lag)
                    n_acts = df["activity_id"].nunique() if "activity_id" in df.columns else 1
                    st.sidebar.success(
                        f"Loaded {len(df):,} points across {n_acts} Garmin workouts! ({n_with_bg:,} with glucose). Optimal lag: {opt_lag} min."
                    )
            except Exception as e:
                st.sidebar.error(f"Garmin Connect Error: {e}")
                logger.exception("Garmin fetch failed")

elif data_mode == "Upload .FIT Files (Direct)":
    st.sidebar.info("Upload `.fit` or `.fit.gz` or `.zip` files exported from Garmin Connect or your device.")
    uploaded_files = st.sidebar.file_uploader(
        "Choose FIT activity files",
        type=["fit", "zip", "gz"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.sidebar.button("⚡ Process Uploaded Files", use_container_width=True, type="primary"):
        with st.spinner(f"Parsing {len(uploaded_files)} FIT files..."):
            dfs = []
            for f in uploaded_files:
                try:
                    df_act = parse_fit_activity(f.getvalue(), activity_id=f.name)
                    if not df_act.empty:
                        dfs.append(df_act)
                except Exception as e:
                    st.sidebar.warning(f"Error parsing {f.name}: {e}")

            if dfs:
                df = pd.concat(dfs, ignore_index=True)
                st.session_state["raw_df"] = df
                st.session_state["data_source"] = f"Uploaded FIT ({len(df):,} sec)"
                n_with_bg = df["icu_blood_glucose"].notna().sum()

                opt_lag, corr_df = estimate_pooled_cgm_lag(df)
                st.session_state["initial_lag"] = opt_lag
                st.session_state["current_lag"] = opt_lag
                st.session_state["corr_df"] = corr_df

                run_pipeline(df, opt_lag)
                st.sidebar.success(
                    f"Parsed {len(df):,} points across {len(dfs)} files ({n_with_bg:,} with glucose)! Optimal lag: {opt_lag} min."
                )
            else:
                st.sidebar.error("Could not parse any valid records from uploaded files.")

elif data_mode == "Intervals.icu API":
    athlete_id = st.sidebar.text_input("Athlete ID", value="", placeholder="e.g. i12345 or 0")
    api_key = st.sidebar.text_input(
        "API Key", value="", type="password", placeholder="Intervals.icu API Key"
    )

    col_d1, col_d2 = st.sidebar.columns(2)
    default_end = date.today()
    default_start = default_end - timedelta(days=60)
    start_date = col_d1.date_input("Start Date", value=default_start)
    end_date = col_d2.date_input("End Date", value=default_end)
    force_refresh = st.sidebar.checkbox("Force Cache Refresh", value=False)

    if st.sidebar.button("📥 Fetch Activities", use_container_width=True, type="primary"):
        if not athlete_id or not api_key:
            st.sidebar.error("Please provide both Athlete ID and API Key.")
        else:
            try:
                progress_bar = st.sidebar.progress(0.0)
                status_text = st.sidebar.empty()

                def update_progress(pct: float, msg: str):
                    progress_bar.progress(min(1.0, max(0.0, pct)))
                    status_text.text(msg)

                client = IntervalsClient(athlete_id, api_key)
                df = client.fetch_all_streams(
                    str(start_date),
                    str(end_date),
                    force_refresh=force_refresh,
                    progress_callback=update_progress,
                )

                progress_bar.empty()
                status_text.empty()

                if df.empty:
                    st.sidebar.warning(
                        "No activities or streams found for this date range with power and glucose."
                    )
                else:
                    st.session_state["raw_df"] = df
                    st.session_state["data_source"] = f"Intervals.icu ({len(df):,} sec)"

                    # Estimate optimal lag
                    with st.spinner("Calculating pooled CGM lag optimization..."):
                        opt_lag, corr_df = estimate_pooled_cgm_lag(df)
                        st.session_state["initial_lag"] = opt_lag
                        st.session_state["current_lag"] = opt_lag
                        st.session_state["corr_df"] = corr_df

                    run_pipeline(df, opt_lag)
                    n_acts = df["activity_id"].nunique() if "activity_id" in df.columns else 1
                    st.sidebar.success(
                        f"Loaded {len(df):,} data points across {n_acts} workouts! Optimal lag: {opt_lag} min."
                    )
            except Exception as e:
                st.sidebar.error(f"API Error: {e}")
                logger.exception("API fetch failed")

else:
    st.sidebar.info("Generate realistic synthetic power, HR, and delayed CGM data for testing.")
    n_workouts = st.sidebar.slider("Workouts to Simulate", 2, 10, 5)
    true_sim_lag = st.sidebar.slider("Simulated CGM Sensor Delay (min)", 5, 20, 12)

    if st.sidebar.button("⚡ Generate Demo Data", use_container_width=True, type="primary"):
        with st.spinner("Generating synthetic physiological streams..."):
            df = IntervalsClient.generate_synthetic_data(
                n_activities=n_workouts,
                ground_truth_lag_min=true_sim_lag,
            )
            st.session_state["raw_df"] = df
            st.session_state["data_source"] = f"Demo Synthetic ({len(df):,} sec)"

            # Estimate optimal lag
            opt_lag, corr_df = estimate_pooled_cgm_lag(df)
            st.session_state["initial_lag"] = opt_lag
            st.session_state["current_lag"] = opt_lag
            st.session_state["corr_df"] = corr_df

            run_pipeline(df, opt_lag)
            st.sidebar.success(f"Generated {len(df):,} points! Estimated lag: {opt_lag} min.")

st.sidebar.divider()
st.sidebar.subheader("2. CGM Lag Adjustment")

# Dynamic CGM Lag Slider
current_initial = int(st.session_state.get("initial_lag", 10))
cgm_lag = st.sidebar.slider(
    "CGM Time Shift / Lag (minutes)",
    min_value=-30,
    max_value=30,
    value=current_initial,
    step=1,
    help="Shifts CGM glucose values to compensate for physiological interstitial fluid delay.",
)

# Reactively re-run pipeline if slider changes
if (
    st.session_state["raw_df"] is not None
    and cgm_lag != st.session_state.get("last_processed_lag")
):
    run_pipeline(st.session_state["raw_df"], cgm_lag)


# --- MAIN DASHBOARD CONTENT ---

st.title("🚴‍♂️ Best BG Performance Dashboard")
st.caption(
    "Mathematical Model: **Power ≈ s(Heart Rate) + s(Blood Glucose) + te(Heart Rate, Blood Glucose)**"
)

raw_df = st.session_state.get("raw_df")
clean_df = st.session_state.get("clean_df")
gam_results = st.session_state.get("gam_results")
corr_df = st.session_state.get("corr_df")

if raw_df is None or gam_results is None:
    st.info(
        "👋 Welcome! Please enter your **preferred input method** or click **'⚡ Generate Demo Data'** in the sidebar to start."
    )
    # Quick launch button in main area
    if st.button("🚀 Quick Launch with Demo Data", type="primary"):
        df = IntervalsClient.generate_synthetic_data(n_activities=5, ground_truth_lag_min=12)
        st.session_state["raw_df"] = df
        st.session_state["data_source"] = f"Demo Synthetic ({len(df):,} sec)"
        opt_lag, corr_df = estimate_pooled_cgm_lag(df)
        st.session_state["initial_lag"] = opt_lag
        st.session_state["current_lag"] = opt_lag
        st.session_state["corr_df"] = corr_df
        run_pipeline(df, opt_lag)
        st.rerun()
    st.stop()

# Key metrics banner
col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
n_workouts = (
    raw_df["activity_id"].nunique() if "activity_id" in raw_df.columns else 1
)
opt_lag_val = st.session_state.get("initial_lag", 0)
r2_val = gam_results.get("r2", 0.0)

col_m1.metric("Workouts Pooled", f"{n_workouts}")
col_m2.metric("Raw Data Points", f"{len(raw_df):,}")
col_m3.metric("Modeled Points", f"{len(clean_df):,}")
col_m4.metric("Estimated Optimal Lag", f"{opt_lag_val} min")
col_m5.metric("Model Explained Var (R²)", f"{r2_val:.3f}")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 Data Summary & Diagnostics",
        "🗺️ GAM 2D Heatmap",
        "📈 Optimal BG Curve",
        "🌈 10 mg/dL Step Power Overlay",
    ]
)

with tab1:
    st.subheader("Data Diagnostics & CGM Lag Optimization")
    col_t1_l, col_t1_r = st.columns([3, 2])

    with col_t1_l:
        if not corr_df.empty:
            fig_lag = plot_lag_correlation(corr_df, opt_lag_val)
            st.plotly_chart(fig_lag, use_container_width=True)
        else:
            st.info("Cross-correlation diagnostic not available.")

    with col_t1_r:
        st.markdown(
            f"""
            ### Physiological Sensor Lag Insights
            - **Estimated Optimal Delay:** `{opt_lag_val} minutes`
            - **Active Applied Shift:** `{cgm_lag} minutes`
            
            **Why Lag Optimization Matters:**
            Continuous Glucose Monitors (CGM) sample interstitial fluid rather than direct arterial blood. 
            During physical exertion, systemic metabolic changes precede subcutaneous readings by **5 to 15 minutes**.
            
            By computing the cross-correlation between the **Aerobic Efficiency Ratio** ($Watts / HR$) 
            and glucose across all pooled workouts, we align glucose readings directly with muscular power output.
            """
        )

    st.divider()
    st.subheader("Cleaned Stream Distributions")
    col_d1, col_d2, col_d3 = st.columns(3)

    col_d1.metric(
        "Avg Power (Filtered)",
        f"{clean_df['watts'].mean():.1f} W",
        f"Max: {clean_df['watts'].max():.0f} W",
    )
    col_d2.metric(
        "Avg Heart Rate",
        f"{clean_df['heartrate'].mean():.1f} bpm",
        f"Max: {clean_df['heartrate'].max():.0f} bpm",
    )
    col_d3.metric(
        "Avg Blood Glucose",
        f"{clean_df['icu_blood_glucose'].mean():.1f} mg/dL",
        f"Range: {clean_df['icu_blood_glucose'].min():.0f} - {clean_df['icu_blood_glucose'].max():.0f}",
    )

    with st.expander("Inspect Sample Cleaned Data"):
        st.dataframe(clean_df.head(100), use_container_width=True)

with tab2:
    st.subheader("2D Generalized Additive Model (GAM) Surface")
    st.markdown(
        "Interactive contour map predicting **Power (Watts)** as a function of **Heart Rate (X-axis)** and **Blood Glucose (Y-axis)**."
    )

    pred_matrix = gam_results["prediction_matrix"]
    hr_grid = gam_results["hr_grid"]
    bg_grid = gam_results["bg_grid"]
    opt_df = gam_results["optimal_bg_df"]

    fig_heat = plot_gam_heatmap(pred_matrix, hr_grid, bg_grid, optimal_bg_df=opt_df)
    st.plotly_chart(fig_heat, use_container_width=True)

    st.info(
        "💡 The red dashed line denotes the **Optimal BG Ridge**, showing the blood glucose concentration that yields maximum predicted watts at each heart rate."
    )

with tab3:
    st.subheader("Optimal Blood Glucose Trajectory by Heart Rate")
    st.markdown(
        "Determines the glycemic state that optimizes power output for each aerobic intensity tier."
    )

    opt_df = gam_results["optimal_bg_df"]
    fig_opt = plot_optimal_bg_curve(opt_df)
    st.plotly_chart(fig_opt, use_container_width=True)

    # Zones Summary Table
    st.subheader("Target Glycemic Recommendations by Heart Rate Tier")
    zones = [
        {"Zone": "Zone 1 / Recovery", "HR Range": "100 - 120 bpm", "min_hr": 100, "max_hr": 120},
        {"Zone": "Zone 2 / Endurance", "HR Range": "121 - 145 bpm", "min_hr": 121, "max_hr": 145},
        {"Zone": "Zone 3 / Tempo", "HR Range": "146 - 160 bpm", "min_hr": 146, "max_hr": 160},
        {"Zone": "Zone 4 / Threshold", "HR Range": "161 - 175 bpm", "min_hr": 161, "max_hr": 175},
        {"Zone": "Zone 5 / VO2 Max", "HR Range": "176+ bpm", "min_hr": 176, "max_hr": 190},
    ]

    zone_recs = []
    for z in zones:
        subset = opt_df[
            (opt_df["heartrate"] >= z["min_hr"]) & (opt_df["heartrate"] <= z["max_hr"])
        ]
        if not subset.empty:
            mean_opt = subset["optimal_bg"].mean()
            min_opt = subset["optimal_bg"].min()
            max_opt = subset["optimal_bg"].max()
            rec_range = f"{min_opt:.0f} - {max_opt:.0f} mg/dL (avg {mean_opt:.0f})"
            peak_power = f"{subset['max_predicted_watts'].mean():.1f} W"
        else:
            rec_range = "N/A"
            peak_power = "N/A"
        zone_recs.append(
            {
                "Training Zone": z["Zone"],
                "Heart Rate Range": z["HR Range"],
                "Optimal BG Target": rec_range,
                "Avg Optimal Power": peak_power,
            }
        )

    st.table(pd.DataFrame(zone_recs))

with tab4:
    st.subheader("Multi-Level Power Curves (10 mg/dL Steps)")
    st.markdown(
        "Comparison of 17 overlaid power curves across blood glucose concentrations from **60 to 220 mg/dL**."
    )

    curves = gam_results["power_curves"]
    fig_multi = plot_multi_bg_power_curves(curves, gam_results["hr_grid"])
    st.plotly_chart(fig_multi, use_container_width=True)

    # Power difference analysis
    with st.expander("Explore Power Deltas: Optimal vs Low/High BG"):
        hr_select = st.slider("Select Heart Rate for Glycemic Comparison", 100, 190, 150, 5)
        hr_idx = np.where(gam_results["hr_grid"] == hr_select)[0]
        if len(hr_idx) > 0:
            idx = hr_idx[0]
            bg_powers = [
                {"Blood Glucose (mg/dL)": bg, "Predicted Power (W)": round(curves[bg][idx], 1)}
                for bg in sorted(curves.keys())
            ]
            bg_df = pd.DataFrame(bg_powers)
            max_p = bg_df["Predicted Power (W)"].max()
            bg_df["Delta from Peak (W)"] = bg_df["Predicted Power (W)"] - max_p

            st.dataframe(bg_df, use_container_width=True)
