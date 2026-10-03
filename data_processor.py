from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CACHE_DIR = Path(".cache/streams")
ACTIVE_DATASET_FILE = CACHE_DIR / "last_active_dataset.parquet"
ACTIVE_METADATA_FILE = CACHE_DIR / "last_active_metadata.json"
ACTIVE_CORR_FILE = CACHE_DIR / "last_active_correlation.parquet"


def save_active_cache(
    df: pd.DataFrame,
    source: str = "Unknown",
    initial_lag: int = 10,
    corr_df: Optional[pd.DataFrame] = None,
):
    """Persist the currently active dataset and metadata to disk."""
    if df is None or df.empty:
        return
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(ACTIVE_DATASET_FILE, index=False)

        if corr_df is not None and not corr_df.empty:
            corr_df.to_parquet(ACTIVE_CORR_FILE, index=False)

        meta = {
            "source": source,
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "row_count": len(df),
            "workout_count": (
                int(df["activity_id"].nunique()) if "activity_id" in df.columns else 1
            ),
            "initial_lag": int(initial_lag),
        }
        ACTIVE_METADATA_FILE.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        logger.info("Persisted active dataset to disk: %s", meta)
        return meta
    except Exception as e:
        logger.warning("Failed to persist active dataset: %s", e)
        return None


def load_active_cache() -> Optional[Tuple[pd.DataFrame, Dict, pd.DataFrame]]:
    """Load previously active dataset from local disk if available."""
    if not ACTIVE_DATASET_FILE.exists():
        return None
    try:
        df = pd.read_parquet(ACTIVE_DATASET_FILE)
        if df.empty:
            return None

        meta = {}
        if ACTIVE_METADATA_FILE.exists():
            try:
                meta = json.loads(ACTIVE_METADATA_FILE.read_text(encoding="utf-8"))
            except Exception:
                pass

        corr_df = pd.DataFrame()
        if ACTIVE_CORR_FILE.exists():
            try:
                corr_df = pd.read_parquet(ACTIVE_CORR_FILE)
            except Exception:
                pass

        return df, meta, corr_df
    except Exception as e:
        logger.warning("Error reading active cache: %s", e)
        return None


def clear_active_cache():
    """Remove persistent dataset files from disk."""
    for p in [ACTIVE_DATASET_FILE, ACTIVE_METADATA_FILE, ACTIVE_CORR_FILE]:
        try:
            p.unlink(missing_ok=True)
        except Exception:
            pass



def estimate_pooled_cgm_lag(
    df: pd.DataFrame,
    lag_min: int = -30,
    lag_max: int = 30,
    step_minutes: int = 1,
    min_power: float = 20.0,
    min_hr: float = 40.0,
) -> Tuple[int, pd.DataFrame]:
    """Estimate the optimal pooled CGM sensor lag across all workouts.

    A CGM sensor exhibits a physiological delay (interstitial fluid lag + sensor processing)
    typically ranging between 5 and 15 minutes behind arterial blood glucose.
    For a positive lag tau, the true glucose at workout time t corresponds to the sensor
    reading observed at time t + tau. Shifting the sensor values backward in time aligns
    them with concurrent metabolic output.

    This function tests minute shifts tau in [lag_min, lag_max], shifts glucose per activity,
    pools all activities, and computes the correlation between shifted glucose and the
    aerobic efficiency ratio (Watts / Heart Rate).

    Args:
        df: Concatenated DataFrame with ['activity_id', 'watts', 'heartrate', 'icu_blood_glucose'].
        lag_min: Minimum lag to test in minutes (default -30).
        lag_max: Maximum lag to test in minutes (default 30).
        step_minutes: Search step size in minutes (default 1).
        min_power: Minimum watts threshold to eliminate coasting/zero periods.
        min_hr: Minimum heart rate threshold.

    Returns:
        Tuple of:
            - optimal_lag (int): The integer lag in minutes that maximizes cross-correlation.
            - correlation_df (pd.DataFrame): DataFrame with ['lag_minutes', 'correlation'].
    """
    if df.empty or "icu_blood_glucose" not in df.columns:
        logger.warning("Empty or invalid DataFrame passed to estimate_pooled_cgm_lag.")
        return 0, pd.DataFrame(columns=["lag_minutes", "correlation"])

    # Auto-convert mmol/L to mg/dL if needed
    df_eval = df.copy()
    valid_bg_sample = df_eval["icu_blood_glucose"].dropna()
    if not valid_bg_sample.empty and valid_bg_sample.median() < 30.0:
        logger.info("Converting mmol/L to mg/dL in estimate_pooled_cgm_lag.")
        df_eval["icu_blood_glucose"] = df_eval["icu_blood_glucose"] * 18.0182

    # Basic validity mask for power & HR
    valid_mask = (
        (df_eval["watts"] > min_power)
        & (df_eval["heartrate"] > min_hr)
        & df_eval["icu_blood_glucose"].notna()
        & (df_eval["icu_blood_glucose"] >= min_bg if 'min_bg' in locals() else df_eval["icu_blood_glucose"] >= 40)
        & (df_eval["icu_blood_glucose"] <= 350)
    )

    df_valid = df_eval[valid_mask].copy()
    if len(df_valid) < 50:
        logger.warning("Insufficient valid data points (%d) for lag estimation.", len(df_valid))
        return 0, pd.DataFrame(columns=["lag_minutes", "correlation"])

    # Compute Aerobic Efficiency Ratio: Watts / HR
    df_valid["aerobic_efficiency"] = df_valid["watts"] / df_valid["heartrate"]

    lags_tested = list(range(lag_min, lag_max + 1, step_minutes))
    correlations = []

    # Group by activity_id to shift within each workout's continuous stream
    has_activity_col = "activity_id" in df.columns

    for tau in lags_tested:
        # If tau > 0 (sensor is delayed), the true glucose at time t was measured by sensor at t + tau.
        # Shifting by -tau * 60 pulls future sensor measurements into the current row.
        shift_steps = -int(tau * 60)

        if has_activity_col:
            shifted_bg = df_valid.groupby("activity_id")["icu_blood_glucose"].shift(shift_steps)
        else:
            shifted_bg = df_valid["icu_blood_glucose"].shift(shift_steps)

        # Pairwise correlation with aerobic efficiency
        combined_series = pd.DataFrame(
            {"bg": shifted_bg, "ef": df_valid["aerobic_efficiency"]}
        ).dropna()

        if len(combined_series) > 30:
            corr = combined_series["bg"].corr(combined_series["ef"])
            if np.isnan(corr):
                corr = 0.0
        else:
            corr = 0.0

        correlations.append(corr)

    corr_df = pd.DataFrame({"lag_minutes": lags_tested, "correlation": correlations})

    # Find lag that maximizes correlation
    if not corr_df.empty and corr_df["correlation"].abs().sum() > 0:
        best_idx = corr_df["correlation"].idxmax()
        optimal_lag = int(corr_df.loc[best_idx, "lag_minutes"])
    else:
        optimal_lag = 0

    return optimal_lag, corr_df


def clean_and_shift_data(
    df: pd.DataFrame,
    lag_minutes: int = 0,
    min_power: Optional[float] = 20.0,
    min_hr: Optional[float] = 40.0,
    min_bg: float = 40.0,
    max_bg: float = 350.0,
) -> pd.DataFrame:
    """Shift CGM values by physiological lag and filter data for modeling.

    Args:
        df: Input DataFrame with activity streams.
        lag_minutes: Time shift in minutes to apply to CGM glucose readings.
        min_power: Minimum watts threshold (excludes coasting/stops if provided, None to keep all).
        min_hr: Minimum heart rate threshold (None to ignore HR filtering).
        min_bg: Minimum valid blood glucose (mg/dL).
        max_bg: Maximum valid blood glucose (mg/dL).

    Returns:
        Cleaned, shifted DataFrame containing valid streams.
    """
    if df.empty:
        return pd.DataFrame(
            columns=["timestamp", "activity_id", "watts", "heartrate", "icu_blood_glucose"]
        )

    df_clean = df.copy()

    # Convert mmol/L to mg/dL if needed
    if "icu_blood_glucose" in df_clean.columns:
        valid_bg = df_clean["icu_blood_glucose"].dropna()
        if not valid_bg.empty and valid_bg.median() < 30.0:
            df_clean["icu_blood_glucose"] = df_clean["icu_blood_glucose"] * 18.0182

    # Shift glucose by lag_minutes (per activity if available)
    if lag_minutes != 0 and "icu_blood_glucose" in df_clean.columns:
        shift_steps = -int(lag_minutes * 60)
        if "activity_id" in df_clean.columns:
            df_clean["icu_blood_glucose"] = df_clean.groupby("activity_id")[
                "icu_blood_glucose"
            ].shift(shift_steps)
        else:
            df_clean["icu_blood_glucose"] = df_clean["icu_blood_glucose"].shift(shift_steps)

    # Filtering criteria
    mask = (
        df_clean["watts"].notna()
        & df_clean["icu_blood_glucose"].notna()
        & (df_clean["icu_blood_glucose"] >= min_bg)
        & (df_clean["icu_blood_glucose"] <= max_bg)
    )

    if min_power is not None:
        mask = mask & (df_clean["watts"] >= min_power)

    if min_hr is not None and "heartrate" in df_clean.columns:
        mask = mask & df_clean["heartrate"].notna() & (df_clean["heartrate"] >= min_hr)

    clean_subset = df_clean[mask].copy()

    # Reset index and ensure numeric types
    clean_subset["watts"] = clean_subset["watts"].astype(float)
    if "heartrate" in clean_subset.columns:
        clean_subset["heartrate"] = pd.to_numeric(clean_subset["heartrate"], errors="coerce")
    clean_subset["icu_blood_glucose"] = clean_subset["icu_blood_glucose"].astype(float)

    return clean_subset.reset_index(drop=True)
