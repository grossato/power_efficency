"""Data processing, cleaning, and pooled CGM lag optimization."""

import logging
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


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
    min_power: float = 20.0,
    min_hr: float = 40.0,
    min_bg: float = 40.0,
    max_bg: float = 350.0,
) -> pd.DataFrame:
    """Shift CGM values by physiological lag and filter data for GAM modeling.

    Args:
        df: Input DataFrame with activity streams.
        lag_minutes: Time shift in minutes to apply to CGM glucose readings.
        min_power: Minimum watts threshold (excludes coasting and stops).
        min_hr: Minimum heart rate threshold.
        min_bg: Minimum valid blood glucose (mg/dL).
        max_bg: Maximum valid blood glucose (mg/dL).

    Returns:
        Cleaned, shifted DataFrame containing valid ['watts', 'heartrate', 'icu_blood_glucose'].
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
        & (df_clean["watts"] > min_power)
        & df_clean["heartrate"].notna()
        & (df_clean["heartrate"] > min_hr)
        & df_clean["icu_blood_glucose"].notna()
        & (df_clean["icu_blood_glucose"] >= min_bg)
        & (df_clean["icu_blood_glucose"] <= max_bg)
    )

    clean_subset = df_clean[mask].copy()

    # Reset index and ensure numeric types
    clean_subset["watts"] = clean_subset["watts"].astype(float)
    clean_subset["heartrate"] = clean_subset["heartrate"].astype(float)
    clean_subset["icu_blood_glucose"] = clean_subset["icu_blood_glucose"].astype(float)

    return clean_subset.reset_index(drop=True)
