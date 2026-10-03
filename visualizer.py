"""Plotly visualization suite for Best BG Performance dashboard."""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_gam_heatmap(
    prediction_matrix: np.ndarray,
    hr_grid: np.ndarray,
    bg_grid: np.ndarray,
    optimal_bg_df: Optional[pd.DataFrame] = None,
    colorscale: str = "Viridis",
) -> go.Figure:
    """Render 2D Contour heatmap of predicted power vs Heart Rate (X) and Blood Glucose (Y).

    Args:
        prediction_matrix: 2D array of shape (len(hr), len(bg)).
        hr_grid: 1D array of heart rates (X-axis).
        bg_grid: 1D array of blood glucose levels (Y-axis).
        optimal_bg_df: Optional DataFrame with ['heartrate', 'optimal_bg'] to overlay ridge.
        colorscale: Plotly colorscale name.

    Returns:
        Plotly Figure object.
    """
    # Note: go.Contour expects z of shape (len(y), len(x)), so we transpose prediction_matrix (which is (len(hr), len(bg)))
    z_data = prediction_matrix.T

    fig = go.Figure()

    # Contour Heatmap
    fig.add_trace(
        go.Contour(
            x=hr_grid,
            y=bg_grid,
            z=z_data,
            colorscale=colorscale,
            contours=dict(
                coloring="heatmap",
                showlabels=True,
                labelfont=dict(size=11, color="white"),
            ),
            colorbar=dict(
                title=dict(text="Predicted Power (W)", font=dict(color="white")),
                tickfont=dict(color="white"),
            ),
            hovertemplate=(
                "<b>Heart Rate:</b> %{x} bpm<br>"
                "<b>Blood Glucose:</b> %{y} mg/dL<br>"
                "<b>Predicted Power:</b> %{z:.1f} W<extra></extra>"
            ),
            name="Power Landscape",
        )
    )

    # Optional: Overlay Optimal BG Ridge Line
    if optimal_bg_df is not None and not optimal_bg_df.empty:
        fig.add_trace(
            go.Scatter(
                x=optimal_bg_df["heartrate"],
                y=optimal_bg_df["optimal_bg"],
                mode="lines",
                line=dict(color="#FF4B4B", width=3, dash="dash"),
                name="Optimal BG Ridge",
                hovertemplate=(
                    "<b>Optimal BG Ridge</b><br>"
                    "HR: %{x} bpm<br>"
                    "Peak BG: %{y:.0f} mg/dL<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=dict(
            text="GAM Power Surface: f(Heart Rate, Blood Glucose)",
            font=dict(size=18, color="white"),
        ),
        xaxis=dict(
            title="Heart Rate (bpm)",
            showgrid=True,
            gridcolor="#333333",
            zeroline=False,
            color="white",
        ),
        yaxis=dict(
            title="Blood Glucose (mg/dL)",
            showgrid=True,
            gridcolor="#333333",
            zeroline=False,
            color="white",
        ),
        template="plotly_dark",
        margin=dict(l=60, r=40, t=60, b=60),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="white"),
        ),
    )

    return fig


def plot_optimal_bg_curve(
    optimal_bg_df: pd.DataFrame,
    bg_sweet_spot: tuple = (110, 150),
) -> go.Figure:
    """Render optimal blood glucose line plot across heart rates.

    Args:
        optimal_bg_df: DataFrame containing ['heartrate', 'optimal_bg', 'max_predicted_watts'].
        bg_sweet_spot: Tuple of (lower_bound, upper_bound) to shade optimal aerobic zone.

    Returns:
        Plotly Figure object.
    """
    fig = go.Figure()

    # Optimal Aerobic Glucose Shaded Zone
    fig.add_hrect(
        y0=bg_sweet_spot[0],
        y1=bg_sweet_spot[1],
        fillcolor="rgba(0, 200, 100, 0.12)",
        line_width=0,
        annotation_text="Target Performance Zone",
        annotation_position="top left",
        annotation=dict(font=dict(color="#00E676", size=11)),
    )

    # Optimal BG trajectory
    custom_data = np.stack([optimal_bg_df["max_predicted_watts"]], axis=-1)
    fig.add_trace(
        go.Scatter(
            x=optimal_bg_df["heartrate"],
            y=optimal_bg_df["optimal_bg"],
            mode="lines+markers",
            line=dict(color="#00E5FF", width=3),
            marker=dict(size=5, color="#00E5FF"),
            customdata=custom_data,
            hovertemplate=(
                "<b>Heart Rate:</b> %{x} bpm<br>"
                "<b>Optimal Blood Glucose:</b> %{y:.0f} mg/dL<br>"
                "<b>Peak Predicted Power:</b> %{customdata[0]:.1f} W<extra></extra>"
            ),
            name="Optimal Glucose",
        )
    )

    fig.update_layout(
        title=dict(
            text="Optimal Blood Glucose Trajectory by Heart Rate Intensity",
            font=dict(size=18, color="white"),
        ),
        xaxis=dict(
            title="Heart Rate (bpm)",
            showgrid=True,
            gridcolor="#333333",
            zeroline=False,
            color="white",
        ),
        yaxis=dict(
            title="Optimal Blood Glucose (mg/dL)",
            showgrid=True,
            gridcolor="#333333",
            zeroline=False,
            color="white",
            range=[
                max(40, optimal_bg_df["optimal_bg"].min() - 20),
                min(240, optimal_bg_df["optimal_bg"].max() + 20),
            ],
        ),
        template="plotly_dark",
        margin=dict(l=60, r=40, t=60, b=60),
    )

    return fig


def plot_multi_bg_power_curves(
    curves_dict: Dict[int, np.ndarray],
    hr_grid: np.ndarray,
    colorscale_name: str = "Turbo",
) -> go.Figure:
    """Render 17 overlaid power curves across blood glucose levels (60 to 220 mg/dL).

    Args:
        curves_dict: Dictionary mapping integer BG level to array of predicted watts.
        hr_grid: 1D array of heart rates.
        colorscale_name: Plotly continuous colorscale.

    Returns:
        Plotly Figure object.
    """
    fig = go.Figure()
    bg_levels = sorted(curves_dict.keys())
    n_levels = len(bg_levels)

    # Sample continuous colors from colorscale
    colors = px.colors.sample_colorscale(colorscale_name, [i / max(1, n_levels - 1) for i in range(n_levels)])

    for idx, bg in enumerate(bg_levels):
        power_series = curves_dict[bg]
        fig.add_trace(
            go.Scatter(
                x=hr_grid,
                y=power_series,
                mode="lines",
                name=f"{bg} mg/dL",
                line=dict(color=colors[idx], width=2),
                hovertemplate=(
                    f"<b>BG {bg} mg/dL</b><br>"
                    "HR: %{x} bpm<br>"
                    "Power: %{y:.1f} W<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=dict(
            text="Power vs Heart Rate Across Blood Glucose Levels (60–220 mg/dL)",
            font=dict(size=18, color="white"),
        ),
        xaxis=dict(
            title="Heart Rate (bpm)",
            showgrid=True,
            gridcolor="#333333",
            zeroline=False,
            color="white",
        ),
        yaxis=dict(
            title="Predicted Power (Watts)",
            showgrid=True,
            gridcolor="#333333",
            zeroline=False,
            color="white",
        ),
        template="plotly_dark",
        legend=dict(
            title=dict(text="Blood Glucose", font=dict(color="white", size=12)),
            font=dict(color="white", size=10),
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=1.02,
        ),
        margin=dict(l=60, r=120, t=60, b=60),
    )

    return fig


def plot_lag_correlation(corr_df: pd.DataFrame, optimal_lag: int) -> go.Figure:
    """Render cross-correlation curve across minute lags tau in [-30, 30].

    Args:
        corr_df: DataFrame with ['lag_minutes', 'correlation'].
        optimal_lag: Identified optimal integer offset tau*.

    Returns:
        Plotly Figure object.
    """
    fig = go.Figure()

    if not corr_df.empty:
        fig.add_trace(
            go.Scatter(
                x=corr_df["lag_minutes"],
                y=corr_df["correlation"],
                mode="lines+markers",
                line=dict(color="#FFD700", width=2.5),
                marker=dict(size=6, color="#FFD700"),
                name="Cross-Correlation",
                hovertemplate="Lag: %{x} min<br>Correlation: %{y:.4f}<extra></extra>",
            )
        )

        # Highlight optimal lag point
        fig.add_vline(
            x=optimal_lag,
            line_width=2,
            line_dash="dash",
            line_color="#00E676",
            annotation_text=f"Optimal Lag: {optimal_lag} min",
            annotation_position="top left",
            annotation=dict(font=dict(color="#00E676", size=12)),
        )

    fig.update_layout(
        title=dict(
            text="Pooled CGM Sensor Lag Optimization (Cross-Correlation vs Aerobic Efficiency)",
            font=dict(size=16, color="white"),
        ),
        xaxis=dict(
            title="Lag Shift (minutes)",
            showgrid=True,
            gridcolor="#333333",
            zeroline=True,
            zerolinecolor="#666666",
            color="white",
        ),
        yaxis=dict(
            title="Pearson Correlation r(Watts / HR, BG_shifted)",
            showgrid=True,
            gridcolor="#333333",
            zeroline=True,
            zerolinecolor="#666666",
            color="white",
        ),
        template="plotly_dark",
        margin=dict(l=60, r=40, t=60, b=60),
    )

    return fig


def plot_empirical_binned_power_curves(
    df: pd.DataFrame,
    bg_bin_width: int = 20,
    hr_bin_width: int = 5,
    min_samples: int = 15,
    colorscale_name: str = "Turbo",
) -> tuple[go.Figure, pd.DataFrame]:
    """Compute and render empirical power curves from observed data binned by blood glucose.

    Bins blood glucose into slices (e.g. 20 mg/dL: 60-80, 80-100, etc.) and
    calculates mean observed power across heart rate bins (e.g. 5 bpm intervals)
    within each glycemic tier.

    Args:
        df: Cleaned & shifted DataFrame with ['watts', 'heartrate', 'icu_blood_glucose'].
        bg_bin_width: Width of blood glucose bins in mg/dL (default 20).
        hr_bin_width: Width of heart rate bins in bpm (default 5).
        min_samples: Minimum data points required in a (BG_bin, HR_bin) cell to plot.
        colorscale_name: Plotly continuous colorscale.

    Returns:
        Tuple of (Plotly Figure, summary DataFrame of binned counts & power).
    """
    fig = go.Figure()

    if df.empty or "icu_blood_glucose" not in df.columns:
        fig.update_layout(
            title="No data available for empirical power curves",
            template="plotly_dark",
        )
        return fig, pd.DataFrame()

    valid_df = df.dropna(subset=["watts", "heartrate", "icu_blood_glucose"]).copy()
    if len(valid_df) < 30:
        fig.update_layout(
            title="Insufficient data points for empirical binned curves (< 30 points)",
            template="plotly_dark",
        )
        return fig, pd.DataFrame()

    # Determine dynamic bin edges aligned to bg_bin_width
    data_min_bg = valid_df["icu_blood_glucose"].min()
    data_max_bg = valid_df["icu_blood_glucose"].max()

    bg_step = max(5, int(round(bg_bin_width)))
    min_edge = int(np.floor(data_min_bg / bg_step) * bg_step)
    max_edge = int(np.ceil(data_max_bg / bg_step) * bg_step)
    min_edge = max(40, min_edge)
    max_edge = min(320, max(min_edge + bg_step, max_edge))

    bg_bins = list(range(min_edge, max_edge + bg_step, bg_step))
    bg_labels = [f"{bg_bins[i]}-{bg_bins[i+1]} mg/dL" for i in range(len(bg_bins) - 1)]

    valid_df["bg_bin"] = pd.cut(
        valid_df["icu_blood_glucose"],
        bins=bg_bins,
        labels=bg_labels,
        include_lowest=True,
    )

    # Bin Heart Rate (center of each interval)
    valid_df["hr_bin"] = (valid_df["heartrate"] // hr_bin_width) * hr_bin_width + (
        hr_bin_width / 2.0
    )

    # Aggregate empirical mean, std, count
    agg = (
        valid_df.groupby(["bg_bin", "hr_bin"], observed=True)
        .agg(
            mean_power=("watts", "mean"),
            std_power=("watts", "std"),
            count=("watts", "count"),
        )
        .reset_index()
    )

    # Filter out cells with sparse data
    agg_filtered = agg[agg["count"] >= min_samples].copy()

    # Active BG bins present in data
    present_bins = [b for b in bg_labels if b in agg_filtered["bg_bin"].unique()]
    n_bins = len(present_bins)

    if n_bins == 0:
        # Fallback with lower min_samples threshold
        agg_filtered = agg[agg["count"] >= max(3, min_samples // 3)].copy()
        present_bins = [b for b in bg_labels if b in agg_filtered["bg_bin"].unique()]
        n_bins = len(present_bins)

    colors = (
        px.colors.sample_colorscale(colorscale_name, [i / max(1, n_bins - 1) for i in range(n_bins)])
        if n_bins > 0
        else []
    )

    summary_rows = []

    for idx, b_label in enumerate(present_bins):
        bin_data = agg_filtered[agg_filtered["bg_bin"] == b_label].sort_values("hr_bin")
        if bin_data.empty:
            continue

        raw_bin_subset = valid_df[valid_df["bg_bin"] == b_label]
        total_pts = len(raw_bin_subset)
        avg_w = raw_bin_subset["watts"].mean()
        avg_hr = raw_bin_subset["heartrate"].mean()

        summary_rows.append(
            {
                "Blood Glucose Bin": b_label,
                "Data Points (sec)": f"{total_pts:,}",
                "Avg Empirical Power": f"{avg_w:.1f} W",
                "Avg Heart Rate": f"{avg_hr:.1f} bpm",
                "HR Span Covered": f"{bin_data['hr_bin'].min():.0f} - {bin_data['hr_bin'].max():.0f} bpm",
            }
        )

        custom_data = np.stack(
            [bin_data["count"], bin_data["std_power"].fillna(0.0)], axis=-1
        )

        fig.add_trace(
            go.Scatter(
                x=bin_data["hr_bin"],
                y=bin_data["mean_power"],
                mode="lines+markers",
                name=f"{b_label} (n={total_pts:,})",
                line=dict(color=colors[idx], width=2.5),
                marker=dict(size=6, color=colors[idx]),
                customdata=custom_data,
                hovertemplate=(
                    f"<b>{b_label}</b><br>"
                    "HR Bin: %{x:.0f} bpm<br>"
                    "Mean Power: %{y:.1f} W<br>"
                    "Points: %{customdata[0]:,}<br>"
                    "Std Dev: ±%{customdata[1]:.1f} W<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=dict(
            text=f"Empirical Observed Power Curves (Binned by {bg_bin_width} mg/dL Blood Glucose)",
            font=dict(size=18, color="white"),
        ),
        xaxis=dict(
            title="Heart Rate Bin (bpm)",
            showgrid=True,
            gridcolor="#333333",
            zeroline=False,
            color="white",
        ),
        yaxis=dict(
            title="Observed Mean Power (Watts)",
            showgrid=True,
            gridcolor="#333333",
            zeroline=False,
            color="white",
        ),
        template="plotly_dark",
        legend=dict(
            title=dict(text=f"BG Bins ({bg_bin_width} mg/dL)", font=dict(color="white", size=12)),
            font=dict(color="white", size=10),
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=1.02,
        ),
        margin=dict(l=60, r=160, t=60, b=60),
    )

    summary_df = pd.DataFrame(summary_rows)
    return fig, summary_df


def format_duration(seconds: int) -> str:
    """Format duration in seconds into human-readable string (e.g. 5s, 1m, 1h20m)."""
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        mins = seconds // 60
        secs = seconds % 60
        return f"{mins}m" if secs == 0 else f"{mins}m{secs}s"
    else:
        hrs = seconds // 3600
        mins = (seconds % 3600) // 60
        return f"{hrs}h" if mins == 0 else f"{hrs}h{mins}m"


def plot_power_duration_by_bg(
    df: pd.DataFrame,
    bg_bin_width: float = 20.0,
    durations: Optional[List[int]] = None,
    metric: str = "max",
    use_log_scale: bool = True,
    min_samples: int = 5,
    colorscale_name: str = "Turbo",
) -> Tuple[go.Figure, pd.DataFrame]:
    """Plot Mean Maximal Power (MMP) Power-Duration curves for each Blood Glucose bin.

    Computes peak (or mean) sustained power output for standard cycling durations
    (e.g., 1s, 5s, 15s, 30s, 1m, 2m, 5m, 10m, 20m, 30m, 60m) when blood glucose is
    within specified glycemic brackets, completely independent of Heart Rate.

    Args:
        df: DataFrame with ['watts', 'icu_blood_glucose'] and optionally ['activity_id'].
        bg_bin_width: Width of blood glucose bins (default: 20 mg/dL).
        durations: List of durations in seconds to evaluate.
        metric: 'max' for Mean Maximal Power (peak MMP), 'mean' for average sustained power.
        use_log_scale: If True, uses logarithmic X-axis for duration.
        min_samples: Minimum rolling samples required to plot a duration point.
        colorscale_name: Plotly colorscale name for BG traces.

    Returns:
        Tuple of (Plotly Figure, formatted Pivot DataFrame).
    """
    fig = go.Figure()
    if df.empty or "watts" not in df.columns or "icu_blood_glucose" not in df.columns:
        fig.update_layout(
            title="Insufficient data for Power-Duration Curve",
            template="plotly_dark",
        )
        return fig, pd.DataFrame()

    if durations is None:
        durations = [1, 5, 10, 15, 30, 45, 60, 90, 120, 180, 240, 300, 420, 600, 900, 1200, 1800, 2400, 3600]

    valid_df = df.copy()
    valid_bg = valid_df["icu_blood_glucose"].dropna()
    if not valid_bg.empty and valid_bg.median() < 30.0:
        valid_df["icu_blood_glucose"] = valid_df["icu_blood_glucose"] * 18.0182

    # Drop invalid watts & glucose
    valid_df = valid_df[
        valid_df["watts"].notna()
        & (valid_df["watts"] >= 0)
        & valid_df["icu_blood_glucose"].notna()
    ].copy()
    if valid_df.empty:
        fig.update_layout(title="No valid records for Power-Duration analysis", template="plotly_dark")
        return fig, pd.DataFrame()

    bg_step = max(5, int(round(bg_bin_width)))
    data_min_bg = valid_df["icu_blood_glucose"].min()
    data_max_bg = valid_df["icu_blood_glucose"].max()
    min_edge = int(np.floor(data_min_bg / bg_step) * bg_step)
    max_edge = int(np.ceil(data_max_bg / bg_step) * bg_step)
    min_edge = max(40, min_edge)
    max_edge = min(320, max(min_edge + bg_step, max_edge))

    bg_bins = list(range(min_edge, max_edge + bg_step, bg_step))
    bg_labels = [f"{bg_bins[i]}-{bg_bins[i+1]} mg/dL" for i in range(len(bg_bins) - 1)]

    # Activity splits
    activities = [valid_df] if "activity_id" not in valid_df.columns else [act_df for _, act_df in valid_df.groupby("activity_id")]

    records = []
    max_act_len = max(len(act) for act in activities)
    active_durations = [d for d in durations if d <= max_act_len]

    for d in active_durations:
        d_lbl = format_duration(d)
        for act_df in activities:
            if len(act_df) < d:
                continue
            rw = act_df["watts"].rolling(d).mean()
            rbg = act_df["icu_blood_glucose"].rolling(d).mean()
            valid_mask = rw.notna() & rbg.notna()
            if not valid_mask.any():
                continue
            b_series = pd.cut(rbg[valid_mask], bins=bg_bins, labels=bg_labels, include_lowest=True)
            grouped = rw[valid_mask].groupby(b_series, observed=True).agg(["max", "mean", "count"])
            for b_name, row in grouped.iterrows():
                if row["count"] >= min_samples and pd.notna(row["max"]):
                    records.append({
                        "duration_sec": d,
                        "duration_label": d_lbl,
                        "bg_bin": b_name,
                        "max_power": float(row["max"]),
                        "mean_power": float(row["mean"]),
                        "count": int(row["count"]),
                    })

    if not records:
        fig.update_layout(title="No duration intervals met the minimum samples threshold", template="plotly_dark")
        return fig, pd.DataFrame()

    rec_df = pd.DataFrame(records)
    agg_df = rec_df.groupby(["bg_bin", "duration_sec", "duration_label"], observed=True).agg({
        "max_power": "max",
        "mean_power": "mean",
        "count": "sum",
    }).reset_index()

    target_col = "max_power" if metric == "max" else "mean_power"
    metric_title = "Peak Power (Mean Maximal Power / MMP)" if metric == "max" else "Mean Sustained Power"

    present_bins = [b for b in bg_labels if b in agg_df["bg_bin"].unique()]
    n_bins = len(present_bins)
    colors = (
        px.colors.sample_colorscale(colorscale_name, [i / max(1, n_bins - 1) for i in range(n_bins)])
        if n_bins > 0
        else []
    )

    for idx, b_label in enumerate(present_bins):
        sub = agg_df[agg_df["bg_bin"] == b_label].sort_values("duration_sec")
        if sub.empty:
            continue

        custom_data = np.stack([sub["duration_label"], sub["count"]], axis=-1)

        fig.add_trace(
            go.Scatter(
                x=sub["duration_sec"],
                y=sub[target_col],
                mode="lines+markers",
                name=f"{b_label}",
                line=dict(color=colors[idx], width=2.5),
                marker=dict(size=6, color=colors[idx]),
                customdata=custom_data,
                hovertemplate=(
                    f"<b>{b_label}</b><br>"
                    "Duration: %{customdata[0]} (%{x}s)<br>"
                    f"{metric_title}: %{{y:.1f}} W<br>"
                    "Samples in Window: %{customdata[1]:,}<extra></extra>"
                ),
            )
        )

    avail_durs = sorted(agg_df["duration_sec"].unique())
    tick_candidates = [1, 5, 10, 15, 30, 60, 120, 180, 300, 600, 1200, 1800, 2400, 3600]
    tick_vals = [d for d in tick_candidates if d in avail_durs]
    tick_texts = [format_duration(d) for d in tick_vals]

    xaxis_kwargs = dict(
        title="Duration / Time",
        showgrid=True,
        gridcolor="#333333",
        color="white",
    )
    if use_log_scale:
        xaxis_kwargs["type"] = "log"
        if tick_vals:
            xaxis_kwargs["tickmode"] = "array"
            xaxis_kwargs["tickvals"] = tick_vals
            xaxis_kwargs["ticktext"] = tick_texts
    else:
        if tick_vals:
            xaxis_kwargs["tickmode"] = "array"
            xaxis_kwargs["tickvals"] = tick_vals
            xaxis_kwargs["ticktext"] = tick_texts

    fig.update_layout(
        title=dict(
            text=f"Power vs Duration Curve by Blood Glucose ({metric_title}) - No HR",
            font=dict(size=18, color="white"),
        ),
        xaxis=xaxis_kwargs,
        yaxis=dict(
            title=f"{metric_title} (Watts)",
            showgrid=True,
            gridcolor="#333333",
            zeroline=False,
            color="white",
        ),
        template="plotly_dark",
        legend=dict(
            title=dict(text=f"BG Bins ({bg_step} mg/dL)", font=dict(color="white", size=12)),
            font=dict(color="white", size=10),
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=1.02,
        ),
        margin=dict(l=60, r=160, t=60, b=60),
    )

    pivot = agg_df.pivot(index="bg_bin", columns="duration_sec", values=target_col)
    sorted_cols = sorted(pivot.columns)
    pivot = pivot[sorted_cols]
    pivot.columns = [format_duration(int(c)) for c in sorted_cols]
    pivot = pivot.round(1)
    pivot = pivot.reindex([b for b in bg_labels if b in pivot.index])
    pivot.index.name = "Blood Glucose Bin"

    return fig, pivot


def plot_power_over_elapsed_time_by_bg(
    df: pd.DataFrame,
    bg_bin_width: float = 20.0,
    time_bin_minutes: float = 5.0,
    min_power: float = 20.0,
    min_samples: int = 10,
    colorscale_name: str = "Turbo",
) -> Tuple[go.Figure, pd.DataFrame]:
    """Plot empirical average power output vs elapsed ride time stratified by blood glucose.

    Visualizes power fatigue and endurance over time in the ride across glycemic brackets,
    without using Heart Rate.

    Args:
        df: DataFrame with ['watts', 'icu_blood_glucose'] and optionally ['timestamp', 'activity_id'].
        bg_bin_width: Width of blood glucose bins (default: 20 mg/dL).
        time_bin_minutes: Width of elapsed time intervals in minutes (default: 5.0 min).
        min_power: Minimum watts to filter coasting.
        min_samples: Minimum data points required per (BG, Time) bin.
        colorscale_name: Plotly colorscale.

    Returns:
        Tuple of (Plotly Figure, formatted Pivot DataFrame).
    """
    fig = go.Figure()
    if df.empty or "watts" not in df.columns or "icu_blood_glucose" not in df.columns:
        fig.update_layout(title="Insufficient data for Power vs Elapsed Time", template="plotly_dark")
        return fig, pd.DataFrame()

    valid_df = df.copy()
    valid_bg = valid_df["icu_blood_glucose"].dropna()
    if not valid_bg.empty and valid_bg.median() < 30.0:
        valid_df["icu_blood_glucose"] = valid_df["icu_blood_glucose"] * 18.0182

    dfs = []
    has_act = "activity_id" in valid_df.columns
    iterator = valid_df.groupby("activity_id") if has_act else [(0, valid_df)]
    for _, act_df in iterator:
        act = act_df.copy()
        if "timestamp" in act.columns and pd.api.types.is_datetime64_any_dtype(act["timestamp"]):
            act = act.sort_values("timestamp")
            t0 = act["timestamp"].iloc[0]
            act["elapsed_min"] = (act["timestamp"] - t0).dt.total_seconds() / 60.0
        else:
            act["elapsed_min"] = np.arange(len(act)) / 60.0
        dfs.append(act)

    pooled_df = pd.concat(dfs, ignore_index=True)

    mask = (
        pooled_df["watts"].notna()
        & (pooled_df["watts"] >= min_power)
        & pooled_df["icu_blood_glucose"].notna()
        & (pooled_df["icu_blood_glucose"] >= 40)
        & (pooled_df["icu_blood_glucose"] <= 350)
    )
    clean = pooled_df[mask].copy()
    if clean.empty:
        fig.update_layout(title="No valid records for Power vs Elapsed Time", template="plotly_dark")
        return fig, pd.DataFrame()

    bg_step = max(5, int(round(bg_bin_width)))
    data_min_bg = clean["icu_blood_glucose"].min()
    data_max_bg = clean["icu_blood_glucose"].max()
    min_edge = int(np.floor(data_min_bg / bg_step) * bg_step)
    max_edge = int(np.ceil(data_max_bg / bg_step) * bg_step)
    min_edge = max(40, min_edge)
    max_edge = min(320, max(min_edge + bg_step, max_edge))

    bg_bins = list(range(min_edge, max_edge + bg_step, bg_step))
    bg_labels = [f"{bg_bins[i]}-{bg_bins[i+1]} mg/dL" for i in range(len(bg_bins) - 1)]

    clean["bg_bin"] = pd.cut(clean["icu_blood_glucose"], bins=bg_bins, labels=bg_labels, include_lowest=True)
    t_step = max(1.0, float(time_bin_minutes))
    clean["time_bin"] = (clean["elapsed_min"] // t_step) * t_step + (t_step / 2.0)

    agg = clean.groupby(["bg_bin", "time_bin"], observed=True).agg(
        mean_power=("watts", "mean"),
        std_power=("watts", "std"),
        count=("watts", "count"),
    ).reset_index()

    agg_filtered = agg[agg["count"] >= min_samples].copy()
    if agg_filtered.empty:
        fig.update_layout(title="No time bins met minimum samples threshold", template="plotly_dark")
        return fig, pd.DataFrame()

    present_bins = [b for b in bg_labels if b in agg_filtered["bg_bin"].unique()]
    n_bins = len(present_bins)
    colors = (
        px.colors.sample_colorscale(colorscale_name, [i / max(1, n_bins - 1) for i in range(n_bins)])
        if n_bins > 0
        else []
    )

    for idx, b_label in enumerate(present_bins):
        sub = agg_filtered[agg_filtered["bg_bin"] == b_label].sort_values("time_bin")
        if sub.empty:
            continue

        custom_data = np.stack([sub["count"], sub["std_power"].fillna(0.0)], axis=-1)

        fig.add_trace(
            go.Scatter(
                x=sub["time_bin"],
                y=sub["mean_power"],
                mode="lines+markers",
                name=f"{b_label}",
                line=dict(color=colors[idx], width=2.5),
                marker=dict(size=6, color=colors[idx]),
                customdata=custom_data,
                hovertemplate=(
                    f"<b>{b_label}</b><br>"
                    "Elapsed Time: ~%{x:.0f} min into ride<br>"
                    "Mean Power: %{y:.1f} W<br>"
                    "Points: %{customdata[0]:,}<br>"
                    "Std Dev: ±%{customdata[1]:.1f} W<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=dict(
            text="Power vs Elapsed Workout Time by Blood Glucose (Fatigue Curve) - No HR",
            font=dict(size=18, color="white"),
        ),
        xaxis=dict(
            title="Elapsed Workout Time (Minutes)",
            showgrid=True,
            gridcolor="#333333",
            color="white",
        ),
        yaxis=dict(
            title="Observed Mean Power (Watts)",
            showgrid=True,
            gridcolor="#333333",
            zeroline=False,
            color="white",
        ),
        template="plotly_dark",
        legend=dict(
            title=dict(text=f"BG Bins ({bg_step} mg/dL)", font=dict(color="white", size=12)),
            font=dict(color="white", size=10),
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=1.02,
        ),
        margin=dict(l=60, r=160, t=60, b=60),
    )

    pivot = agg_filtered.pivot(index="bg_bin", columns="time_bin", values="mean_power")
    pivot = pivot.round(1)
    pivot.columns = [f"{int(c)}m" for c in pivot.columns]
    pivot = pivot.reindex([b for b in bg_labels if b in pivot.index])
    pivot.index.name = "Blood Glucose Bin"

    return fig, pivot


def plot_activity_power_bg_timeline(
    df: pd.DataFrame,
    activity_name: str = "Activity",
) -> go.Figure:
    """Plot second-by-second timeline of Power and Blood Glucose over elapsed time.

    Primary Y-axis shows Power (Watts), secondary Y-axis shows Blood Glucose (mg/dL).
    Heart Rate is omitted.

    Args:
        df: DataFrame with ['watts', 'icu_blood_glucose'] and optionally ['timestamp'].
        activity_name: Name or label for the activity.

    Returns:
        Plotly Figure with dual Y-axes.
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if df.empty or "watts" not in df.columns:
        fig.update_layout(title="No data available for activity timeline", template="plotly_dark")
        return fig

    sub = df.copy()
    if "timestamp" in sub.columns and pd.api.types.is_datetime64_any_dtype(sub["timestamp"]):
        sub = sub.sort_values("timestamp")
        t0 = sub["timestamp"].iloc[0]
        x_time = (sub["timestamp"] - t0).dt.total_seconds() / 60.0
        x_label = "Elapsed Workout Time (Minutes)"
    else:
        x_time = np.arange(len(sub)) / 60.0
        x_label = "Elapsed Time (Minutes)"

    # Power trace (Primary Y-axis)
    fig.add_trace(
        go.Scatter(
            x=x_time,
            y=sub["watts"],
            name="Power (Watts)",
            mode="lines",
            line=dict(color="#00e5ff", width=1.2),
            opacity=0.85,
            hovertemplate="Time: %{x:.1f} min<br>Power: %{y:.0f} W<extra></extra>",
        ),
        secondary_y=False,
    )

    # Glucose trace (Secondary Y-axis) if available
    if "icu_blood_glucose" in sub.columns:
        bg_vals = sub["icu_blood_glucose"].copy()
        valid_bg = bg_vals.dropna()
        if not valid_bg.empty and valid_bg.median() < 30.0:
            bg_vals = bg_vals * 18.0182

        fig.add_trace(
            go.Scatter(
                x=x_time,
                y=bg_vals,
                name="Blood Glucose (mg/dL)",
                mode="lines",
                line=dict(color="#ff9100", width=2.8),
                hovertemplate="Time: %{x:.1f} min<br>Glucose: %{y:.1f} mg/dL<extra></extra>",
            ),
            secondary_y=True,
        )

    fig.update_layout(
        title=dict(
            text=f"Workout Timeline: Power & Blood Glucose Output ({activity_name}) - No HR",
            font=dict(size=18, color="white"),
        ),
        xaxis=dict(
            title=x_label,
            showgrid=True,
            gridcolor="#333333",
            color="white",
        ),
        template="plotly_dark",
        margin=dict(l=60, r=60, t=60, b=60),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="white"),
        ),
    )
    fig.update_yaxes(
        title_text="Power (Watts)",
        showgrid=True,
        gridcolor="#333333",
        color="#00e5ff",
        secondary_y=False,
    )
    fig.update_yaxes(
        title_text="Blood Glucose (mg/dL)",
        showgrid=False,
        color="#ff9100",
        secondary_y=True,
    )

    return fig

