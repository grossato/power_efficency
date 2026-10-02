"""Plotly visualization suite for Best BG Performance dashboard."""

from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


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
