"""Generalized Additive Model (GAM) engine for power, heart rate, and blood glucose modeling."""

import logging
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    from pygam import LinearGAM, s, te
    PYGAM_AVAILABLE = True
except ImportError:
    PYGAM_AVAILABLE = False

logger = logging.getLogger(__name__)


class BGPerformanceGAM:
    """Wrapper around pygam.LinearGAM modeling Power ~ s(HR) + s(BG) + te(HR, BG)."""

    def __init__(
        self,
        n_splines_hr: int = 10,
        n_splines_bg: int = 10,
        n_splines_te: int = 8,
        lam: Optional[float] = None,
    ):
        """Initialize GAM architecture with tensor product interaction term.

        Args:
            n_splines_hr: Number of spline basis functions for heart rate (term 0).
            n_splines_bg: Number of spline basis functions for blood glucose (term 1).
            n_splines_te: Number of spline basis functions for HR x BG tensor product (te(0, 1)).
            lam: Optional regularization / smoothing penalty.
        """
        self.n_splines_hr = n_splines_hr
        self.n_splines_bg = n_splines_bg
        self.n_splines_te = n_splines_te
        self.lam = lam
        self.model: Optional[LinearGAM] = None
        self.r2_score: float = 0.0
        self.is_fitted: bool = False

    def fit(
        self,
        df: pd.DataFrame,
        max_samples: int = 15000,
        random_state: int = 42,
    ) -> "BGPerformanceGAM":
        """Fit LinearGAM on heart rate and blood glucose predicting watts.

        Formula: Power = s(Heart Rate) + s(Blood Glucose) + te(Heart Rate, Blood Glucose)

        Args:
            df: Clean DataFrame containing 'watts', 'heartrate', 'icu_blood_glucose'.
            max_samples: Maximum rows to sample for fast reactive fitting.
            random_state: Seed for downsampling reproducibility.

        Returns:
            self
        """
        if not PYGAM_AVAILABLE:
            raise ImportError("pygam is required. Install via `pip install pygam`.")

        if df.empty or len(df) < 30:
            raise ValueError(
                f"Insufficient data to fit GAM model. Expected >= 30 points, got {len(df)}."
            )

        # Ensure required columns
        for col in ["watts", "heartrate", "icu_blood_glucose"]:
            if col not in df.columns:
                raise KeyError(f"Missing required column '{col}' in DataFrame.")

        # Subsample if dataset is very large to maintain fast UI responsiveness
        train_df = df
        if len(df) > max_samples:
            train_df = df.sample(n=max_samples, random_state=random_state)

        X = train_df[["heartrate", "icu_blood_glucose"]].values
        y = train_df["watts"].values

        # Build GAM: s(0) for HR, s(1) for BG, te(0, 1) for interaction
        formula = (
            s(0, n_splines=self.n_splines_hr)
            + s(1, n_splines=self.n_splines_bg)
            + te(0, 1, n_splines=self.n_splines_te)
        )

        if self.lam is not None:
            self.model = LinearGAM(formula, lam=self.lam)
        else:
            self.model = LinearGAM(formula)

        self.model.fit(X, y)
        self.is_fitted = True

        # Calculate R^2 on training subset
        y_pred = self.model.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        self.r2_score = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

        logger.info("Fitted GAM model: R^2 = %.4f (on %d points)", self.r2_score, len(train_df))
        return self

    def generate_predictions(
        self,
        hr_range: Tuple[int, int] = (100, 190),
        bg_min: int = 60,
        bg_max: int = 220,
        bg_step: int = 10,
    ) -> Dict[str, Union[np.ndarray, pd.DataFrame, Dict, float]]:
        """Generate prediction grid, optimal BG vector, and multi-BG power curves.

        Args:
            hr_range: Tuple of (min_hr, max_hr) in bpm (step 1 bpm).
            bg_min: Minimum blood glucose level (mg/dL).
            bg_max: Maximum blood glucose level (mg/dL).
            bg_step: Step size for blood glucose levels (mg/dL).

        Returns:
            Dictionary with:
                - 'hr_grid': 1D array of heart rates.
                - 'bg_grid': 1D array of blood glucose levels (17 levels for 60 to 220).
                - 'prediction_matrix': 2D array of shape (len(hr), len(bg)) with predicted watts.
                - 'optimal_bg_df': DataFrame mapping each HR to optimal BG and peak power.
                - 'power_curves': Dict mapping each BG level to predicted watts across hr_grid.
                - 'r2': Model R^2 score.
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model is not fitted. Call fit() before generating predictions.")

        hr_grid = np.arange(hr_range[0], hr_range[1] + 1, 1, dtype=float)
        bg_grid = np.arange(bg_min, bg_max + 1, bg_step, dtype=float)

        # Meshgrid: shape (len(hr), len(bg))
        hr_mesh, bg_mesh = np.meshgrid(hr_grid, bg_grid, indexing="ij")
        X_grid = np.column_stack([hr_mesh.ravel(), bg_mesh.ravel()])

        preds = self.model.predict(X_grid)
        pred_matrix = preds.reshape(len(hr_grid), len(bg_grid))

        # Optimal BG vector: for each HR, argmax_BG Power(HR, BG)
        best_bg_indices = np.argmax(pred_matrix, axis=1)
        optimal_bgs = bg_grid[best_bg_indices]
        max_powers = pred_matrix[np.arange(len(hr_grid)), best_bg_indices]

        optimal_bg_df = pd.DataFrame(
            {
                "heartrate": hr_grid,
                "optimal_bg": optimal_bgs,
                "max_predicted_watts": np.round(max_powers, 1),
            }
        )

        # 17 Power Curves: mapping each BG level to power across hr_grid
        power_curves = {}
        for idx, bg_val in enumerate(bg_grid):
            power_curves[int(bg_val)] = pred_matrix[:, idx]

        return {
            "hr_grid": hr_grid,
            "bg_grid": bg_grid,
            "prediction_matrix": pred_matrix,
            "optimal_bg_df": optimal_bg_df,
            "power_curves": power_curves,
            "r2": self.r2_score,
        }
