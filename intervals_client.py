"""Intervals.icu API client with disk caching for activity streams."""

import os
from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)


class IntervalsClient:
    """Client for querying Intervals.icu API and caching second-by-second activity streams."""

    BASE_URL = "https://intervals.icu/api/v1"

    def __init__(
        self,
        athlete_id: str,
        api_key: str,
        cache_dir: Union[str, Path] = ".cache/streams",
    ):
        """Initialize the client with athlete credentials and local cache directory.

        Args:
            athlete_id: Intervals.icu athlete identifier (e.g., 'i12345' or '0').
            api_key: Intervals.icu API key.
            cache_dir: Local path to persist parquet stream files.
        """
        self.athlete_id = str(athlete_id).strip()
        self.api_key = str(api_key).strip()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.session = requests.Session()
        # Intervals.icu uses Basic Auth with username 'API_KEY' and password = api_key
        self.session.auth = ("API_KEY", self.api_key)
        self.session.headers.update({"Accept": "application/json"})

    def _get_cache_filepath(self, start_date: str, end_date: str) -> Path:
        """Generate standardized cache file path for date range."""
        clean_start = str(start_date).split("T")[0]
        clean_end = str(end_date).split("T")[0]
        filename = f"{self.athlete_id}_{clean_start}_{clean_end}.parquet"
        return self.cache_dir / filename

    def get_activities(self, start_date: str, end_date: str) -> List[Dict]:
        """Fetch list of athlete activities within date range.

        Args:
            start_date: Start date string (YYYY-MM-DD or ISO format).
            end_date: End date string (YYYY-MM-DD or ISO format).

        Returns:
            List of activity dict objects.
        """
        url = f"{self.BASE_URL}/athlete/{self.athlete_id}/activities"
        params = {
            "oldest": str(start_date).split("T")[0],
            "newest": str(end_date).split("T")[0],
        }

        response = self.session.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()

    def get_activity_streams(
        self, activity_id: Union[str, int], keys: Optional[List[str]] = None
    ) -> List[Dict]:
        """Fetch second-by-second telemetry streams for an activity from Intervals.icu.

        Endpoint: GET /api/v1/activity/{id}/streams.json
        Uses 'types' query param (or none to fetch all available streams).

        Args:
            activity_id: Unique activity identifier (e.g. 'i173122846' or '173122846').
            keys: Optional list of stream keys to filter (None fetches all streams).

        Returns:
            List of stream objects containing type and data.
        """
        raw_id = str(activity_id).strip()
        numeric_id = raw_id.lstrip("iI")

        # Candidate endpoints to support both prefixed and integer IDs
        candidate_urls = [
            f"{self.BASE_URL}/activity/{raw_id}/streams.json",
            f"{self.BASE_URL}/activity/{numeric_id}/streams.json",
            f"{self.BASE_URL}/activity/{raw_id}/streams",
            f"{self.BASE_URL}/activity/{numeric_id}/streams",
        ]

        params = {}
        if keys is not None:
            params["types"] = ",".join(keys)

        last_error = None
        for url in candidate_urls:
            try:
                response = self.session.get(url, params=params, timeout=30)
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 404:
                    continue
                else:
                    response.raise_for_status()
            except requests.RequestException as e:
                last_error = e

        # If filtered query failed or was 404, fallback to requesting all streams without filter
        if params:
            for url in candidate_urls[:2]:
                try:
                    response = self.session.get(url, timeout=30)
                    if response.status_code == 200:
                        return response.json()
                except requests.RequestException as e:
                    last_error = e

        if last_error:
            raise last_error
        raise requests.HTTPError(
            f"404 Not Found: Could not locate streams for activity {activity_id} at {candidate_urls[0]}"
        )

    def parse_activity_streams(
        self, activity_id: Union[str, int], activity_meta: Dict, raw_streams: Union[List, Dict]
    ) -> pd.DataFrame:
        """Parse raw stream array into a structured second-by-second DataFrame.

        Detects watts/power, heartrate/hr, and glucose/CGM streams dynamically.
        Auto-converts mmol/L to mg/dL if needed.

        Args:
            activity_id: Activity ID.
            activity_meta: Activity metadata dict containing start dates.
            raw_streams: Raw stream list or dict from API response.

        Returns:
            DataFrame with columns ['timestamp', 'activity_id', 'watts', 'heartrate', 'icu_blood_glucose'].
        """
        stream_dict = {}
        max_len = 0

        # Handle list of stream dicts: [{'type': 'watts', 'data': [...]}, ...]
        if isinstance(raw_streams, list):
            for stream in raw_streams:
                if not isinstance(stream, dict):
                    continue
                s_type = stream.get("type") or stream.get("name")
                s_data = stream.get("data", [])
                if s_type and isinstance(s_data, list):
                    stream_dict[s_type] = s_data
                    if len(s_data) > max_len:
                        max_len = len(s_data)
        elif isinstance(raw_streams, dict):
            for s_type, s_data in raw_streams.items():
                if isinstance(s_data, list):
                    stream_dict[s_type] = s_data
                    if len(s_data) > max_len:
                        max_len = len(s_data)

        if max_len == 0:
            return pd.DataFrame(
                columns=[
                    "timestamp",
                    "activity_id",
                    "watts",
                    "heartrate",
                    "icu_blood_glucose",
                ]
            )

        logger.info(
            "Activity %s streams detected: %s (length: %d)",
            activity_id,
            list(stream_dict.keys()),
            max_len,
        )

        # Detect power stream
        watts_stream = None
        for candidate in ["watts", "power"]:
            if candidate in stream_dict:
                watts_stream = stream_dict[candidate]
                break

        # Detect heartrate stream
        hr_stream = None
        for candidate in ["heartrate", "hr", "heart_rate"]:
            if candidate in stream_dict:
                hr_stream = stream_dict[candidate]
                break

        # Detect glucose stream (exact candidates first, then fuzzy match)
        glucose_stream = None
        for candidate in [
            "icu_blood_glucose",
            "blood_glucose",
            "glucose",
            "cgm",
            "cgm_glucose",
            "CGMGlucose",
            "glucose_level",
        ]:
            if candidate in stream_dict:
                glucose_stream = stream_dict[candidate]
                logger.info("Matched glucose stream on key '%s'", candidate)
                break

        if glucose_stream is None:
            for k, v in stream_dict.items():
                k_lower = k.lower()
                if "glucose" in k_lower or "cgm" in k_lower:
                    glucose_stream = v
                    logger.info("Fuzzy matched glucose stream on key '%s'", k)
                    break

        # Determine start timestamp
        start_str = (
            activity_meta.get("start_date_local")
            or activity_meta.get("start_date")
            or datetime.now().isoformat()
        )
        try:
            start_dt = pd.to_datetime(start_str)
        except Exception:
            start_dt = pd.Timestamp.now()

        # Build timestamps from 'time' stream or incremental seconds
        time_series = stream_dict.get("time") or stream_dict.get("secs")
        if time_series and len(time_series) == max_len:
            timestamps = [
                start_dt + timedelta(seconds=float(t or 0)) for t in time_series
            ]
        else:
            timestamps = [start_dt + timedelta(seconds=i) for i in range(max_len)]

        def _pad(lst: Optional[List], length: int):
            if not lst:
                return [np.nan] * length
            if len(lst) < length:
                return lst + [np.nan] * (length - len(lst))
            return lst[:length]

        watts_series = pd.to_numeric(_pad(watts_stream, max_len), errors="coerce")
        hr_series = pd.to_numeric(_pad(hr_stream, max_len), errors="coerce")
        glucose_series = pd.to_numeric(_pad(glucose_stream, max_len), errors="coerce")

        # Check for mmol/L vs mg/dL (if median < 30.0, convert to mg/dL: * 18.0182)
        valid_glucose = glucose_series.dropna()
        if not valid_glucose.empty and valid_glucose.median() < 30.0:
            logger.info(
                "Converting glucose values from mmol/L to mg/dL (median %.1f -> %.1f)",
                valid_glucose.median(),
                valid_glucose.median() * 18.0182,
            )
            glucose_series = glucose_series * 18.0182

        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "activity_id": str(activity_id),
                "watts": watts_series,
                "heartrate": hr_series,
                "icu_blood_glucose": glucose_series,
            }
        )

        return df

    def fetch_all_streams(
        self,
        start_date: str,
        end_date: str,
        force_refresh: bool = False,
        progress_callback: Optional[callable] = None,
    ) -> pd.DataFrame:
        """Retrieve and pool streams across all activities in date range, with caching.

        Args:
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
            force_refresh: If True, bypass cache and fetch fresh data from API.
            progress_callback: Optional callback func(pct: float, msg: str) for UI progress bars.

        Returns:
            Concatenated DataFrame of all streams.
        """
        cache_path = self._get_cache_filepath(start_date, end_date)

        if not force_refresh and cache_path.exists():
            logger.info("Loading cached streams from %s", cache_path)
            try:
                cached = pd.read_parquet(cache_path)
                if not cached.empty:
                    return cached
            except Exception as e:
                logger.warning("Failed reading cache (%s), re-fetching from API", e)

        activities = self.get_activities(start_date, end_date)
        activity_dfs = []
        total_acts = len(activities)

        for idx, act in enumerate(activities):
            act_id = act.get("id")
            if not act_id:
                continue

            act_name = act.get("name") or str(act_id)
            if progress_callback:
                progress_callback(
                    float(idx) / max(1, total_acts),
                    f"Fetching streams for {act_name} ({idx + 1}/{total_acts})...",
                )

            # Skip activities with 0 moving time
            if act.get("moving_time", 0) <= 0:
                continue

            try:
                raw_streams = self.get_activity_streams(act_id)
                df_act = self.parse_activity_streams(act_id, act, raw_streams)
                if not df_act.empty:
                    activity_dfs.append(df_act)
            except Exception as e:
                logger.warning("Error fetching streams for activity %s: %s", act_id, e)

        if progress_callback:
            progress_callback(1.0, "Parsing and pooling all activity streams...")

        if activity_dfs:
            combined_df = pd.concat(activity_dfs, ignore_index=True)
            try:
                combined_df.to_parquet(cache_path, index=False)
                logger.info("Cached %d stream rows to %s", len(combined_df), cache_path)
            except Exception as e:
                logger.warning("Failed to save parquet cache: %s", e)
        else:
            combined_df = pd.DataFrame(
                columns=[
                    "timestamp",
                    "activity_id",
                    "watts",
                    "heartrate",
                    "icu_blood_glucose",
                ]
            )

        return combined_df

    @staticmethod
    def generate_synthetic_data(
        n_activities: int = 5,
        duration_minutes: int = 60,
        ground_truth_lag_min: int = 12,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Generate realistic synthetic cycling data with CGM lag for demonstration & testing.

        Simulates physiological power-heartrate response with an optimal blood glucose
        band (e.g. peak power around 130-150 mg/dL) and sensor delay.
        """
        np.random.seed(seed)
        dfs = []
        base_date = datetime.now() - timedelta(days=n_activities * 2)

        for act_idx in range(n_activities):
            act_id = f"mock_{1000 + act_idx}"
            act_start = base_date + timedelta(days=act_idx * 2, hours=9)
            n_seconds = duration_minutes * 60
            t_sec = np.arange(n_seconds)

            # Heart rate pattern (intervals between 120 and 175 bpm)
            freq = 2 * np.pi / (15 * 60)  # 15 min interval cycle
            base_hr = 145 + 20 * np.sin(freq * t_sec) + np.random.normal(0, 1.5, n_seconds)
            hr = np.clip(base_hr, 105, 185)

            # True underlying blood glucose (smooth trajectory between 80 and 180 mg/dL)
            bg_freq = 2 * np.pi / (40 * 60)
            true_bg = 135 + 35 * np.cos(bg_freq * t_sec + act_idx) + np.random.normal(0, 1.0, n_seconds)
            true_bg = np.clip(true_bg, 70, 210)

            # Power relationship based on HR and BG
            # Target BG ~ 135 mg/dL yields peak efficiency
            bg_penalty = -0.015 * ((true_bg - 135) ** 2)
            # Power increases monotonically with HR
            hr_power = 2.4 * (hr - 90)
            # Interaction: higher HR benefits slightly more from optimal BG
            interaction = 0.005 * (hr - 120) * (true_bg - 100)
            
            watts = hr_power + bg_penalty + interaction + np.random.normal(0, 12.0, n_seconds)
            # Simulate occasional coasting
            coasting_mask = np.random.rand(n_seconds) < 0.05
            watts[coasting_mask] = 0
            watts = np.clip(watts, 0, 480)

            # CGM sensor reads delayed blood glucose (ground_truth_lag_min delay)
            lag_sec = ground_truth_lag_min * 60
            sensor_bg = np.empty_like(true_bg)
            sensor_bg[lag_sec:] = true_bg[:-lag_sec]
            sensor_bg[:lag_sec] = true_bg[0]  # fill initial lag with starting value
            sensor_bg += np.random.normal(0, 2.0, n_seconds)  # sensor noise

            timestamps = [act_start + timedelta(seconds=int(s)) for s in t_sec]

            df = pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "activity_id": act_id,
                    "watts": np.round(watts, 1),
                    "heartrate": np.round(hr, 1),
                    "icu_blood_glucose": np.round(sensor_bg, 1),
                }
            )
            dfs.append(df)

        return pd.concat(dfs, ignore_index=True)
