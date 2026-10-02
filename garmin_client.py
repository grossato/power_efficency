"""Garmin Connect API client for downloading and parsing activities with CGM data."""

import json
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import pandas as pd
from garminconnect import (
    Garmin,
    GarminConnectAuthenticationError,
    GarminConnectConnectionError,
    GarminConnectTooManyRequestsError,
)

from fit_parser import parse_fit_activity

logger = logging.getLogger(__name__)


class GarminClient:
    """Client for logging into Garmin Connect and retrieving activity streams from FIT files."""

    def __init__(
        self,
        email: str,
        password: str,
        token_dir: Union[str, Path] = ".cache/garmin_tokens",
        cache_dir: Union[str, Path] = ".cache/streams",
    ):
        """Initialize Garmin client with credentials and session token directory.

        Args:
            email: Garmin Connect account email.
            password: Garmin Connect account password.
            token_dir: Directory to save session tokens to avoid repeated logins.
            cache_dir: Directory to cache combined parquet files.
        """
        self.email = email.strip()
        self.password = password.strip()
        self.token_dir = Path(token_dir)
        self.token_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.garmin: Optional[Garmin] = None

    def _get_token_file(self) -> Path:
        clean_email = "".join(c if c.isalnum() else "_" for c in self.email)
        return self.token_dir / f"token_{clean_email}.json"

    def login(self) -> None:
        """Authenticate with Garmin Connect using token store or credentials."""
        token_file = self._get_token_file()

        self.garmin = Garmin(self.email, self.password)

        try:
            logger.info("Logging into Garmin Connect (tokenstore: %s)...", token_file)
            # In garminconnect, passing tokenstore automatically loads and saves session tokens
            self.garmin.login(tokenstore=str(token_file))
        except (GarminConnectAuthenticationError, GarminConnectConnectionError) as e:
            # If saved token failed, retry with fresh login
            if token_file.exists():
                logger.warning("Saved session token failed (%s), removing and re-attempting...", e)
                try:
                    token_file.unlink(missing_ok=True)
                    self.garmin = Garmin(self.email, self.password)
                    self.garmin.login(tokenstore=str(token_file))
                    return
                except Exception as e2:
                    raise e2
            raise e

    def get_activities(self, start_date: str, end_date: str) -> List[Dict]:
        """Fetch list of activities between start_date and end_date (YYYY-MM-DD)."""
        if self.garmin is None:
            self.login()

        clean_start = str(start_date).split("T")[0]
        clean_end = str(end_date).split("T")[0]

        logger.info("Querying Garmin activities from %s to %s", clean_start, clean_end)
        activities = []
        try:
            activities = self.garmin.get_activities_by_date(clean_start, clean_end)
        except Exception as e:
            logger.warning("get_activities_by_date failed (%s), attempting fallback...", e)

        if not activities:
            try:
                # Fallback: get up to 100 recent activities and filter locally by date
                recent = self.garmin.get_activities(0, 100)
                if recent:
                    activities = [
                        a
                        for a in recent
                        if clean_start
                        <= str(a.get("startTimeLocal") or a.get("startDate") or "")[:10]
                        <= clean_end
                    ]
                    logger.info(
                        "Found %d activities in date range using recent activities fallback.",
                        len(activities),
                    )
            except Exception as e:
                logger.warning("Recent activities fallback failed: %s", e)

        return activities or []

    def download_and_parse_activity(self, activity_id: Union[str, int]) -> pd.DataFrame:
        """Download original FIT file from Garmin and parse into DataFrame."""
        if self.garmin is None:
            self.login()

        act_id_int = int(str(activity_id).lstrip("iI"))
        zip_bytes = self.garmin.download_activity(
            act_id_int, dl_fmt=Garmin.ActivityDownloadFormat.ORIGINAL
        )
        return parse_fit_activity(zip_bytes, activity_id=str(activity_id))

    def fetch_all_streams(
        self,
        start_date: str,
        end_date: str,
        force_refresh: bool = False,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> pd.DataFrame:
        """Fetch, download, and pool streams across all Garmin activities in date range."""
        clean_email = "".join(c if c.isalnum() else "_" for c in self.email)
        clean_start = str(start_date).split("T")[0]
        clean_end = str(end_date).split("T")[0]
        cache_path = self.cache_dir / f"garmin_{clean_email}_{clean_start}_{clean_end}.parquet"

        if not force_refresh and cache_path.exists():
            try:
                cached = pd.read_parquet(cache_path)
                if not cached.empty:
                    logger.info("Loaded %d cached Garmin rows from %s", len(cached), cache_path)
                    return cached
            except Exception as e:
                logger.warning("Failed reading cache: %s", e)

        activities = self.get_activities(start_date, end_date)
        activity_dfs = []
        total_acts = len(activities)

        for idx, act in enumerate(activities):
            act_id = act.get("activityId") or act.get("activity_id")
            act_name = act.get("activityName") or f"Activity {act_id}"

            if not act_id:
                continue

            if progress_callback:
                progress_callback(
                    float(idx) / max(1, total_acts),
                    f"Downloading FIT for {act_name} ({idx + 1}/{total_acts})...",
                )

            try:
                df_act = self.download_and_parse_activity(act_id)
                if not df_act.empty:
                    activity_dfs.append(df_act)
            except Exception as e:
                logger.warning("Error fetching Garmin activity %s: %s", act_id, e)

        if progress_callback:
            progress_callback(1.0, "Pooling Garmin activity streams...")

        if activity_dfs:
            combined_df = pd.concat(activity_dfs, ignore_index=True)
            try:
                combined_df.to_parquet(cache_path, index=False)
                logger.info("Saved %d rows to %s", len(combined_df), cache_path)
            except Exception as e:
                logger.warning("Failed saving cache: %s", e)
        else:
            combined_df = pd.DataFrame(
                columns=["timestamp", "activity_id", "watts", "heartrate", "icu_blood_glucose"]
            )

        return combined_df
