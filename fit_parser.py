"""FIT file parser extracting second-by-second power, HR, and CGM developer data fields."""

import gzip
import io
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import zipfile

import fitdecode
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def extract_fit_bytes(raw_bytes: bytes) -> bytes:
    """Decompress raw bytes if wrapped in a zip or gzip container.

    Args:
        raw_bytes: Raw file content from disk or download.

    Returns:
        Decompressed .fit binary bytes.
    """
    if not raw_bytes:
        return b""

    # GZIP format check
    if len(raw_bytes) > 2 and raw_bytes[:2] == b"\x1f\x8b":
        try:
            return gzip.decompress(raw_bytes)
        except Exception as e:
            logger.warning("Gzip decompression failed: %s", e)

    # ZIP format check (Garmin ActivityDownloadFormat.ORIGINAL usually returns a zip)
    if len(raw_bytes) > 4 and raw_bytes[:4] == b"PK\x03\x04":
        try:
            with zipfile.ZipFile(io.BytesIO(raw_bytes)) as z:
                # Find the first .fit file in the archive
                fit_names = [n for n in z.namelist() if n.lower().endswith(".fit")]
                if fit_names:
                    return z.read(fit_names[0])
                # If no file with .fit extension, take the first entry
                if z.namelist():
                    return z.read(z.namelist()[0])
        except Exception as e:
            logger.warning("Zip decompression failed: %s", e)

    return raw_bytes


def parse_fit_activity(
    source: Union[bytes, str, Path],
    activity_id: str = "fit_activity",
) -> pd.DataFrame:
    """Parse a Garmin FIT file and extract timestamp, power, heartrate, and CGM glucose.

    Connect IQ apps (Dexcom, xDrip+, Libre, Supersapiens, etc.) write blood glucose
    as FIT developer fields within 'record' messages.

    Args:
        source: File path, Path object, or raw bytes of a .fit / .zip / .fit.gz file.
        activity_id: Identifier to tag each row.

    Returns:
        DataFrame with columns ['timestamp', 'activity_id', 'watts', 'heartrate', 'icu_blood_glucose'].
    """
    if isinstance(source, (str, Path)):
        p = Path(source)
        if not p.exists():
            logger.warning("FIT file path %s does not exist.", p)
            return pd.DataFrame()
        with open(p, "rb") as f:
            raw_bytes = f.read()
    else:
        raw_bytes = source

    fit_bytes = extract_fit_bytes(raw_bytes)
    if not fit_bytes:
        return pd.DataFrame(
            columns=["timestamp", "activity_id", "watts", "heartrate", "icu_blood_glucose"]
        )

    timestamps = []
    watts_list = []
    hr_list = []
    glucose_list = []

    detected_glucose_field_name = None

    try:
        with fitdecode.FitReader(io.BytesIO(fit_bytes)) as fit:
            record_count = 0
            logged_sample_fields = False

            for frame in fit:
                # In fitdecode, frames can be FitHeader, FitDefinitionMessage, FitDataMessage, FitCRC.
                # Only FitDataMessage frames contain actual records.
                if not isinstance(frame, fitdecode.FitDataMessage):
                    continue

                if frame.name != "record":
                    continue

                record_count += 1

                # 1. Timestamp
                ts = frame.get_value("timestamp", fallback=None)
                if ts is None:
                    continue

                # 2. Power / Watts
                power = frame.get_value(
                    "power",
                    fallback=frame.get_value("watts", fallback=np.nan),
                )

                # 3. Heart Rate
                hr = frame.get_value(
                    "heart_rate",
                    fallback=frame.get_value("heartrate", fallback=np.nan),
                )

                # 4. Glucose / CGM (Developer fields or standard fields)
                bg_val = None

                # If we previously identified the field name, check it directly first
                if detected_glucose_field_name and frame.has_field(detected_glucose_field_name):
                    val = frame.get_value(detected_glucose_field_name)
                    if val is not None and isinstance(val, (int, float)) and val > 0:
                        bg_val = float(val)

                # Otherwise scan all fields in this record message
                if bg_val is None:
                    for field in frame.fields:
                        fname = str(field.name or field.name_or_num or "").lower()
                        funits = str(getattr(field, "units", "") or "").lower()

                        # Match developer or standard names for glucose
                        is_glucose_name = any(
                            k in fname
                            for k in [
                                "glucose",
                                "cgm",
                                "bloodsugar",
                                "blood_sugar",
                                "dexcom",
                                "libre",
                                "supersapiens",
                                "sensorglucose",
                                "gluc",
                            ]
                        ) or fname == "bg"

                        # Or match by units (e.g. mg/dL, mmol/L)
                        is_glucose_unit = "mg/dl" in funits or "mmol" in funits

                        if is_glucose_name or is_glucose_unit:
                            val = field.value
                            if val is not None and isinstance(val, (int, float)) and val > 0:
                                bg_val = float(val)
                                if not detected_glucose_field_name:
                                    detected_glucose_field_name = field.name or field.name_or_num
                                    logger.info(
                                        "Detected glucose field '%s' (units: '%s', sample: %s)",
                                        detected_glucose_field_name,
                                        funits,
                                        val,
                                    )
                                break

                # Diagnostic log of available fields on first records if glucose not yet found
                if not logged_sample_fields and record_count <= 5 and bg_val is None:
                    all_field_names = [f"{f.name or f.name_or_num}:{getattr(f, 'units', '')}" for f in frame.fields]
                    logger.debug("Record #%d field sample: %s", record_count, all_field_names)
                    if record_count == 5:
                        logged_sample_fields = True

                timestamps.append(ts)
                watts_list.append(float(power) if power is not None else np.nan)
                hr_list.append(float(hr) if hr is not None else np.nan)
                glucose_list.append(float(bg_val) if bg_val is not None else np.nan)

    except Exception as e:
        logger.error("Error parsing FIT frame stream for %s: %s", activity_id, e)

    if not timestamps:
        return pd.DataFrame(
            columns=["timestamp", "activity_id", "watts", "heartrate", "icu_blood_glucose"]
        )

    # Convert glucose from mmol/L to mg/dL if needed
    bg_series = pd.Series(glucose_list, dtype=float)
    valid_bg = bg_series.dropna()
    if not valid_bg.empty and valid_bg.median() < 30.0:
        logger.info(
            "FIT glucose appears to be in mmol/L (median %.1f); auto-converting to mg/dL (*18.0182)",
            valid_bg.median(),
        )
        bg_series = bg_series * 18.0182

    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamps),
            "activity_id": str(activity_id),
            "watts": pd.Series(watts_list, dtype=float),
            "heartrate": pd.Series(hr_list, dtype=float),
            "icu_blood_glucose": bg_series,
        }
    )

    n_with_bg = df["icu_blood_glucose"].notna().sum()
    logger.info(
        "Parsed FIT activity %s: %d total rows, %d rows with glucose data.",
        activity_id,
        len(df),
        n_with_bg,
    )

    return df
