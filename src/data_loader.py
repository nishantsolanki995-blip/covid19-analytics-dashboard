"""Dataset download, caching and processed-data build utilities.

The dashboard never claims to be a live feed.  The raw files are archived
historical downloads; the processed files are deterministic derivatives and can
be rebuilt with ``python -m src.data_loader --build``.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import requests
import streamlit as st

from src.data_cleaning import (
    DataSchemaError,
    OWID_PROCESSED_COLUMNS,
    build_india_quality_report,
    build_owid_quality_report,
    combine_quality_reports,
    prepare_india,
    prepare_owid,
    validate_owid_schema,
)

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

OWID_RAW_PATH = RAW_DIR / "owid_covid_compact.csv"
INDIA_RAW_PATH = RAW_DIR / "india_states.csv"
GEO_PATH = RAW_DIR / "india_states.geojson"
OWID_PROCESSED_PATH = PROCESSED_DIR / "owid_country_daily.csv.gz"
INDIA_PROCESSED_PATH = PROCESSED_DIR / "india_states_daily.csv.gz"
QUALITY_PATH = PROCESSED_DIR / "data_quality.json"

OWID_URL = "https://catalog.ourworldindata.org/garden/covid/latest/compact/compact.csv"
OWID_DOCS = "https://docs.owid.io/projects/etl/api/covid/"
INDIA_URL = "https://api.covid19india.org/csv/latest/states.csv"
INDIA_DOCS = "https://data.covid19india.org/"
GEO_URL = (
    "https://raw.githubusercontent.com/datta07/INDIAN-SHAPEFILES/"
    "master/INDIA/INDIA_STATES.geojson"
)

SOURCE_INFO = {
    "owid": {
        "name": "Our World in Data — COVID-19 Dataset",
        "url": OWID_URL,
        "docs": OWID_DOCS,
        "licence": "CC BY 4.0",
        "description": (
            "Country/date panel of confirmed cases, deaths, testing, vaccination "
            "and contextual indicators, compiled by Our World in Data."
        ),
    },
    "india": {
        "name": "covid19india.org — State-wise archive",
        "url": INDIA_URL,
        "docs": INDIA_DOCS,
        "description": (
            "Volunteer-aggregated state/UT cumulative confirmed, recovered and "
            "deceased counts. This public CSV archive ends on 1 June 2021."
        ),
    },
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; COVIDAnalyticsDashboard/1.0; "
        "+https://github.com/) python-requests"
    )
}


class DataUnavailableError(RuntimeError):
    """Raised when a required historical file cannot be obtained."""


# --------------------------------------------------------------------------- #
# Downloads
# --------------------------------------------------------------------------- #
def _human_size(num_bytes: float) -> str:
    for divisor, suffix in ((1_000_000_000, "GB"), (1_000_000, "MB"), (1_000, "KB")):
        if num_bytes >= divisor:
            return f"{num_bytes / divisor:.1f} {suffix}"
    return f"{num_bytes:.0f} B"


def download_file(
    url: str,
    destination: Path,
    *,
    min_bytes: int = 5_000,
    timeout: tuple[int, int] = (20, 600),
    label: str = "dataset",
) -> Path:
    """Stream a URL to *destination* atomically (``.part`` then rename)."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with requests.get(url, stream=True, timeout=timeout, headers=HEADERS) as response:
            response.raise_for_status()
            expected = response.headers.get("Content-Length")
            written = 0
            with temporary.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
                        written += len(chunk)
            if expected is not None:
                try:
                    expected_bytes = int(expected)
                except ValueError:
                    expected_bytes = 0
                if expected_bytes and written != expected_bytes:
                    raise DataUnavailableError(
                        f"Incomplete {label} download: {_human_size(written)} of "
                        f"{_human_size(expected_bytes)} received."
                    )
        if temporary.stat().st_size < min_bytes:
            raise DataUnavailableError(
                f"The downloaded {label} file is unexpectedly small "
                f"({_human_size(temporary.stat().st_size)})."
            )
        os.replace(temporary, destination)
        return destination
    except requests.exceptions.SSLError as exc:
        raise DataUnavailableError(
            "A TLS certificate error occurred while downloading the dataset. "
            "Check your system certificates or download the CSV manually."
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise DataUnavailableError(f"Could not download the {label}: {exc}") from exc
    finally:
        if temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                pass


def ensure_owid_raw(force: bool = False) -> Path:
    if force or not OWID_RAW_PATH.exists() or OWID_RAW_PATH.stat().st_size < 5_000:
        try:
            download_file(OWID_URL, OWID_RAW_PATH, label="OWID COVID-19 dataset")
        except DataUnavailableError as exc:
            raise DataUnavailableError(
                "The OWID historical dataset could not be downloaded.\n"
                f"Download it manually from {OWID_URL}\n"
                f"and save it as {OWID_RAW_PATH}.\n\nDetails: {exc}"
            ) from exc
    return OWID_RAW_PATH


def ensure_india_raw(force: bool = False) -> Path:
    if force or not INDIA_RAW_PATH.exists() or INDIA_RAW_PATH.stat().st_size < 5_000:
        try:
            download_file(INDIA_URL, INDIA_RAW_PATH, label="covid19india state archive")
        except DataUnavailableError as exc:
            raise DataUnavailableError(
                "The covid19india state-level archive is not available.\n"
                f"The public endpoint is {INDIA_URL}.\n"
                f"Place a compatible states.csv at {INDIA_RAW_PATH} to run the India page.\n\n"
                f"Details: {exc}"
            ) from exc
    return INDIA_RAW_PATH


def ensure_geojson(force: bool = False) -> Path:
    if force or not GEO_PATH.exists() or GEO_PATH.stat().st_size < 10_000:
        try:
            download_file(GEO_URL, GEO_PATH, label="India state boundaries", min_bytes=100_000)
        except DataUnavailableError:
            # The map degrades gracefully; the rest of the India page still works.
            return GEO_PATH
    return GEO_PATH


# --------------------------------------------------------------------------- #
# Pure builders (used by CLI and tests)
# --------------------------------------------------------------------------- #
def load_owid_raw(usecols: Sequence[str] | None = None) -> pd.DataFrame:
    path = ensure_owid_raw()
    try:
        frame = pd.read_csv(path, usecols=usecols, low_memory=False)
    except ValueError as exc:
        raise DataSchemaError(
            "The OWID CSV does not contain the expected columns. "
            "Delete data/raw/owid_covid_compact.csv and download it again."
        ) from exc
    missing = validate_owid_schema(frame)
    if missing:
        raise DataSchemaError(
            "The OWID CSV is missing required columns: " + ", ".join(missing)
        )
    return frame


def build_owid_processed(
    raw: pd.DataFrame | None = None,
    *,
    write: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if raw is None:
        raw = load_owid_raw()
    processed, log = prepare_owid(raw)
    if write:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        processed.to_csv(OWID_PROCESSED_PATH, index=False, compression="gzip")
    return processed, log


def build_india_processed(
    raw: pd.DataFrame | None = None,
    *,
    write: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if raw is None:
        path = ensure_india_raw()
        raw = pd.read_csv(path)
    processed, log = prepare_india(raw)
    if write:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        processed.to_csv(INDIA_PROCESSED_PATH, index=False, compression="gzip")
    return processed, log


def build_quality_report(
    *,
    owid_raw: pd.DataFrame,
    owid_processed: pd.DataFrame,
    india_raw: pd.DataFrame,
    india_processed: pd.DataFrame,
    write: bool = True,
) -> dict[str, Any]:
    report = combine_quality_reports(
        build_owid_quality_report(owid_raw, owid_processed),
        build_india_quality_report(india_raw, india_processed),
    )
    if write:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        QUALITY_PATH.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return report


def build_all(force_download: bool = False, *, verbose: bool = True) -> dict[str, Any]:
    """Download (if needed), clean, persist and report on both datasets."""
    if verbose:
        print("COVID Analytics - preparing historical datasets")
        print("=" * 55)
    ensure_geojson(force=force_download)
    owid_path = ensure_owid_raw(force=force_download)
    india_path = ensure_india_raw(force=force_download)

    if verbose:
        print(f"Reading OWID raw file  ({_human_size(owid_path.stat().st_size)}) ...")
    owid_raw = pd.read_csv(owid_path, low_memory=False)
    if verbose:
        print("Cleaning OWID panel ...")
    owid_processed, owid_log = build_owid_processed(owid_raw)

    if verbose:
        print("Reading covid19india archive ...")
    india_raw = pd.read_csv(india_path)
    if verbose:
        print("Cleaning India state archive ...")
    india_processed, india_log = build_india_processed(india_raw)

    if verbose:
        print("Writing quality report ...")
    quality = build_quality_report(
        owid_raw=owid_raw,
        owid_processed=owid_processed,
        india_raw=india_raw,
        india_processed=india_processed,
    )

    summary = {
        "owid": owid_log,
        "india": india_log,
        "processed_files": {
            "owid": str(OWID_PROCESSED_PATH),
            "india": str(INDIA_PROCESSED_PATH),
            "quality": str(QUALITY_PATH),
        },
    }
    if verbose:
        print(
            f"OWID : {owid_log['output_rows']:,} rows, "
            f"{owid_log['countries']} countries, {owid_log['date_min']} to {owid_log['date_max']}"
        )
        print(
            f"INDIA: {india_log['output_rows']:,} rows, "
            f"{india_log['states']} states/UTs, {india_log['date_min']} to {india_log['date_max']}"
        )
        print("Done.")
    return summary


# --------------------------------------------------------------------------- #
# Cached readers
# --------------------------------------------------------------------------- #
def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


@st.cache_data(show_spinner=False, max_entries=2)
def _read_owid_cached(stamp: float) -> pd.DataFrame:
    frame = pd.read_csv(
        OWID_PROCESSED_PATH,
        parse_dates=["date"],
        low_memory=False,
    )
    if "is_country" in frame.columns:
        frame["is_country"] = frame["is_country"].astype(bool)
    return frame


def load_owid_data() -> pd.DataFrame:
    """Load the processed OWID panel, building it from raw data when needed."""
    if not OWID_PROCESSED_PATH.exists() or _mtime(OWID_PROCESSED_PATH) < _mtime(OWID_RAW_PATH):
        raw = load_owid_raw()
        build_owid_processed(raw)
        if not INDIA_PROCESSED_PATH.exists():
            build_india_processed()
    return _read_owid_cached(_mtime(OWID_PROCESSED_PATH))


@st.cache_data(show_spinner=False, max_entries=2)
def _read_india_cached(stamp: float) -> pd.DataFrame:
    frame = pd.read_csv(
        INDIA_PROCESSED_PATH,
        parse_dates=["date"],
    )
    for column in ("is_state",):
        if column in frame.columns:
            frame[column] = frame[column].astype(bool)
    return frame


def load_india_data() -> pd.DataFrame:
    if not INDIA_PROCESSED_PATH.exists():
        build_india_processed()
    return _read_india_cached(_mtime(INDIA_PROCESSED_PATH))


@st.cache_data(show_spinner=False, max_entries=2)
def _read_quality_cached(stamp: float) -> dict[str, Any]:
    return json.loads(QUALITY_PATH.read_text(encoding="utf-8"))


def load_quality_report() -> dict[str, Any] | None:
    if not QUALITY_PATH.exists():
        return None
    return _read_quality_cached(_mtime(QUALITY_PATH))


def ensure_quality_report() -> dict[str, Any] | None:
    """Return the quality report, building it once if the processed files exist.

    The report needs the raw columns for schema and missingness checks, so the
    first call may be slower.  Subsequent page loads read the cached JSON file.
    """
    report = load_quality_report()
    if report is not None:
        return report
    try:
        if not OWID_PROCESSED_PATH.exists():
            build_owid_processed()
        if not INDIA_PROCESSED_PATH.exists():
            build_india_processed()
        owid_processed = _read_owid_cached(_mtime(OWID_PROCESSED_PATH))
        india_processed = _read_india_cached(_mtime(INDIA_PROCESSED_PATH))
        owid_raw = load_owid_raw()
        india_raw = pd.read_csv(ensure_india_raw())
        return build_quality_report(
            owid_raw=owid_raw,
            owid_processed=owid_processed,
            india_raw=india_raw,
            india_processed=india_processed,
        )
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def _read_geojson_cached(stamp: float) -> dict[str, Any] | None:
    try:
        return json.loads(GEO_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def load_india_geojson() -> dict[str, Any] | None:
    ensure_geojson()
    if not GEO_PATH.exists():
        return None
    return _read_geojson_cached(_mtime(GEO_PATH))


def load_raw_preview(rows: int = 500) -> pd.DataFrame:
    """Read a small raw-file preview without loading the full 180 MB file."""
    path = ensure_owid_raw()
    return pd.read_csv(path, nrows=rows)


def dataset_status() -> dict[str, Any]:
    """Lightweight status used by the landing page and sidebar."""
    quality = load_quality_report() or {}
    owid = quality.get("owid", {})
    india = quality.get("india", {})
    return {
        "owid_ready": OWID_PROCESSED_PATH.exists() or OWID_RAW_PATH.exists(),
        "india_ready": INDIA_PROCESSED_PATH.exists() or INDIA_RAW_PATH.exists(),
        "owid_date_max": owid.get("date_max") or owid.get("processed_date_max"),
        "owid_date_min": owid.get("date_min"),
        "owid_rows": owid.get("processed_rows"),
        "owid_countries": owid.get("countries"),
        "india_date_min": india.get("date_min"),
        "india_date_max": india.get("date_max"),
        "india_states": india.get("states"),
        "quality": quality,
        "owid_source": SOURCE_INFO["owid"],
        "india_source": SOURCE_INFO["india"],
    }


if __name__ == "__main__":
    arguments = set(sys.argv[1:])
    force = "--force" in arguments
    if "--download" in arguments:
        path = ensure_owid_raw(force=force)
        print(f"OWID raw file ready: {path}")
    else:
        build_all(force_download=force)
