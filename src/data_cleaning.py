"""Data validation and cleaning pipeline.

Two independent sources are supported:

* Our World in Data (OWID) country/date panel.
* covid19india.org state/date cumulative archive.

The functions in this module are pure (no Streamlit, no network) so they can be
unit-tested and reused by the build script.  Source data is never modified in
place: every function returns a new frame plus an auditable cleaning log.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
REQUIRED_OWID_COLUMNS: tuple[str, ...] = ("country", "date", "total_cases", "total_deaths")
REQUIRED_INDIA_COLUMNS: tuple[str, ...] = ("Date", "State", "Confirmed", "Recovered", "Deceased")

NUMERIC_OWID_COLUMNS: tuple[str, ...] = (
    "total_cases",
    "new_cases",
    "new_cases_smoothed",
    "total_cases_per_million",
    "new_cases_per_million",
    "new_cases_smoothed_per_million",
    "total_deaths",
    "new_deaths",
    "new_deaths_smoothed",
    "total_deaths_per_million",
    "new_deaths_per_million",
    "new_deaths_smoothed_per_million",
    "excess_mortality_cumulative_per_million",
    "stringency_index",
    "reproduction_rate",
    "total_tests",
    "new_tests",
    "positive_rate",
    "total_vaccinations",
    "people_vaccinated",
    "people_fully_vaccinated",
    "total_boosters",
    "new_vaccinations_smoothed",
    "total_vaccinations_per_hundred",
    "people_vaccinated_per_hundred",
    "people_fully_vaccinated_per_hundred",
    "total_boosters_per_hundred",
    "population",
    "population_density",
    "median_age",
    "life_expectancy",
    "gdp_per_capita",
    "hospital_beds_per_thousand",
    "human_development_index",
)

#: Columns persisted in the processed OWID dataset.
OWID_PROCESSED_COLUMNS: tuple[str, ...] = (
    "country",
    "iso_code",
    "continent",
    "date",
    "total_cases",
    "new_cases",
    "new_cases_smoothed",
    "new_cases_7d",
    "total_cases_per_million",
    "new_cases_per_million",
    "new_cases_smoothed_per_million",
    "total_deaths",
    "new_deaths",
    "new_deaths_smoothed",
    "new_deaths_7d",
    "total_deaths_per_million",
    "new_deaths_per_million",
    "new_deaths_smoothed_per_million",
    "case_fatality_ratio",
    "excess_mortality_cumulative_per_million",
    "stringency_index",
    "reproduction_rate",
    "total_tests",
    "new_tests",
    "positive_rate",
    "total_vaccinations",
    "people_vaccinated",
    "people_fully_vaccinated",
    "total_boosters",
    "new_vaccinations_smoothed",
    "vaccination_rate",
    "fully_vaccinated_rate",
    "booster_rate",
    "population",
    "population_density",
    "median_age",
    "life_expectancy",
    "gdp_per_capita",
    "hospital_beds_per_thousand",
    "human_development_index",
    "entity_type",
    "is_country",
    "revision_flag",
)

INDIA_PROCESSED_COLUMNS: tuple[str, ...] = (
    "date",
    "state",
    "entity_type",
    "is_state",
    "total_cases",
    "new_cases",
    "new_cases_7d",
    "total_deaths",
    "new_deaths",
    "new_deaths_7d",
    "total_recovered",
    "new_recovered",
    "active_cases",
    "case_fatality_ratio",
    "recovery_ratio",
    "revision_flag",
)

#: covid19india spellings -> Survey-of-India GeoJSON feature names.
INDIA_STATE_TO_GEO: dict[str, list[str]] = {
    "Andaman and Nicobar Islands": ["ANDAMAN & NICOBAR"],
    "Andhra Pradesh": ["ANDHRA PRADESH"],
    "Arunachal Pradesh": ["ARUNACHAL PRADESH"],
    "Assam": ["ASSAM"],
    "Bihar": ["BIHAR"],
    "Chandigarh": ["CHANDIGARH"],
    "Chhattisgarh": ["CHHATTISGARH"],
    "Dadra and Nagar Haveli and Daman and Diu": ["DADRA & NAGAR HAVELI", "DAMAN & DIU"],
    "Delhi": ["DELHI"],
    "Goa": ["GOA"],
    "Gujarat": ["GUJARAT"],
    "Haryana": ["HARYANA"],
    "Himachal Pradesh": ["HIMACHAL PRADESH"],
    "Jammu and Kashmir": ["JAMMU & KASHMIR"],
    "Jharkhand": ["JHARKHAND"],
    "Karnataka": ["KARNATAKA"],
    "Kerala": ["KERALA"],
    "Ladakh": ["LADAKH"],
    "Lakshadweep": ["LAKSHADWEEP"],
    "Madhya Pradesh": ["MADHYA PRADESH"],
    "Maharashtra": ["MAHARASHTRA"],
    "Manipur": ["MANIPUR"],
    "Meghalaya": ["MEGHALAYA"],
    "Mizoram": ["MIZORAM"],
    "Nagaland": ["NAGALAND"],
    "Odisha": ["ODISHA"],
    "Puducherry": ["PUDUCHERRY"],
    "Punjab": ["PUNJAB"],
    "Rajasthan": ["RAJASTHAN"],
    "Sikkim": ["SIKKIM"],
    "Tamil Nadu": ["TAMIL NADU"],
    "Telangana": ["TELANGANA"],
    "Tripura": ["TRIPURA"],
    "Uttar Pradesh": ["UTTAR PRADESH"],
    "Uttarakhand": ["UTTARAKHAND"],
    "West Bengal": ["WEST BENGAL"],
}


class DataSchemaError(ValueError):
    """Raised when a source file does not look like the expected dataset."""


# --------------------------------------------------------------------------- #
# Generic helpers
# --------------------------------------------------------------------------- #
def coerce_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Convert present columns to numeric, turning invalid strings into NaN."""
    out = df.copy()
    for column in columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def add_rolling_average(
    df: pd.DataFrame,
    value_col: str,
    *,
    group_col: str,
    out_col: str,
    window: int = 7,
    min_periods: int = 1,
) -> pd.DataFrame:
    """Add a per-entity trailing moving average.

    The frame is expected to be sorted by ``group_col`` and date.  The original
    index is preserved.  ``min_periods=1`` keeps the first days visible instead
    of silently dropping them.
    """
    if value_col not in df.columns:
        raise KeyError(f"Column not found: {value_col}")
    out = df.copy()
    grouped = out.groupby(group_col, sort=False)[value_col]
    out[out_col] = grouped.transform(
        lambda series: series.rolling(window=window, min_periods=min_periods).mean()
    )
    return out


def derive_daily_from_cumulative(
    df: pd.DataFrame,
    cumulative_col: str,
    *,
    group_col: str,
    out_col: str,
) -> pd.DataFrame:
    """Compute daily values from cumulative totals, keeping revisions.

    Negative differences are deliberately retained: they represent downward
    reporting revisions and are flagged downstream instead of being hidden.
    """
    if cumulative_col not in df.columns:
        raise KeyError(f"Column not found: {cumulative_col}")
    out = df.copy()
    grouped = out.groupby(group_col, sort=False)[cumulative_col]
    daily = grouped.diff()
    first = daily.isna() & out[cumulative_col].notna()
    daily = daily.mask(first, out[cumulative_col])
    out[out_col] = daily
    return out


def compute_case_fatality(deaths: pd.Series, cases: pd.Series) -> pd.Series:
    """Deaths / confirmed * 100 with safe zero and missing handling."""
    deaths_n = pd.to_numeric(deaths, errors="coerce")
    cases_n = pd.to_numeric(cases, errors="coerce")
    ratio = np.where((cases_n > 0) & cases_n.notna() & deaths_n.notna(), deaths_n / cases_n * 100.0, np.nan)
    return pd.Series(ratio, index=cases.index, dtype="float64")


def compute_recovery_ratio(recovered: pd.Series, cases: pd.Series) -> pd.Series:
    recovered_n = pd.to_numeric(recovered, errors="coerce")
    cases_n = pd.to_numeric(cases, errors="coerce")
    ratio = np.where(
        (cases_n > 0) & cases_n.notna() & recovered_n.notna(),
        recovered_n / cases_n * 100.0,
        np.nan,
    )
    return pd.Series(ratio, index=cases.index, dtype="float64")


def validate_owid_schema(df: pd.DataFrame) -> list[str]:
    """Return a list of missing required columns (empty means valid)."""
    return [column for column in REQUIRED_OWID_COLUMNS if column not in df.columns]


def validate_india_schema(df: pd.DataFrame) -> list[str]:
    return [column for column in REQUIRED_INDIA_COLUMNS if column not in df.columns]


def standardize_state_name(name: str) -> str:
    """Normalise whitespace and known covid19india spellings."""
    text = re.sub(r"\s+", " ", str(name)).strip()
    replacements = {
        "Orissa": "Odisha",
        "Uttaranchal": "Uttarakhand",
        "Pondicherry": "Puducherry",
        "NCT of Delhi": "Delhi",
        "Jammu & Kashmir": "Jammu and Kashmir",
        "Andaman & Nicobar Islands": "Andaman and Nicobar Islands",
        "Dadra and Nagar Haveli": "Dadra and Nagar Haveli and Daman and Diu",
    }
    return replacements.get(text, text)


# --------------------------------------------------------------------------- #
# OWID pipeline
# --------------------------------------------------------------------------- #
def prepare_owid(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean and enrich the OWID country/date panel.

    Returns
    -------
    processed, log
        The cleaned frame and a JSON-serialisable cleaning report.
    """
    missing = validate_owid_schema(raw)
    if missing:
        raise DataSchemaError(
            "The OWID file is missing required columns: " + ", ".join(missing)
        )

    actions: list[dict[str, Any]] = []
    working = raw.copy()

    # 1. Dates -------------------------------------------------------------
    working["date"] = pd.to_datetime(working["date"], errors="coerce")
    invalid_dates = int(working["date"].isna().sum())
    if invalid_dates:
        working = working[working["date"].notna()]
    actions.append(
        {
            "step": "Parse dates",
            "detail": "Converted the date column to datetime; invalid dates were dropped.",
            "affected_rows": invalid_dates,
        }
    )

    pre_2020 = int((working["date"] < pd.Timestamp("2020-01-01")).sum())
    if pre_2020:
        working = working[working["date"] >= pd.Timestamp("2020-01-01")]
    actions.append(
        {
            "step": "Restrict period",
            "detail": "Kept records from 1 January 2020 onwards.",
            "affected_rows": pre_2020,
        }
    )

    # 2. Types -------------------------------------------------------------
    working = coerce_numeric(working, NUMERIC_OWID_COLUMNS)
    working["country"] = working["country"].astype("string").str.strip()
    if "code" in working.columns:
        working["code"] = working["code"].astype("string").str.strip()
    if "continent" in working.columns:
        working["continent"] = working["continent"].astype("string").str.strip()
    actions.append(
        {
            "step": "Validate numeric fields",
            "detail": "Coerced numeric columns; non-numeric entries become NaN (not zero).",
            "affected_rows": 0,
        }
    )

    # 3. Duplicates --------------------------------------------------------
    duplicate_rows = int(working.duplicated(subset=["country", "date"], keep="last").sum())
    if duplicate_rows:
        working = working.sort_values(["country", "date"]).drop_duplicates(
            subset=["country", "date"], keep="last"
        )
    actions.append(
        {
            "step": "Remove duplicates",
            "detail": "De-duplicated on country + date, keeping the latest record.",
            "affected_rows": duplicate_rows,
        }
    )

    working = working.sort_values(["country", "date"]).reset_index(drop=True)

    # 4. Entity classification --------------------------------------------
    working["country"] = working["country"].fillna("Unknown")
    if "continent" not in working.columns:
        working["continent"] = pd.NA
    working["is_country"] = working["continent"].notna() & (working["country"] != "World")
    working["entity_type"] = np.where(working["is_country"], "Country", "Aggregate")

    # 5. Standardise key derived fields -----------------------------------
    if "code" not in working.columns:
        working["code"] = pd.NA
    working = working.rename(columns={"code": "iso_code"})

    if "total_cases_per_million" not in working.columns:
        working["total_cases_per_million"] = np.nan
    if "total_deaths_per_million" not in working.columns:
        working["total_deaths_per_million"] = np.nan

    valid_pop = working["population"].gt(0) if "population" in working.columns else False
    if "population" in working.columns:
        computed_cases_pm = np.where(
            valid_pop & working["total_cases"].notna(),
            working["total_cases"] / working["population"] * 1_000_000,
            np.nan,
        )
        computed_deaths_pm = np.where(
            valid_pop & working["total_deaths"].notna(),
            working["total_deaths"] / working["population"] * 1_000_000,
            np.nan,
        )
        working["total_cases_per_million"] = working["total_cases_per_million"].fillna(
            pd.Series(computed_cases_pm, index=working.index)
        )
        working["total_deaths_per_million"] = working["total_deaths_per_million"].fillna(
            pd.Series(computed_deaths_pm, index=working.index)
        )

    working["case_fatality_ratio"] = compute_case_fatality(
        working["total_deaths"], working["total_cases"]
    )

    # vaccination columns with explicit, stable names
    if "people_vaccinated_per_hundred" in working.columns:
        working["vaccination_rate"] = working["people_vaccinated_per_hundred"]
    else:
        working["vaccination_rate"] = np.nan
    if "people_fully_vaccinated_per_hundred" in working.columns:
        working["fully_vaccinated_rate"] = working["people_fully_vaccinated_per_hundred"]
    else:
        working["fully_vaccinated_rate"] = np.nan
    if "total_boosters_per_hundred" in working.columns:
        working["booster_rate"] = working["total_boosters_per_hundred"]
    else:
        working["booster_rate"] = np.nan

    # 6. Seven-day averages ------------------------------------------------
    working = add_rolling_average(
        working, "new_cases", group_col="country", out_col="_new_cases_roll"
    )
    working = add_rolling_average(
        working, "new_deaths", group_col="country", out_col="_new_deaths_roll"
    )
    if "new_cases_smoothed" in working.columns:
        working["new_cases_7d"] = working["new_cases_smoothed"].fillna(
            working["_new_cases_roll"]
        )
    else:
        working["new_cases_7d"] = working["_new_cases_roll"]
    if "new_deaths_smoothed" in working.columns:
        working["new_deaths_7d"] = working["new_deaths_smoothed"].fillna(
            working["_new_deaths_roll"]
        )
    else:
        working["new_deaths_7d"] = working["_new_deaths_roll"]
    actions.append(
        {
            "step": "Calculate 7-day averages",
            "detail": (
                "Used OWID's smoothed daily series where supplied; otherwise a "
                "trailing 7-day mean of reported daily values was computed."
            ),
            "affected_rows": int(working["new_cases_7d"].notna().sum()),
        }
    )

    # 7. Reporting revisions ----------------------------------------------
    working["revision_flag"] = (
        working["new_cases"].lt(0).fillna(False) | working["new_deaths"].lt(0).fillna(False)
    )
    revision_count = int(working["revision_flag"].sum())
    actions.append(
        {
            "step": "Flag negative daily values",
            "detail": (
                "Negative daily counts are retained as reporting revisions and "
                "marked with revision_flag; they are not silently replaced."
            ),
            "affected_rows": revision_count,
        }
    )

    # 8. Drop fully empty indicator rows ----------------------------------
    indicator_cols = [
        column
        for column in (
            "total_cases",
            "total_deaths",
            "total_vaccinations",
            "people_vaccinated",
            "total_tests",
        )
        if column in working.columns
    ]
    empty_rows = int(working[indicator_cols].isna().all(axis=1).sum())
    if empty_rows:
        working = working[~working[indicator_cols].isna().all(axis=1)].copy()
    actions.append(
        {
            "step": "Drop empty records",
            "detail": "Removed rows with no cases, deaths, tests or vaccination data.",
            "affected_rows": empty_rows,
        }
    )

    for column in OWID_PROCESSED_COLUMNS:
        if column not in working.columns:
            working[column] = np.nan
    processed = working.loc[:, OWID_PROCESSED_COLUMNS].reset_index(drop=True)

    log = {
        "actions": actions,
        "input_rows": int(len(raw)),
        "output_rows": int(len(processed)),
        "revision_rows": revision_count,
        "countries": int(processed.loc[processed["is_country"], "country"].nunique()),
        "aggregates": int(
            processed.loc[~processed["is_country"].astype(bool), "country"].nunique()
        ),
        "date_min": _iso(processed["date"].min()),
        "date_max": _iso(processed["date"].max()),
    }
    return processed, log


# --------------------------------------------------------------------------- #
# India pipeline
# --------------------------------------------------------------------------- #
def prepare_india(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean the covid19india state-level cumulative archive."""
    missing = validate_india_schema(raw)
    if missing:
        raise DataSchemaError(
            "The India file is missing required columns: " + ", ".join(missing)
        )

    actions: list[dict[str, Any]] = []
    working = raw.copy()
    working["Date"] = pd.to_datetime(working["Date"], errors="coerce")
    invalid_dates = int(working["Date"].isna().sum())
    working = working[working["Date"].notna()].copy()

    working["State"] = working["State"].astype(str).map(standardize_state_name)
    working = coerce_numeric(
        working, ["Confirmed", "Recovered", "Deceased", "Other", "Tested"]
    )
    working = working.rename(
        columns={
            "Date": "date",
            "State": "state",
            "Confirmed": "total_cases",
            "Recovered": "total_recovered",
            "Deceased": "total_deaths",
            "Other": "other",
            "Tested": "tests",
        }
    )

    duplicate_rows = int(working.duplicated(subset=["state", "date"], keep="last").sum())
    working = working.sort_values(["state", "date"]).drop_duplicates(
        subset=["state", "date"], keep="last"
    )
    actions.append(
        {
            "step": "Parse and de-duplicate",
            "detail": "Parsed dates, standardised state names and de-duplicated state + date.",
            "affected_rows": duplicate_rows + invalid_dates,
        }
    )

    # Complete daily grid per state; source only records days with changes.
    date_min, date_max = working["date"].min(), working["date"].max()
    dates = pd.date_range(date_min, date_max, freq="D")
    states = sorted(working["state"].unique())
    grid = pd.MultiIndex.from_product([states, dates], names=["state", "date"])
    working = (
        working.set_index(["state", "date"])
        .reindex(grid)
        .reset_index()
        .sort_values(["state", "date"])
    )

    cumulative_cols = ["total_cases", "total_recovered", "total_deaths", "other", "tests"]
    for column in cumulative_cols:
        working[column] = working.groupby("state", sort=False)[column].ffill()
    for column in ["total_cases", "total_recovered", "total_deaths", "other"]:
        working[column] = working[column].fillna(0.0)
    actions.append(
        {
            "step": "Forward-fill cumulative totals",
            "detail": (
                "Built a complete daily grid and forward-filled cumulative values; "
                "pre-reporting days are zero."
            ),
            "affected_rows": int(len(working)),
        }
    )

    working = derive_daily_from_cumulative(
        working, "total_cases", group_col="state", out_col="new_cases"
    )
    working = derive_daily_from_cumulative(
        working, "total_deaths", group_col="state", out_col="new_deaths"
    )
    working = derive_daily_from_cumulative(
        working, "total_recovered", group_col="state", out_col="new_recovered"
    )

    working = add_rolling_average(
        working, "new_cases", group_col="state", out_col="new_cases_7d"
    )
    working = add_rolling_average(
        working, "new_deaths", group_col="state", out_col="new_deaths_7d"
    )

    working["active_cases"] = (
        working["total_cases"] - working["total_recovered"] - working["total_deaths"]
    ).clip(lower=0)
    working["case_fatality_ratio"] = compute_case_fatality(
        working["total_deaths"], working["total_cases"]
    )
    working["recovery_ratio"] = compute_recovery_ratio(
        working["total_recovered"], working["total_cases"]
    )
    working["revision_flag"] = (
        working["new_cases"].lt(0).fillna(False) | working["new_deaths"].lt(0).fillna(False)
    )

    def entity_type(state: str) -> str:
        if state == "India":
            return "National"
        if state == "State Unassigned":
            return "Unassigned"
        return "State/UT"

    working["entity_type"] = working["state"].map(entity_type)
    working["is_state"] = working["entity_type"].eq("State/UT")

    negative_days = int((working["new_cases"] < 0).sum())
    actions.append(
        {
            "step": "Derive daily values",
            "detail": (
                "Daily counts are differences of cumulative totals. Negative "
                "differences are retained as reporting revisions and flagged."
            ),
            "affected_rows": negative_days,
        }
    )

    testing_coverage = float(working["tests"].notna().mean() * 100) if "tests" in working else 0.0
    if "tests" in working:
        working = working.drop(columns=["tests"])
    actions.append(
        {
            "step": "Drop sparse testing column",
            "detail": (
                f"State-level testing was available for only {testing_coverage:.1f}% of "
                "state-days; it was excluded from the processed dataset."
            ),
            "affected_rows": 0,
        }
    )

    for column in INDIA_PROCESSED_COLUMNS:
        if column not in working.columns:
            working[column] = np.nan
    processed = working.loc[:, INDIA_PROCESSED_COLUMNS].reset_index(drop=True)

    log = {
        "actions": actions,
        "input_rows": int(len(raw)),
        "output_rows": int(len(processed)),
        "states": int(processed.loc[processed["is_state"], "state"].nunique()),
        "date_min": _iso(processed["date"].min()),
        "date_max": _iso(processed["date"].max()),
        "negative_daily_rows": negative_days,
        "testing_coverage_pct": round(testing_coverage, 2),
    }
    return processed, log


# --------------------------------------------------------------------------- #
# Quality reports
# --------------------------------------------------------------------------- #
def build_owid_quality_report(raw: pd.DataFrame, processed: pd.DataFrame) -> dict[str, Any]:
    """Build a JSON-serialisable data-quality report for the EDA page."""
    parsed_dates = pd.to_datetime(raw.get("date"), errors="coerce")
    invalid_dates = int(parsed_dates.isna().sum())
    duplicates = int(raw.duplicated(subset=["country", "date"]).sum()) if {
        "country",
        "date",
    }.issubset(raw.columns) else 0

    missing = raw.isna().sum()
    schema = []
    for column in raw.columns:
        schema.append(
            {
                "column": str(column),
                "dtype": str(raw[column].dtype),
                "non_null": int(raw[column].notna().sum()),
                "missing": int(missing[column]),
                "missing_pct": round(float(missing[column]) / max(len(raw), 1) * 100, 3),
                "unique": int(raw[column].nunique(dropna=True)),
            }
        )

    total_cases = pd.to_numeric(raw.get("total_cases"), errors="coerce")
    total_deaths = pd.to_numeric(raw.get("total_deaths"), errors="coerce")
    negative_cases = int((total_cases < 0).sum())
    negative_deaths = int((total_deaths < 0).sum())
    deaths_gt_cases = int((total_deaths > total_cases).sum())
    negative_daily = int(
        (pd.to_numeric(raw.get("new_cases"), errors="coerce") < 0).sum()
    ) + int((pd.to_numeric(raw.get("new_deaths"), errors="coerce") < 0).sum())

    processed_dates = pd.to_datetime(processed["date"], errors="coerce")
    country_rows = processed[processed["is_country"].astype(bool)]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": int(len(raw)),
        "columns": int(raw.shape[1]),
        "memory_mb": round(float(raw.memory_usage(deep=True).sum()) / 1e6, 1),
        "date_min": _iso(parsed_dates.min()),
        "date_max": _iso(parsed_dates.max()),
        "duplicate_rows": duplicates,
        "invalid_dates": invalid_dates,
        "negative_cumulative_cases": negative_cases,
        "negative_cumulative_deaths": negative_deaths,
        "deaths_greater_than_cases": deaths_gt_cases,
        "negative_daily_values": negative_daily,
        "missing_cells": int(missing.sum()),
        "missing_pct": round(float(missing.sum()) / max(raw.size, 1) * 100, 2),
        "schema": schema,
        "processed_rows": int(len(processed)),
        "processed_columns": int(processed.shape[1]),
        "countries": int(country_rows["country"].nunique()),
        "entities": int(processed["country"].nunique()),
        "processed_date_min": _iso(processed_dates.min()),
        "processed_date_max": _iso(processed_dates.max()),
        "processed_missing_pct": round(
            float(processed.isna().sum().sum()) / max(processed.size, 1) * 100, 2
        ),
    }


def build_india_quality_report(raw: pd.DataFrame, processed: pd.DataFrame) -> dict[str, Any]:
    missing = raw.isna().sum()
    duplicates = int(raw.duplicated(subset=["Date", "State"]).sum()) if {
        "Date",
        "State",
    }.issubset(raw.columns) else 0
    state_rows = processed[processed["is_state"].astype(bool)]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": int(len(raw)),
        "columns": int(raw.shape[1]),
        "memory_mb": round(float(raw.memory_usage(deep=True).sum()) / 1e6, 2),
        "date_min": _iso(pd.to_datetime(raw["Date"], errors="coerce").min()),
        "date_max": _iso(pd.to_datetime(raw["Date"], errors="coerce").max()),
        "duplicate_rows": duplicates,
        "states": int(state_rows["state"].nunique()),
        "entities": int(processed["state"].nunique()),
        "processed_rows": int(len(processed)),
        "missing_cells": int(missing.sum()),
        "missing_pct": round(float(missing.sum()) / max(raw.size, 1) * 100, 2),
        "schema": [
            {
                "column": str(column),
                "dtype": str(raw[column].dtype),
                "non_null": int(raw[column].notna().sum()),
                "missing": int(missing[column]),
                "missing_pct": round(float(missing[column]) / max(len(raw), 1) * 100, 3),
                "unique": int(raw[column].nunique(dropna=True)),
            }
            for column in raw.columns
        ],
        "recoveries_coverage_pct": round(
            float(processed["total_recovered"].notna().mean() * 100), 2
        ),
        "negative_daily_rows": int(processed["revision_flag"].sum()),
        "note": (
            "covid19india.org was a volunteer-led aggregator. The public CSV "
            "archive available for this project ends on 1 June 2021; values are "
            "reported, retrospective revisions are possible."
        ),
    }


def combine_quality_reports(
    owid_report: dict[str, Any], india_report: dict[str, Any]
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "owid": owid_report,
        "india": india_report,
    }


# --------------------------------------------------------------------------- #
# Small utilities
# --------------------------------------------------------------------------- #
def _iso(value: Any) -> str | None:
    try:
        stamp = pd.Timestamp(value)
        if pd.isna(stamp):
            return None
        return stamp.strftime("%Y-%m-%d")
    except (TypeError, ValueError):
        return None


def prepare_india_map_data(snapshot: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Explode state snapshot rows onto the GeoJSON feature names.

    The combined Dadra & Nagar Haveli and Daman & Diu entity is drawn on both
    historical polygons; this is the only honest option with a pre-merger map.
    """
    if snapshot.empty:
        return snapshot.iloc[0:0].assign(geo_state=pd.Series(dtype="object"))
    rows: list[dict[str, Any]] = []
    for _, row in snapshot.iterrows():
        state = row["state"]
        for geo_name in INDIA_STATE_TO_GEO.get(state, [state.upper()]):
            record = row.to_dict()
            record["geo_state"] = geo_name
            rows.append(record)
    mapped = pd.DataFrame(rows)
    return mapped
