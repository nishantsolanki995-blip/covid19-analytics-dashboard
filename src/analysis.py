"""Analytical computations used by every page.

Everything here is deterministic and operates on the processed frames.  The
functions avoid hard-coded statistics and handle missing values explicitly so
the UI can distinguish "zero" from "not reported".
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from src.data_cleaning import (
    compute_case_fatality,
    compute_recovery_ratio,
)

# --------------------------------------------------------------------------- #
# Metric metadata
# --------------------------------------------------------------------------- #
COUNTRY_METRICS: dict[str, dict[str, str]] = {
    "total_cases": {
        "label": "Cumulative confirmed cases",
        "short": "Confirmed cases",
        "unit": "people",
        "axis": "Reported cases",
    },
    "total_deaths": {
        "label": "Cumulative deaths",
        "short": "Deaths",
        "unit": "people",
        "axis": "Reported deaths",
    },
    "total_cases_per_million": {
        "label": "Cumulative cases per million",
        "short": "Cases per million",
        "unit": "per million",
        "axis": "Cases per million",
    },
    "total_deaths_per_million": {
        "label": "Cumulative deaths per million",
        "short": "Deaths per million",
        "unit": "per million",
        "axis": "Deaths per million",
    },
    "case_fatality_ratio": {
        "label": "Case fatality ratio",
        "short": "CFR",
        "unit": "%",
        "axis": "Deaths ÷ confirmed (%)",
    },
    "vaccination_rate": {
        "label": "At least one vaccine dose",
        "short": "Vaccinated (%)",
        "unit": "%",
        "axis": "% of population",
    },
    "fully_vaccinated_rate": {
        "label": "Fully vaccinated",
        "short": "Fully vaccinated (%)",
        "unit": "%",
        "axis": "% of population",
    },
    "new_cases_7d": {
        "label": "Daily new cases (7-day average)",
        "short": "New cases (7d avg)",
        "unit": "people/day",
        "axis": "Cases per day",
    },
    "new_deaths_7d": {
        "label": "Daily new deaths (7-day average)",
        "short": "New deaths (7d avg)",
        "unit": "people/day",
        "axis": "Deaths per day",
    },
}

INDIA_METRICS: dict[str, dict[str, str]] = {
    "total_cases": {"label": "Confirmed cases", "short": "Confirmed"},
    "total_deaths": {"label": "Deaths", "short": "Deaths"},
    "total_recovered": {"label": "Recoveries", "short": "Recovered"},
    "active_cases": {"label": "Active cases", "short": "Active"},
    "case_fatality_ratio": {"label": "Case fatality ratio", "short": "CFR"},
    "recovery_ratio": {"label": "Recovery ratio", "short": "Recovery ratio"},
    "new_cases": {"label": "Daily new cases", "short": "Daily cases"},
    "new_cases_7d": {"label": "Daily cases (7-day average)", "short": "Cases 7d"},
}

INDIA_STATE_METRICS = {
    "total_cases": "Confirmed cases",
    "total_deaths": "Deaths",
    "total_recovered": "Recoveries",
    "active_cases": "Active cases",
    "case_fatality_ratio": "Case fatality ratio",
    "recovery_ratio": "Recovery ratio",
}

SNAPSHOT_COLUMNS: tuple[str, ...] = (
    "total_cases",
    "total_deaths",
    "total_cases_per_million",
    "total_deaths_per_million",
    "case_fatality_ratio",
    "population",
    "population_density",
    "median_age",
    "life_expectancy",
    "gdp_per_capita",
    "hospital_beds_per_thousand",
    "human_development_index",
    "excess_mortality_cumulative_per_million",
    "vaccination_rate",
    "fully_vaccinated_rate",
    "booster_rate",
)


# --------------------------------------------------------------------------- #
# Basic helpers
# --------------------------------------------------------------------------- #
def safe_divide(numerator: Any, denominator: Any, multiplier: float = 1.0) -> float:
    try:
        n, d = float(numerator), float(denominator)
    except (TypeError, ValueError):
        return float("nan")
    if pd.isna(n) or pd.isna(d) or d == 0:
        return float("nan")
    return n / d * multiplier


def per_million(value: Any, population: Any) -> float:
    return safe_divide(value, population, 1_000_000)


def calculate_active(confirmed: Any, recovered: Any, deaths: Any) -> float:
    """Active = confirmed - recovered - deaths, clipped at zero.

    Returns NaN when any input is missing, because a partial estimate would be
    misleading.  The India page shows this metric only for the national series
    where all three inputs are reported.
    """
    try:
        c, r, d = float(confirmed), float(recovered), float(deaths)
    except (TypeError, ValueError):
        return float("nan")
    if any(pd.isna(v) for v in (c, r, d)):
        return float("nan")
    return max(c - r - d, 0.0)


def date_bounds(df: pd.DataFrame, date_col: str = "date") -> tuple[pd.Timestamp, pd.Timestamp]:
    if df.empty or date_col not in df.columns:
        today = pd.Timestamp.today().normalize()
        return today - pd.Timedelta(days=365), today
    values = pd.to_datetime(df[date_col], errors="coerce").dropna()
    if values.empty:
        today = pd.Timestamp.today().normalize()
        return today - pd.Timedelta(days=365), today
    return values.min().normalize(), values.max().normalize()


def filter_by_date(
    df: pd.DataFrame,
    start: Any | None = None,
    end: Any | None = None,
    date_col: str = "date",
) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df
    dates = pd.to_datetime(out[date_col], errors="coerce")
    if start is not None:
        out = out[dates >= pd.Timestamp(start)]
        dates = pd.to_datetime(out[date_col], errors="coerce")
    if end is not None:
        out = out[dates <= pd.Timestamp(end)]
    return out.copy()


def filter_dataframe(
    df: pd.DataFrame,
    *,
    entities: Sequence[str] | None = None,
    entity_col: str = "country",
    start: Any | None = None,
    end: Any | None = None,
    search: str | None = None,
    date_col: str = "date",
) -> pd.DataFrame:
    """Reusable filter used by the Data Explorer and by CSV export tests."""
    out = filter_by_date(df, start=start, end=end, date_col=date_col)
    if entities and entity_col in out.columns:
        out = out[out[entity_col].isin(list(entities))]
    if search and entity_col in out.columns:
        term = str(search).strip()
        if term:
            out = out[out[entity_col].astype(str).str.contains(term, case=False, na=False)]
    return out.copy()


def paginate(
    df: pd.DataFrame, page: int = 1, page_size: int = 50
) -> tuple[pd.DataFrame, dict[str, int]]:
    total_rows = int(len(df))
    page_size = max(int(page_size), 1)
    total_pages = max(1, math.ceil(total_rows / page_size))
    page = min(max(int(page), 1), total_pages)
    start = (page - 1) * page_size
    return df.iloc[start : start + page_size].copy(), {
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
        "total_rows": total_rows,
    }


# --------------------------------------------------------------------------- #
# Snapshots and rankings
# --------------------------------------------------------------------------- #
def latest_snapshot(
    df: pd.DataFrame,
    *,
    entity_col: str = "country",
    countries_only: bool = True,
    require: str = "total_cases",
) -> pd.DataFrame:
    """One latest-available row per entity, forward-filling cumulative fields.

    Using each country's last *reported* observation avoids treating a row that
    only contains vaccination or population data as the country's case count.
    """
    if df.empty or entity_col not in df.columns:
        return df.copy()
    working = df.copy()
    if countries_only and "is_country" in working.columns:
        working = working[working["is_country"].astype(bool)]
    working = working.sort_values([entity_col, "date"])

    metrics = [column for column in SNAPSHOT_COLUMNS if column in working.columns]
    if metrics:
        working[metrics] = working.groupby(entity_col, sort=False)[metrics].ffill()

    if require in working.columns:
        working = working[working[require].notna()]
    snapshot = working.groupby(entity_col, sort=False, as_index=False).tail(1).copy()

    if "vaccination_rate" in df.columns:
        vax = (
            df.dropna(subset=["vaccination_rate"])
            .groupby(entity_col, as_index=False)["date"]
            .max()
            .rename(columns={"date": "vaccination_as_of"})
        )
        snapshot = snapshot.merge(vax, on=entity_col, how="left")

    return snapshot.reset_index(drop=True)


def rank_entities(
    snapshot: pd.DataFrame,
    metric: str,
    *,
    entity_col: str = "country",
    n: int = 10,
    ascending: bool = False,
) -> pd.DataFrame:
    if metric not in snapshot.columns:
        return snapshot.iloc[0:0]
    ranked = snapshot.dropna(subset=[metric]).sort_values(metric, ascending=ascending)
    return ranked.head(n).copy()


def country_series(
    df: pd.DataFrame,
    countries: Sequence[str],
    metric: str,
    *,
    country_col: str = "country",
) -> pd.DataFrame:
    if df.empty or metric not in df.columns:
        return df.iloc[0:0]
    subset = df[df[country_col].isin(list(countries))].copy()
    return subset[[country_col, "date", metric]].dropna(subset=[metric]).sort_values(["country", "date"])


def align_by_days_since_threshold(
    df: pd.DataFrame,
    countries: Sequence[str],
    metric: str = "total_cases",
    *,
    threshold: float = 100,
    country_col: str = "country",
) -> pd.DataFrame:
    """Align cumulative curves at the first day a country crossed a threshold."""
    frames: list[pd.DataFrame] = []
    for country in countries:
        sub = df[df[country_col] == country].sort_values("date")
        if sub.empty or "total_cases" not in sub.columns:
            continue
        validated = sub.dropna(subset=["total_cases"])
        crossing = validated[validated["total_cases"] >= threshold]
        if crossing.empty:
            continue
        first_date = crossing["date"].iloc[0]
        aligned = sub[sub["date"] >= first_date].copy()
        aligned["day"] = (aligned["date"] - first_date).dt.days
        aligned[country_col] = country
        frames.append(aligned[[country_col, "date", "day", metric]].dropna(subset=[metric]))
    if not frames:
        return pd.DataFrame(columns=[country_col, "date", "day", metric])
    return pd.concat(frames, ignore_index=True)


def continent_summary(
    snapshot: pd.DataFrame,
    metric: str,
    *,
    continent_col: str = "continent",
) -> pd.DataFrame:
    """Population-aware regional aggregation of a country snapshot."""
    if snapshot.empty or metric not in snapshot.columns or continent_col not in snapshot.columns:
        return pd.DataFrame(columns=[continent_col, "value", "countries"])
    working = snapshot.dropna(subset=[continent_col]).copy()
    working[continent_col] = working[continent_col].astype(str)
    rows: list[dict[str, Any]] = []
    for continent, group in working.groupby(continent_col):
        value = float("nan")
        if metric in {"total_cases", "total_deaths"}:
            value = float(group[metric].sum(min_count=1))
        elif metric == "total_cases_per_million":
            value = safe_divide(group["total_cases"].sum(min_count=1), group["population"].sum(min_count=1), 1_000_000)
        elif metric == "total_deaths_per_million":
            value = safe_divide(group["total_deaths"].sum(min_count=1), group["population"].sum(min_count=1), 1_000_000)
        elif metric == "case_fatality_ratio":
            value = safe_divide(group["total_deaths"].sum(min_count=1), group["total_cases"].sum(min_count=1), 100)
        elif metric in {"vaccination_rate", "fully_vaccinated_rate", "booster_rate"}:
            valid = group.dropna(subset=[metric, "population"])
            value = safe_divide(
                (valid[metric] * valid["population"]).sum(),
                valid["population"].sum(),
            )
        else:
            value = float(group[metric].mean())
        rows.append({continent_col: continent, "value": value, "countries": int(len(group))})
    return pd.DataFrame(rows).sort_values("value", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# World / national totals
# --------------------------------------------------------------------------- #
def world_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "country" not in df.columns:
        return df.iloc[0:0]
    return df[df["country"].eq("World")].sort_values("date").copy()


def latest_world(df: pd.DataFrame) -> pd.Series | None:
    world = world_frame(df)
    if world.empty:
        return None
    world = world.sort_values("date")
    for column in ("total_cases", "total_deaths", "population", "vaccination_rate"):
        if column in world.columns:
            world[column] = world[column].ffill()
    valid = world.dropna(subset=["total_cases"])
    if valid.empty:
        return None
    return valid.iloc[-1]


def latest_national_india(india: pd.DataFrame) -> pd.Series | None:
    national = india[india["entity_type"].eq("National")].sort_values("date")
    if national.empty:
        return None
    valid = national.dropna(subset=["total_cases"])
    if valid.empty:
        return None
    return valid.iloc[-1]


def latest_state_snapshot(india: pd.DataFrame) -> pd.DataFrame:
    states = india[india["is_state"].astype(bool)]
    return latest_snapshot(states, entity_col="state", countries_only=False)


def global_totals(df: pd.DataFrame) -> dict[str, Any]:
    """Global KPIs from the World aggregate, with a sum-of-countries fallback."""
    latest = latest_world(df)
    snapshot = latest_snapshot(df)
    countries_reporting = int(snapshot["country"].nunique()) if not snapshot.empty else 0
    if latest is not None:
        cases = float(latest.get("total_cases", np.nan))
        deaths = float(latest.get("total_deaths", np.nan))
        return {
            "confirmed": cases,
            "deaths": deaths,
            "cfr": safe_divide(deaths, cases, 100),
            "date": latest.get("date"),
            "countries_reporting": countries_reporting,
            "vaccination_rate": float(latest.get("vaccination_rate", np.nan)),
            "fully_vaccinated_rate": float(latest.get("fully_vaccinated_rate", np.nan)),
        }
    if snapshot.empty:
        return {key: np.nan for key in ("confirmed", "deaths", "cfr")}
    cases = float(snapshot["total_cases"].sum(min_count=1))
    deaths = float(snapshot["total_deaths"].sum(min_count=1))
    return {
        "confirmed": cases,
        "deaths": deaths,
        "cfr": safe_divide(deaths, cases, 100),
        "date": snapshot["date"].max(),
        "countries_reporting": countries_reporting,
        "vaccination_rate": float("nan"),
        "fully_vaccinated_rate": float("nan"),
    }


def india_totals(india: pd.DataFrame) -> dict[str, Any]:
    row = latest_national_india(india)
    if row is None:
        return {key: np.nan for key in ("confirmed", "deaths", "recovered", "active", "cfr", "recovery_rate")}
    confirmed = float(row.get("total_cases", np.nan))
    deaths = float(row.get("total_deaths", np.nan))
    recovered = float(row.get("total_recovered", np.nan))
    return {
        "confirmed": confirmed,
        "deaths": deaths,
        "recovered": recovered,
        "active": calculate_active(confirmed, recovered, deaths),
        "cfr": safe_divide(deaths, confirmed, 100),
        "recovery_rate": safe_divide(recovered, confirmed, 100),
        "date": row.get("date"),
    }


# --------------------------------------------------------------------------- #
# Period comparisons and trends
# --------------------------------------------------------------------------- #
def trailing_window_sum(
    df: pd.DataFrame,
    value_col: str,
    end_date: Any,
    days: int = 7,
    *,
    date_col: str = "date",
) -> float:
    if df.empty or value_col not in df.columns:
        return float("nan")
    end = pd.Timestamp(end_date).normalize()
    start = end - pd.Timedelta(days=days - 1)
    dates = pd.to_datetime(df[date_col], errors="coerce")
    window = df.loc[(dates >= start) & (dates <= end), value_col]
    if window.empty:
        return float("nan")
    return float(window.sum(min_count=1))


def window_comparison(
    df: pd.DataFrame,
    value_col: str,
    end_date: Any,
    days: int = 7,
    *,
    date_col: str = "date",
) -> dict[str, Any]:
    """Compare two adjacent equal-length windows.

    Zeros are legitimate (no reported cases); a missing denominator is returned
    as ``None`` rather than infinity.
    """
    end = pd.Timestamp(end_date).normalize()
    current_start = end - pd.Timedelta(days=days - 1)
    previous_end = current_start - pd.Timedelta(days=1)
    previous_start = previous_end - pd.Timedelta(days=days - 1)
    dates = pd.to_datetime(df[date_col], errors="coerce")
    current = df.loc[(dates >= current_start) & (dates <= end), value_col].sum(min_count=1)
    previous = df.loc[(dates >= previous_start) & (dates <= previous_end), value_col].sum(min_count=1)
    current_f = float(current) if pd.notna(current) else float("nan")
    previous_f = float(previous) if pd.notna(previous) else float("nan")
    change = (
        (current_f - previous_f) / abs(previous_f) * 100.0
        if pd.notna(current_f) and pd.notna(previous_f) and previous_f != 0
        else None
    )
    return {
        "current": current_f,
        "previous": previous_f,
        "change_pct": change,
        "current_start": current_start,
        "current_end": end,
        "previous_start": previous_start,
        "previous_end": previous_end,
        "days": days,
    }


def trend_change(
    df: pd.DataFrame,
    value_col: str,
    *,
    window: int = 7,
    date_col: str = "date",
) -> dict[str, Any]:
    """Latest window mean vs the preceding window mean."""
    if df.empty or value_col not in df.columns:
        return {"direction": "unavailable", "change_pct": None}
    series = df[[date_col, value_col]].dropna().sort_values(date_col)
    if len(series) < 2:
        return {"direction": "insufficient data", "change_pct": None}
    values = series[value_col]
    recent = values.tail(window)
    previous = values.iloc[max(0, len(values) - 2 * window) : max(0, len(values) - window)]
    if recent.empty or previous.empty:
        return {"direction": "insufficient data", "change_pct": None}
    recent_mean = float(recent.mean())
    previous_mean = float(previous.mean())
    change = (
        (recent_mean - previous_mean) / abs(previous_mean) * 100.0
        if previous_mean != 0
        else None
    )
    if change is None:
        direction = "flat / insufficient baseline"
    elif change > 5:
        direction = "increasing"
    elif change < -5:
        direction = "decreasing"
    else:
        direction = "broadly stable"
    return {
        "recent_mean": recent_mean,
        "previous_mean": previous_mean,
        "change_pct": change,
        "direction": direction,
        "window": window,
    }


def compare_periods(
    df: pd.DataFrame,
    value_col: str,
    current: tuple[Any, Any],
    previous: tuple[Any, Any],
    *,
    date_col: str = "date",
) -> dict[str, Any]:
    def totals(period: tuple[Any, Any]) -> tuple[float, int]:
        start, end = period
        dates = pd.to_datetime(df[date_col], errors="coerce")
        window = df.loc[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end)), value_col]
        return float(window.sum(min_count=1)), int(window.notna().sum())

    current_total, current_days = totals(current)
    previous_total, previous_days = totals(previous)
    change = (
        (current_total - previous_total) / abs(previous_total) * 100.0
        if pd.notna(current_total) and pd.notna(previous_total) and previous_total != 0
        else None
    )
    return {
        "current_total": current_total,
        "current_days": current_days,
        "previous_total": previous_total,
        "previous_days": previous_days,
        "change_pct": change,
    }


# --------------------------------------------------------------------------- #
# Monthly aggregation and peak detection
# --------------------------------------------------------------------------- #
def monthly_summary(
    df: pd.DataFrame,
    value_cols: Sequence[str],
    *,
    entity_col: str | None = None,
    date_col: str = "date",
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    columns = [column for column in value_cols if column in df.columns]
    if not columns:
        return pd.DataFrame()
    working = df[[date_col, *([entity_col] if entity_col else []), *columns]].copy()
    working["month"] = pd.to_datetime(working[date_col]).dt.to_period("M").dt.to_timestamp()
    groups = ["month", *([entity_col] if entity_col else [])]
    result = working.groupby(groups, dropna=False)[columns].sum(min_count=1).reset_index()
    return result.sort_values(groups).reset_index(drop=True)


def monthly_heatmap_matrix(
    df: pd.DataFrame,
    value_col: str,
    *,
    date_col: str = "date",
) -> pd.DataFrame:
    monthly = monthly_summary(df, [value_col], date_col=date_col)
    if monthly.empty:
        return pd.DataFrame()
    monthly["year"] = monthly["month"].dt.year
    monthly["month_num"] = monthly["month"].dt.month
    return monthly.pivot_table(index="year", columns="month_num", values=value_col, aggfunc="sum")


def detect_peaks(
    series: pd.Series,
    *,
    window: int = 14,
    top_n: int = 5,
    min_ratio: float = 0.05,
) -> pd.DataFrame:
    """Find well-separated local maxima in a daily series.

    A peak must be the maximum inside a centred ±``window`` window and at least
    ``min_ratio`` of the series maximum.  Nearby candidates are suppressed so a
    single wave does not produce dozens of "peaks".
    """
    if series is None or len(series) == 0:
        return pd.DataFrame(columns=["date", "value"])
    # Rolling over a zero-dropped series would be positional, so a sparse
    # series would incorrectly compare non-adjacent dates. Keep the complete
    # daily index for the rolling maximum and only filter to positives later.
    clean = series.dropna().sort_index()
    if clean.empty or not (clean > 0).any():
        return pd.DataFrame(columns=["date", "value"])
    if isinstance(clean.index, pd.DatetimeIndex):
        full_index = pd.date_range(clean.index.min(), clean.index.max(), freq="D")
        clean = clean.reindex(full_index, fill_value=0.0)
    tolerance = 1e-9
    rolling_max = clean.rolling(window=window * 2 + 1, center=True, min_periods=1).max()
    candidates = clean[(clean > 0) & (clean >= rolling_max - tolerance)]
    candidates = candidates[candidates >= clean.max() * min_ratio]

    chosen: list[tuple[pd.Timestamp, float]] = []
    for stamp, value in candidates.sort_values(ascending=False).items():
        if all(abs((pd.Timestamp(stamp) - existing).days) > window for existing, _ in chosen):
            chosen.append((pd.Timestamp(stamp), float(value)))
        if len(chosen) >= top_n:
            break
    if not chosen:
        return pd.DataFrame(columns=["date", "value"])
    peaks = pd.DataFrame(chosen, columns=["date", "value"]).sort_values("date")
    return peaks.reset_index(drop=True)


def peak_of(series: pd.Series) -> tuple[pd.Timestamp | None, float]:
    if series is None or len(series) == 0:
        return None, float("nan")
    clean = series.dropna()
    if clean.empty:
        return None, float("nan")
    idx = clean.idxmax()
    return pd.Timestamp(idx), float(clean.loc[idx])


# --------------------------------------------------------------------------- #
# EDA helpers
# --------------------------------------------------------------------------- #
def describe_metrics(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    present = [column for column in columns if column in df.columns]
    if not present:
        return pd.DataFrame()
    numeric = df[present].apply(pd.to_numeric, errors="coerce")
    described = numeric.describe(percentiles=[0.25, 0.5, 0.75]).T
    described = described.rename(
        columns={
            "count": "Count",
            "mean": "Mean",
            "std": "Std. dev.",
            "min": "Minimum",
            "25%": "Q1",
            "50%": "Median",
            "75%": "Q3",
            "max": "Maximum",
        }
    )
    return described[["Count", "Mean", "Median", "Std. dev.", "Minimum", "Q1", "Q3", "Maximum"]]


def correlation_matrix(
    df: pd.DataFrame,
    columns: Sequence[str],
    *,
    method: str = "spearman",
    min_periods: int = 10,
) -> pd.DataFrame:
    present = [column for column in columns if column in df.columns]
    if len(present) < 2:
        return pd.DataFrame()
    numeric = df[present].apply(pd.to_numeric, errors="coerce")
    return numeric.corr(method=method, min_periods=min_periods)


def iqr_outliers(series: pd.Series) -> dict[str, Any]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return {"count": 0, "lower": np.nan, "upper": np.nan, "share_pct": 0.0}
    q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    flags = (clean < lower) | (clean > upper)
    return {
        "count": int(flags.sum()),
        "lower": float(lower),
        "upper": float(upper),
        "share_pct": round(float(flags.mean() * 100), 2),
    }


def coverage_summary(df: pd.DataFrame, entity_col: str = "country") -> dict[str, Any]:
    if df.empty:
        return {"rows": 0, "entities": 0, "date_min": None, "date_max": None}
    dates = pd.to_datetime(df["date"], errors="coerce")
    return {
        "rows": int(len(df)),
        "entities": int(df[entity_col].nunique()),
        "date_min": dates.min(),
        "date_max": dates.max(),
    }


def latest_metric_date(df: pd.DataFrame, metric: str, entity_col: str = "country") -> pd.Timestamp | None:
    if metric not in df.columns:
        return None
    valid = df.dropna(subset=[metric])
    if valid.empty:
        return None
    return pd.Timestamp(valid["date"].max())


def vaccination_coverage(snapshot: pd.DataFrame) -> dict[str, Any]:
    if "vaccination_rate" not in snapshot.columns:
        return {"weighted": np.nan, "countries": 0, "as_of": None}
    valid = snapshot.dropna(subset=["vaccination_rate", "population"])
    weighted = safe_divide(
        (valid["vaccination_rate"] * valid["population"]).sum(),
        valid["population"].sum(),
    )
    return {
        "weighted": weighted,
        "countries": int(len(valid)),
        "as_of": pd.to_datetime(valid["vaccination_as_of"], errors="coerce").max()
        if "vaccination_as_of" in valid.columns
        else None,
    }
