"""Tests for the analytical layer."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis import (
    align_by_days_since_threshold,
    calculate_active,
    compare_periods,
    continent_summary,
    correlation_matrix,
    country_series,
    coverage_summary,
    date_bounds,
    describe_metrics,
    detect_peaks,
    filter_by_date,
    filter_dataframe,
    global_totals,
    india_totals,
    iqr_outliers,
    latest_metric_date,
    latest_snapshot,
    latest_world,
    monthly_heatmap_matrix,
    monthly_summary,
    paginate,
    peak_of,
    rank_entities,
    safe_divide,
    trend_change,
    vaccination_coverage,
    window_comparison,
    world_frame,
)


def _owid_panel() -> pd.DataFrame:
    dates = pd.date_range("2021-01-01", periods=10, freq="D")
    rows = []
    for country, continent, base, pop, rate in [
        ("Alpha", "Asia", 100.0, 1_000_000.0, 50.0),
        ("Beta", "Europe", 40.0, 3_000_000.0, 100.0),
        ("World", pd.NA, 500.0, 4_000_000.0, np.nan),
    ]:
        for index, date in enumerate(dates):
            cases = base * (index + 1)
            deaths = cases * 0.02
            rows.append(
                {
                    "country": country,
                    "continent": continent,
                    "is_country": country != "World",
                    "date": date,
                    "total_cases": cases,
                    "new_cases": base,
                    "new_cases_7d": base,
                    "total_deaths": deaths,
                    "new_deaths": base * 0.02,
                    "new_deaths_7d": base * 0.02,
                    "total_cases_per_million": cases / pop * 1_000_000,
                    "total_deaths_per_million": deaths / pop * 1_000_000,
                    "case_fatality_ratio": 2.0,
                    "population": pop,
                    "vaccination_rate": rate if country != "World" else np.nan,
                }
            )
    frame = pd.DataFrame(rows)
    # A missing cumulative observation on the last day must be forward-filled.
    frame.loc[
        (frame["country"] == "Alpha") & (frame["date"] == dates[-1]), "total_cases"
    ] = np.nan
    return frame


def _india_panel() -> pd.DataFrame:
    dates = pd.date_range("2021-05-01", periods=3, freq="D")
    rows = []
    for index, date in enumerate(dates):
        cases = 1000.0 + index * 100
        recovered = 600.0 + index * 80
        deaths = 20.0 + index * 2
        rows.append(
            {
                "date": date,
                "state": "India",
                "entity_type": "National",
                "is_state": False,
                "total_cases": cases,
                "total_recovered": recovered,
                "total_deaths": deaths,
                "new_cases": 100.0,
                "new_deaths": 2.0,
                "active_cases": cases - recovered - deaths,
                "case_fatality_ratio": deaths / cases * 100,
                "recovery_ratio": recovered / cases * 100,
            }
        )
    return pd.DataFrame(rows)


def test_date_helpers_and_filters() -> None:
    frame = _owid_panel()
    start, end = date_bounds(frame)
    assert start == pd.Timestamp("2021-01-01")
    assert end == pd.Timestamp("2021-01-10")
    sliced = filter_by_date(frame, pd.Timestamp("2021-01-03"), pd.Timestamp("2021-01-05"))
    assert sliced["date"].min() == pd.Timestamp("2021-01-03")
    assert sliced["date"].max() == pd.Timestamp("2021-01-05")
    filtered = filter_dataframe(
        frame,
        entities=["Alpha"],
        start=pd.Timestamp("2021-01-02"),
        end=pd.Timestamp("2021-01-04"),
        search="alp",
    )
    assert set(filtered["country"]) == {"Alpha"}
    assert len(filtered) == 3


def test_pagination_metadata() -> None:
    frame = pd.DataFrame({"value": range(120)})
    page, meta = paginate(frame, page=2, page_size=50)
    assert meta == {"page": 2, "page_size": 50, "total_pages": 3, "total_rows": 120}
    assert page["value"].tolist()[0] == 50
    assert len(page) == 50


def test_snapshot_excludes_aggregates_and_forward_fills() -> None:
    snapshot = latest_snapshot(_owid_panel())
    assert set(snapshot["country"]) == {"Alpha", "Beta"}
    alpha = snapshot[snapshot["country"] == "Alpha"].iloc[0]
    assert alpha["date"] == pd.Timestamp("2021-01-10")
    # The final cumulative value is missing, so the previous 900 is carried forward.
    assert alpha["total_cases"] == pytest.approx(900.0)
    assert alpha["vaccination_rate"] == pytest.approx(50.0)


def test_rank_entities_respects_missing_and_order() -> None:
    snapshot = latest_snapshot(_owid_panel())
    ranked = rank_entities(snapshot, "total_cases", n=1)
    assert ranked["country"].iloc[0] == "Alpha"
    ranked_asc = rank_entities(snapshot, "total_cases", n=2, ascending=True)
    assert ranked_asc["country"].tolist() == ["Beta", "Alpha"]
    empty = rank_entities(snapshot, "not_a_metric", n=2)
    assert empty.empty


def test_world_and_global_totals() -> None:
    frame = _owid_panel()
    world = world_frame(frame)
    assert len(world) == 10
    latest = latest_world(frame)
    assert latest is not None
    assert latest["total_cases"] == pytest.approx(5000.0)
    totals = global_totals(frame)
    assert totals["confirmed"] == pytest.approx(5000.0)
    assert totals["deaths"] == pytest.approx(100.0)
    assert totals["cfr"] == pytest.approx(2.0)
    assert totals["countries_reporting"] == 2


def test_india_totals_and_active_cases() -> None:
    totals = india_totals(_india_panel())
    assert totals["confirmed"] == pytest.approx(1200.0)
    assert totals["recovered"] == pytest.approx(760.0)
    assert totals["deaths"] == pytest.approx(24.0)
    assert totals["active"] == pytest.approx(416.0)
    assert total_active_is_valid(totals)
    assert calculate_active(10, 5, 2) == 3
    assert np.isnan(calculate_active(10, np.nan, 2))
    assert calculate_active(10, 20, 2) == 0


def total_active_is_valid(totals: dict) -> bool:
    return totals["active"] == pytest.approx(totals["confirmed"] - totals["recovered"] - totals["deaths"])


def test_window_and_period_comparisons() -> None:
    dates = pd.date_range("2021-01-01", periods=14, freq="D")
    frame = pd.DataFrame({"date": dates, "new_cases": np.arange(1, 15, dtype=float)})
    comparison = window_comparison(frame, "new_cases", dates[-1], days=7)
    assert comparison["current"] == pytest.approx(sum(range(8, 15)))
    assert comparison["previous"] == pytest.approx(sum(range(1, 8)))
    assert comparison["change_pct"] == pytest.approx(175.0)
    missing = window_comparison(frame.iloc[0:0], "new_cases", dates[-1], days=7)
    assert np.isnan(missing["current"])
    assert missing["change_pct"] is None
    periods = compare_periods(
        frame,
        "new_cases",
        (dates[7], dates[13]),
        (dates[0], dates[6]),
    )
    assert periods["current_total"] == pytest.approx(77.0)
    assert periods["previous_total"] == pytest.approx(28.0)
    assert periods["change_pct"] == pytest.approx((77 - 28) / 28 * 100)


def test_trend_change_direction() -> None:
    dates = pd.date_range("2021-01-01", periods=60, freq="D")
    values = np.r_[np.full(30, 100.0), np.full(30, 50.0)]
    frame = pd.DataFrame({"date": dates, "value": values})
    trend = trend_change(frame, "value", window=30)
    assert trend["direction"] == "decreasing"
    assert trend["change_pct"] == pytest.approx(-50.0)
    flat = trend_change(frame.assign(value=1.0), "value", window=30)
    assert flat["direction"] == "broadly stable"


def test_monthly_aggregation_and_heatmap() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2021-01-05", "2021-01-20", "2021-02-01", "2021-02-02"]
            ),
            "new_cases": [1.0, 2.0, 3.0, 4.0],
        }
    )
    monthly = monthly_summary(frame, ["new_cases"])
    assert monthly["new_cases"].tolist() == [3.0, 7.0]
    matrix = monthly_heatmap_matrix(frame, "new_cases")
    assert matrix.loc[2021, 1] == pytest.approx(3.0)
    assert matrix.loc[2021, 2] == pytest.approx(7.0)


def test_peak_detection_and_global_peak() -> None:
    dates = pd.date_range("2021-01-01", periods=60, freq="D")
    values = np.zeros(60)
    values[20] = 100
    values[45] = 50
    series = pd.Series(values, index=dates)
    peaks = detect_peaks(series, window=7, top_n=2)
    assert len(peaks) == 2
    assert peaks["value"].tolist() == [100.0, 50.0]
    peak_date, peak_value = peak_of(series)
    assert peak_date == dates[20]
    assert peak_value == pytest.approx(100.0)
    empty_date, empty_value = peak_of(pd.Series(dtype=float))
    assert empty_date is None
    assert np.isnan(empty_value)


def test_descriptive_and_correlation_helpers() -> None:
    frame = pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "y": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        }
    )
    description = describe_metrics(frame, ["x", "y"])
    assert description.loc["x", "Count"] == 10
    assert description.loc["x", "Median"] == 5.5
    corr = correlation_matrix(frame, ["x", "y"])
    assert corr.loc["x", "y"] == pytest.approx(1.0)


def test_outliers_coverage_and_metric_dates() -> None:
    series = pd.Series([1, 1, 1, 1, 1, 1, 1, 1, 1, 100])
    outliers = iqr_outliers(series)
    assert outliers["count"] == 1
    assert outliers["share_pct"] == pytest.approx(10.0)
    frame = _owid_panel()
    summary = coverage_summary(frame, "country")
    assert summary["rows"] == 30
    assert summary["entities"] == 3
    assert latest_metric_date(frame, "total_cases") == pd.Timestamp("2021-01-10")
    assert latest_metric_date(frame, "not_a_metric") is None


def test_continent_summary_is_population_aware() -> None:
    frame = _owid_panel()
    snapshot = latest_snapshot(frame)
    continents = continent_summary(snapshot, "total_cases_per_million")
    assert set(continents["continent"]) == {"Asia", "Europe"}
    asia = continents[continents["continent"] == "Asia"].iloc[0]
    assert asia["value"] == pytest.approx(900.0)
    vaccination = continent_summary(snapshot, "vaccination_rate")
    assert set(vaccination["continent"]) == {"Asia", "Europe"}


def test_vaccination_weighted_coverage() -> None:
    snapshot = latest_snapshot(_owid_panel())
    coverage = vaccination_coverage(snapshot)
    expected = (50.0 * 1_000_000 + 100.0 * 3_000_000) / 4_000_000
    assert coverage["weighted"] == pytest.approx(expected)
    assert coverage["countries"] == 2
    assert coverage["as_of"] == pd.Timestamp("2021-01-10")


def test_alignment_and_country_series() -> None:
    frame = _owid_panel()
    aligned = align_by_days_since_threshold(frame, ["Alpha", "Beta"], threshold=200)
    assert set(aligned["country"]) == {"Alpha", "Beta"}
    assert (aligned.groupby("country")["day"].min() == 0).all()
    alpha = aligned[aligned["country"] == "Alpha"]
    assert alpha["total_cases"].iloc[0] >= 200
    series = country_series(frame, ["Alpha"], "total_cases")
    assert set(series.columns) == {"country", "date", "total_cases"}
    assert len(series) == 9
    assert country_series(frame, ["Alpha"], "missing").empty


def test_safe_divide_handles_invalid_denominators() -> None:
    assert safe_divide(10, 2) == 5
    assert np.isnan(safe_divide(10, 0))
    assert np.isnan(safe_divide(None, 2))
