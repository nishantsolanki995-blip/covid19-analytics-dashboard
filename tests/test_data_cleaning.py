"""Tests for the pure data-cleaning transforms."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_cleaning import (
    DataSchemaError,
    add_rolling_average,
    build_india_quality_report,
    build_owid_quality_report,
    combine_quality_reports,
    compute_case_fatality,
    compute_recovery_ratio,
    derive_daily_from_cumulative,
    prepare_india,
    prepare_india_map_data,
    prepare_owid,
    standardize_state_name,
    validate_india_schema,
    validate_owid_schema,
)


def _owid_raw() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=8, freq="D")
    rows = []
    for country, continent, code, offset in [
        ("Alpha", "Asia", "ALP", 0),
        ("Beta", "Europe", "BET", 0),
        ("World", pd.NA, "OWID_WRL", 0),
    ]:
        for index, date in enumerate(dates):
            cases = index * 10.0 + offset
            rows.append(
                {
                    "country": country,
                    "code": code,
                    "continent": continent,
                    "date": date,
                    "total_cases": cases,
                    "new_cases": 10.0 if index else 0.0,
                    "total_deaths": cases * 0.01,
                    "new_deaths": 0.1 if index else 0.0,
                    "population": 1_000_000.0,
                }
            )
    return pd.DataFrame(rows)


def _india_raw() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": [
                "2020-01-30",
                "2020-01-31",
                "2020-02-01",
                "2020-01-30",
                "2020-01-31",
                "2020-02-01",
            ],
            "State": ["Kerala", "Kerala", "Kerala", "Delhi", "Delhi", "Delhi"],
            "Confirmed": [1, 2, 3, 10, 12, 9],
            "Recovered": [0, 0, 1, 5, 6, 4],
            "Deceased": [0, 0, 0, 1, 1, 1],
            "Other": [0, 0, 0, 0, 0, 0],
            "Tested": [10, 20, 30, 100, 120, 110],
        }
    )


def test_owid_schema_validation() -> None:
    assert validate_owid_schema(_owid_raw()) == []
    with pytest.raises(DataSchemaError):
        prepare_owid(_owid_raw().drop(columns=["total_deaths"]))


def test_india_schema_validation() -> None:
    assert validate_india_schema(_india_raw()) == []
    assert "Confirmed" in validate_india_schema(_india_raw().drop(columns=["Confirmed"]))


def test_prepare_owid_classifies_entities_and_derives_rates() -> None:
    processed, log = prepare_owid(_owid_raw())
    assert log["input_rows"] == len(_owid_raw())
    assert log["countries"] == 2
    assert set(processed["entity_type"]) == {"Country", "Aggregate"}
    alpha = processed[processed["country"] == "Alpha"].sort_values("date")
    assert alpha["is_country"].all()
    assert processed[processed["country"] == "World"]["is_country"].sum() == 0
    # Cases per million was computed from the population denominator.
    assert alpha["total_cases_per_million"].notna().all()
    assert alpha["total_cases_per_million"].iloc[-1] == pytest.approx(70 / 1_000_000 * 1_000_000)
    assert alpha["new_cases_7d"].notna().all()


def test_compute_ratios_handle_zero_and_missing() -> None:
    cases = pd.Series([100.0, 0.0, np.nan, 50.0])
    deaths = pd.Series([10.0, 1.0, 2.0, np.nan])
    cfr = compute_case_fatality(deaths, cases)
    assert cfr.iloc[0] == pytest.approx(10.0)
    assert pd.isna(cfr.iloc[1])
    assert pd.isna(cfr.iloc[2])
    assert pd.isna(cfr.iloc[3])

    recovered = pd.Series([90.0, 0.0, 1.0, 20.0])
    ratio = compute_recovery_ratio(recovered, cases)
    assert ratio.iloc[0] == pytest.approx(90.0)
    assert pd.isna(ratio.iloc[1])
    assert pd.isna(ratio.iloc[2])
    assert ratio.iloc[3] == pytest.approx(40.0)


def test_rolling_average_is_grouped_and_trailing() -> None:
    frame = pd.DataFrame(
        {
            "entity": ["A"] * 4 + ["B"] * 4,
            "value": [1, 2, 3, 4, 10, 20, 30, 40],
        }
    )
    out = add_rolling_average(frame, "value", group_col="entity", out_col="avg", window=2)
    assert out.loc[0, "avg"] == pytest.approx(1.0)
    assert out.loc[1, "avg"] == pytest.approx(1.5)
    assert out.loc[3, "avg"] == pytest.approx(3.5)
    assert out.loc[4, "avg"] == pytest.approx(10.0)
    assert out.loc[7, "avg"] == pytest.approx(35.0)


def test_daily_difference_keeps_negative_revisions() -> None:
    frame = pd.DataFrame(
        {
            "entity": ["A", "A", "A"],
            "date": pd.date_range("2020-01-01", periods=3),
            "cumulative": [10, 25, 20],
        }
    )
    out = derive_daily_from_cumulative(frame, "cumulative", group_col="entity", out_col="daily")
    assert out["daily"].tolist() == [10.0, 15.0, -5.0]


def test_standardize_state_names() -> None:
    assert standardize_state_name("  NCT of Delhi ") == "Delhi"
    assert standardize_state_name("Jammu & Kashmir") == "Jammu and Kashmir"
    assert standardize_state_name("Orissa") == "Odisha"
    assert standardize_state_name("Kerala") == "Kerala"


def test_prepare_india_builds_complete_grid_and_flags_revisions() -> None:
    processed, log = prepare_india(_india_raw())
    assert log["states"] == 2
    assert log["date_min"] == "2020-01-30"
    assert log["date_max"] == "2020-02-01"
    assert len(processed) == 6
    assert processed[["total_cases", "total_recovered", "total_deaths"]].notna().all().all()
    delhi = processed[processed["state"] == "Delhi"].sort_values("date")
    assert delhi["active_cases"].iloc[1] == pytest.approx(12 - 6 - 1)
    assert delhi["new_cases"].iloc[-1] == pytest.approx(-3.0)
    assert bool(delhi["revision_flag"].iloc[-1]) is True
    assert "tests" not in processed.columns


def test_quality_reports_have_auditable_keys() -> None:
    owid_processed, _ = prepare_owid(_owid_raw())
    india_processed, _ = prepare_india(_india_raw())
    combined = combine_quality_reports(
        build_owid_quality_report(_owid_raw(), owid_processed),
        build_india_quality_report(_india_raw(), india_processed),
    )
    assert set(combined) == {"generated_at", "owid", "india"}
    assert combined["owid"]["processed_rows"] == len(owid_processed)
    assert combined["owid"]["countries"] == 2
    assert combined["india"]["states"] == 2
    assert combined["india"]["negative_daily_rows"] >= 1
    assert isinstance(combined["owid"]["schema"], list)
    assert combined["owid"]["schema"][0]["column"]


def test_india_map_data_expands_combined_union_territory() -> None:
    snapshot = pd.DataFrame(
        [
            {
                "state": "Dadra and Nagar Haveli and Daman and Diu",
                "total_cases": 100.0,
                "total_deaths": 1.0,
            }
        ]
    )
    mapped = prepare_india_map_data(snapshot, "total_cases")
    assert set(mapped["geo_state"]) == {"DADRA & NAGAR HAVELI", "DAMAN & DIU"}
    assert mapped["total_cases"].tolist() == [100.0, 100.0]
