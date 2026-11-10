"""Tests for the data-loader builders, status contract and atomic downloads."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src import data_loader
from src.data_loader import (
    DataUnavailableError,
    SOURCE_INFO,
    _human_size,
    build_india_processed,
    build_owid_processed,
    build_quality_report,
    dataset_status,
    download_file,
)


def _owid_raw() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    rows = []
    for country, continent, code in [
        ("Alpha", "Asia", "ALP"),
        ("Beta", "Europe", "BET"),
        ("World", pd.NA, "OWID_WRL"),
    ]:
        for index, date in enumerate(dates):
            cases = 10.0 * (index + 1)
            rows.append(
                {
                    "country": country,
                    "code": code,
                    "continent": continent,
                    "date": date,
                    "total_cases": cases,
                    "new_cases": 10.0,
                    "total_deaths": cases * 0.01,
                    "new_deaths": 0.1,
                    "population": 1_000_000.0,
                }
            )
    return pd.DataFrame(rows)


def _india_raw() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": ["2020-01-30", "2020-01-31", "2020-01-30", "2020-01-31"],
            "State": ["Kerala", "Kerala", "Delhi", "Delhi"],
            "Confirmed": [1, 3, 10, 12],
            "Recovered": [0, 1, 5, 6],
            "Deceased": [0, 0, 1, 1],
            "Other": [0, 0, 0, 0],
            "Tested": [10, 30, 100, 120],
        }
    )


def test_human_size_boundaries() -> None:
    assert _human_size(0) == "0 B"
    assert _human_size(999) == "999 B"
    assert _human_size(1_000) == "1.0 KB"
    assert _human_size(1_500_000) == "1.5 MB"
    assert _human_size(2_000_000_000) == "2.0 GB"


def test_source_info_documents_real_endpoints() -> None:
    assert SOURCE_INFO["owid"]["url"].startswith("https://")
    assert SOURCE_INFO["india"]["url"].startswith("https://")
    assert SOURCE_INFO["owid"]["licence"] == "CC BY 4.0"
    assert "1 June 2021" in SOURCE_INFO["india"]["description"]


def test_build_owid_processed_does_not_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(data_loader, "PROCESSED_DIR", tmp_path)
    processed, log = build_owid_processed(_owid_raw(), write=False)
    assert log["input_rows"] == 15
    assert log["countries"] == 2
    assert len(processed) == 15
    assert "new_cases_7d" in processed.columns
    assert list(tmp_path.iterdir()) == []


def test_build_india_processed_does_not_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(data_loader, "PROCESSED_DIR", tmp_path)
    processed, log = build_india_processed(_india_raw(), write=False)
    assert log["states"] == 2
    assert len(processed) == 4
    assert "active_cases" in processed.columns
    assert list(tmp_path.iterdir()) == []


def test_build_quality_report_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(data_loader, "PROCESSED_DIR", tmp_path)
    owid_processed, _ = build_owid_processed(_owid_raw(), write=False)
    india_processed, _ = build_india_processed(_india_raw(), write=False)
    report = build_quality_report(
        owid_raw=_owid_raw(),
        owid_processed=owid_processed,
        india_raw=_india_raw(),
        india_processed=india_processed,
        write=False,
    )
    assert set(report) == {"generated_at", "owid", "india"}
    assert report["owid"]["countries"] == 2
    assert report["india"]["states"] == 2
    assert "schema" in report["owid"]
    assert "schema" in report["india"]


def test_dataset_status_exposes_frontend_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = {
        "owid": {
            "date_min": "2020-01-01",
            "date_max": "2024-01-01",
            "processed_rows": 123,
            "countries": 10,
        },
        "india": {
            "date_min": "2020-01-30",
            "date_max": "2021-06-01",
            "states": 36,
        },
    }
    monkeypatch.setattr(data_loader, "load_quality_report", lambda: fake)
    status = dataset_status()
    assert status["owid_date_min"] == "2020-01-01"
    assert status["owid_date_max"] == "2024-01-01"
    assert status["owid_rows"] == 123
    assert status["owid_countries"] == 10
    assert status["india_date_min"] == "2020-01-30"
    assert status["india_date_max"] == "2021-06-01"
    assert status["india_states"] == 36
    assert status["quality"] is fake
    assert status["owid_source"] is SOURCE_INFO["owid"]


class _FakeResponse:
    def __init__(self, payload: bytes, content_length: str | None = None) -> None:
        self.payload = payload
        self.headers = {"Content-Length": content_length} if content_length is not None else {}
        self.status_code = 200

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int = 1024) -> list[bytes]:
        return [self.payload]


def test_download_file_is_atomic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = b"a,b\n1,2\n"
    monkeypatch.setattr(
        data_loader.requests,
        "get",
        lambda *args, **kwargs: _FakeResponse(payload, str(len(payload))),
    )
    target = tmp_path / "nested" / "file.csv"
    returned = download_file("https://example.test/file.csv", target, min_bytes=1)
    assert returned == target
    assert target.read_bytes() == payload
    assert not target.with_suffix(target.suffix + ".part").exists()


def test_download_file_rejects_incomplete_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        data_loader.requests,
        "get",
        lambda *args, **kwargs: _FakeResponse(b"short", "999"),
    )
    target = tmp_path / "file.csv"
    with pytest.raises(DataUnavailableError):
        download_file("https://example.test/file.csv", target, min_bytes=1)
    assert not target.exists()
    assert not target.with_suffix(target.suffix + ".part").exists()
