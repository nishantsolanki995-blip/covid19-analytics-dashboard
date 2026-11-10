"""Rule-based insights engine.

Every sentence is derived from the filtered data passed in by a page.  There is
no language model, no hard-coded statistic and no causal claim.  Each card also
records the metric and date basis so a reviewer can trace the number back to the
dataset.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.analysis import (
    country_series,
    monthly_summary,
    peak_of,
    safe_divide,
    trend_change,
    vaccination_coverage,
    world_frame,
)
from src.utils import (
    esc,
    format_compact,
    format_date,
    format_int,
    format_pct,
    format_signed_pct,
)


@dataclass(frozen=True)
class Insight:
    kicker: str
    title: str
    body: str
    source: str
    tone: str = "blue"

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


def _clean(insights: Iterable[Insight | None]) -> list[dict[str, str]]:
    return [item.as_dict() for item in insights if item is not None]


# --------------------------------------------------------------------------- #
# Global
# --------------------------------------------------------------------------- #
def global_insights(
    df: pd.DataFrame,
    snapshot: pd.DataFrame,
    totals: Mapping[str, Any],
    *,
    start: Any | None = None,
    end: Any | None = None,
) -> list[dict[str, str]]:
    insights: list[Insight | None] = []
    world_total = totals.get("confirmed")
    world_deaths = totals.get("deaths")
    as_of = format_date(totals.get("date"))

    valid = snapshot.dropna(subset=["total_cases"]) if "total_cases" in snapshot else snapshot
    if not valid.empty:
        top = valid.nlargest(1, "total_cases").iloc[0]
        share = safe_divide(top["total_cases"], world_total, 100)
        share_text = f"{format_pct(share)} of the global total" if pd.notna(share) else "global total unavailable"
        insights.append(
            Insight(
                kicker="CASE CONCENTRATION",
                title=f"{top['country']} reports the most confirmed cases",
                body=(
                    f"<strong>{esc(top['country'])}</strong> has <strong>{format_int(top['total_cases'])}</strong> "
                    f"cumulative reported cases ({esc(share_text)}). The World aggregate stands at "
                    f"{format_int(world_total)} cases."
                ),
                source=f"Metric: total_cases · latest country snapshot · World as of {as_of}",
                tone="blue",
            )
        )

    valid_deaths = snapshot.dropna(subset=["total_deaths"]) if "total_deaths" in snapshot else snapshot.iloc[0:0]
    if not valid_deaths.empty:
        top_deaths = valid_deaths.nlargest(1, "total_deaths").iloc[0]
        share = safe_divide(top_deaths["total_deaths"], world_deaths, 100)
        insights.append(
            Insight(
                kicker="MORTALITY",
                title=f"{top_deaths['country']} reports the highest death count",
                body=(
                    f"<strong>{esc(top_deaths['country'])}</strong> accounts for "
                    f"<strong>{format_int(top_deaths['total_deaths'])}</strong> reported deaths"
                    + (f" ({format_pct(share)} of the global total)" if pd.notna(share) else "")
                    + ". Deaths are reported counts and depend on national reporting practice."
                ),
                source=f"Metric: total_deaths · latest country snapshot · World as of {as_of}",
                tone="red",
            )
        )

    per_capita = snapshot.dropna(subset=["total_cases_per_million", "population"])
    per_capita = per_capita[per_capita["population"] >= 1_000_000]
    if not per_capita.empty:
        top_rate = per_capita.nlargest(1, "total_cases_per_million").iloc[0]
        insights.append(
            Insight(
                kicker="PER-CAPITA BURDEN",
                title=f"{top_rate['country']} has the highest reported case rate",
                body=(
                    f"Among countries with at least one million people, "
                    f"<strong>{esc(top_rate['country'])}</strong> reports "
                    f"<strong>{format_compact(top_rate['total_cases_per_million'])}</strong> cases per million. "
                    "Population-adjusted rates are more comparable than raw totals, but testing "
                    "and reporting differences remain."
                ),
                source="Metric: total_cases_per_million · countries with population ≥ 1M",
                tone="cyan",
            )
        )

    world = world_frame(df)
    if start is not None and end is not None and not world.empty:
        dates = pd.to_datetime(world["date"])
        world = world[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))]
    peak_date, peak_value = peak_of(
        world.set_index("date")["new_cases"] if not world.empty else pd.Series(dtype=float)
    )
    if peak_date is not None and pd.notna(peak_value):
        insights.append(
            Insight(
                kicker="PEAK ACTIVITY",
                title=f"Highest global daily increase: {format_compact(peak_value)} cases",
                body=(
                    f"The largest single-day increase inside the selected window was "
                    f"<strong>{format_int(peak_value)}</strong> on <strong>{format_date(peak_date)}</strong>. "
                    "Daily totals can include batches, corrections and uneven weekend reporting; "
                    "use the 7-day average for trend interpretation."
                ),
                source=f"Metric: new_cases · World · {format_date(start)} → {format_date(end)}",
                tone="amber",
            )
        )

    trend = trend_change(world, "new_cases_7d", window=28) if not world.empty else {}
    if trend.get("change_pct") is not None:
        direction = trend["direction"]
        insights.append(
            Insight(
                kicker="RECENT DIRECTION",
                title=f"Global reported activity is {direction}",
                body=(
                    f"The mean 7-day case average in the final 28 rows of the period is "
                    f"<strong>{format_compact(trend['recent_mean'])}</strong> per day versus "
                    f"<strong>{format_compact(trend['previous_mean'])}</strong> in the preceding 28 rows "
                    f"(<strong>{format_signed_pct(trend['change_pct'])}</strong>). This describes "
                    "the reported series only, not a causal claim."
                ),
                source="Metric: new_cases_7d · last 28 observations vs previous 28",
                tone="green" if trend["change_pct"] < 0 else "amber",
            )
        )

    vax = vaccination_coverage(snapshot)
    if pd.notna(vax["weighted"]) and vax["countries"]:
        insights.append(
            Insight(
                kicker="VACCINATION",
                title=f"{format_pct(vax['weighted'])} population-weighted first-dose coverage",
                body=(
                    f"Across <strong>{vax['countries']}</strong> countries with vaccination data, "
                    "the population-weighted share with at least one dose is "
                    f"<strong>{format_pct(vax['weighted'])}</strong>. Coverage is not directly "
                    "comparable across countries with different reporting dates."
                ),
                source="Metric: people_vaccinated_per_hundred, weighted by population",
                tone="violet",
            )
        )

    return _clean(insights)


# --------------------------------------------------------------------------- #
# India
# --------------------------------------------------------------------------- #
def india_insights(
    national: Mapping[str, Any],
    states: pd.DataFrame,
    *,
    national_series: pd.DataFrame | None = None,
) -> list[dict[str, str]]:
    insights: list[Insight | None] = []
    if not states.empty and "total_cases" in states:
        top = states.nlargest(1, "total_cases").iloc[0]
        share = safe_divide(top["total_cases"], national.get("confirmed"), 100)
        insights.append(
            Insight(
                kicker="STATE CONCENTRATION",
                title=f"{top['state']} accounts for the largest state caseload",
                body=(
                    f"<strong>{esc(top['state'])}</strong> reports <strong>{format_int(top['total_cases'])}</strong> "
                    f"confirmed cases"
                    + (f", or {format_pct(share)} of the national total" if pd.notna(share) else "")
                    + f". The national archive records {format_int(national.get('confirmed'))} cases."
                ),
                source=f"Metric: total_cases · latest state snapshot · archive through {format_date(national.get('date'))}",
                tone="blue",
            )
        )

    deaths = states.dropna(subset=["total_deaths"]) if "total_deaths" in states else states.iloc[0:0]
    if not deaths.empty:
        top_death = deaths.nlargest(1, "total_deaths").iloc[0]
        insights.append(
            Insight(
                kicker="MORTALITY",
                title=f"{top_death['state']} reports the most deaths",
                body=(
                    f"<strong>{esc(top_death['state'])}</strong> reports "
                    f"<strong>{format_int(top_death['total_deaths'])}</strong> deaths, with a reported "
                    f"case fatality ratio of <strong>{format_pct(top_death.get('case_fatality_ratio'))}</strong>. "
                    "Fatality ratios reflect reported cases and deaths, not the true infection fatality rate."
                ),
                source="Metric: total_deaths, case_fatality_ratio · latest state snapshot",
                tone="red",
            )
        )

    if not states.empty and "recovery_ratio" in states:
        valid = states.dropna(subset=["recovery_ratio"])
        if not valid.empty:
            avg_recovery = safe_divide(national.get("recovered"), national.get("confirmed"), 100)
            insights.append(
                Insight(
                    kicker="RECOVERY REPORTING",
                    title=f"National recovery ratio: {format_pct(avg_recovery)}",
                    body=(
                        f"<strong>{format_int(national.get('recovered'))}</strong> recoveries are reported against "
                        f"<strong>{format_int(national.get('confirmed'))}</strong> confirmed cases. "
                        "State recovery counts are self-reported and can lag behind case updates, so active "
                        "cases are an estimate derived from the three cumulative series."
                    ),
                    source="Metrics: total_recovered, total_cases, total_deaths · national row",
                    tone="green",
                )
            )

    if national_series is not None and not national_series.empty:
        peak_date, peak_value = peak_of(national_series.set_index("date")["new_cases"])
        if peak_date is not None and pd.notna(peak_value):
            insights.append(
                Insight(
                    kicker="PEAK WAVE",
                    title=f"Highest recorded daily increase: {format_int(peak_value)} cases",
                    body=(
                        f"The national archive's largest single-day increase is "
                        f"<strong>{format_int(peak_value)}</strong> on <strong>{format_date(peak_date)}</strong>. "
                        "The wave structure is clearer in the 7-day average than in raw daily bars."
                    ),
                    source=f"Metric: new_cases · national series · {format_date(national_series['date'].min())} → {format_date(national_series['date'].max())}",
                    tone="amber",
                )
            )

    if not states.empty and "case_fatality_ratio" in states:
        eligible = states[(states["total_cases"] >= 10_000)].dropna(subset=["case_fatality_ratio"])
        if not eligible.empty:
            highest = eligible.nlargest(1, "case_fatality_ratio").iloc[0]
            insights.append(
                Insight(
                    kicker="REPORTED CFR",
                    title=f"{highest['state']} has the highest reported CFR",
                    body=(
                        f"Among states/UTs with at least 10,000 confirmed cases, "
                        f"<strong>{esc(highest['state'])}</strong> reports a case fatality ratio of "
                        f"<strong>{format_pct(highest['case_fatality_ratio'])}</strong>. Differences can reflect "
                        "age structure, healthcare capacity and how deaths are attributed."
                    ),
                    source="Metric: case_fatality_ratio · states/UTs with ≥ 10,000 cases",
                    tone="violet",
                )
            )

    insights.append(
        Insight(
            kicker="DATA COVERAGE",
            title="India archive ends on 1 June 2021",
            body=(
                "This page uses the covid19india.org public state archive. It is a historical "
                "snapshot, not live data, and the latest date available in the file is "
                "<strong>1 June 2021</strong>. State populations are not included in the source, "
                "so per-million state comparisons are deliberately not shown."
            ),
            source="Source note · covid19india.org state-wise CSV archive",
            tone="cyan",
        )
    )
    return _clean(insights)


# --------------------------------------------------------------------------- #
# Comparative
# --------------------------------------------------------------------------- #
def comparison_insights(
    df: pd.DataFrame,
    countries: Sequence[str],
    metric: str,
    label: str,
    *,
    start: Any | None = None,
    end: Any | None = None,
) -> list[dict[str, str]]:
    insights: list[Insight | None] = []
    series = country_series(df, countries, metric)
    if start is not None and end is not None:
        dates = pd.to_datetime(series["date"])
        series = series[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))]
    if series.empty:
        return []

    latest_rows = series.sort_values("date").groupby("country", as_index=False).tail(1)
    leader = latest_rows.nlargest(1, metric).iloc[0]
    trailer = latest_rows.nsmallest(1, metric).iloc[0]
    insights.append(
        Insight(
            kicker="SELECTED SCOPE",
            title=f"{leader['country']} leads on {label.lower()}",
            body=(
                f"At the end of the selected period, <strong>{esc(leader['country'])}</strong> records "
                f"<strong>{format_compact(leader[metric])}</strong> against "
                f"<strong>{format_compact(trailer[metric])}</strong> for {esc(trailer['country'])}. "
                "The comparison uses the same metric and date window for every selected country."
            ),
            source=f"Metric: {metric} · latest value in selected window",
            tone="blue",
        )
    )

    full = df[df["country"].isin(list(countries))].copy() if "country" in df.columns else df.iloc[0:0]
    if start is not None and end is not None and not full.empty:
        full_dates = pd.to_datetime(full["date"])
        full = full[(full_dates >= pd.Timestamp(start)) & (full_dates <= pd.Timestamp(end))]
    if "new_cases_7d" in full.columns and metric in {"total_cases", "total_cases_per_million", "new_cases_7d", "total_deaths"}:
        recent = full.dropna(subset=["new_cases_7d"]).sort_values("date").groupby("country").tail(28)
        if not recent.empty:
            averages = recent.groupby("country")["new_cases_7d"].mean().dropna()
            if not averages.empty:
                top = averages.idxmax()
                insights.append(
                    Insight(
                        kicker="RECENT PACE",
                        title=f"{top} has the fastest recent reported pace",
                        body=(
                            f"Over the final 28 available observations, <strong>{esc(top)}</strong> averages "
                            f"<strong>{format_compact(averages.loc[top])}</strong> reported cases per day "
                            "(7-day average basis). This is a reporting-rate comparison, not a measure "
                            "of true transmission intensity."
                        ),
                        source="Metric: new_cases_7d · mean of last 28 observations per country",
                        tone="amber",
                    )
                )

    if metric == "case_fatality_ratio":
        valid = latest_rows.dropna(subset=[metric])
        if not valid.empty:
            highest = valid.nlargest(1, metric).iloc[0]
            insights.append(
                Insight(
                    kicker="SAFE INTERPRETATION",
                    title=f"Highest reported CFR: {highest['country']}",
                    body=(
                        f"<strong>{esc(highest['country'])}</strong> reports a case fatality ratio of "
                        f"<strong>{format_pct(highest[metric])}</strong>. CFR uses reported deaths divided by "
                        "reported confirmed cases; it is not the infection fatality ratio and is sensitive "
                        "to testing coverage."
                    ),
                    source="Metric: case_fatality_ratio · latest country snapshot",
                    tone="red",
                )
            )

    if len(countries) >= 2:
        span_days = (
            (pd.Timestamp(end) - pd.Timestamp(start)).days + 1
            if start is not None and end is not None
            else int(series["date"].nunique())
        )
        insights.append(
            Insight(
                kicker="COMPARABILITY",
                title=f"{len(countries)} countries compared over {span_days} days",
                body=(
                    "All series share the selected date window and metric. Cumulative totals are "
                    "sensitive to population size; use cases/deaths per million for cross-country "
                    "burden comparisons. Differences in testing and death registration remain."
                ),
                source="Method note · comparative analytics",
                tone="cyan",
            )
        )
    return _clean(insights)


# --------------------------------------------------------------------------- #
# Time series
# --------------------------------------------------------------------------- #
def timeseries_insights(
    data: pd.DataFrame,
    *,
    country: str,
    daily_col: str,
    rolling_col: str,
    daily_label: str,
    metric_col: str,
    metric_label: str,
    start: Any,
    end: Any,
) -> list[dict[str, str]]:
    insights: list[Insight | None] = []
    if data.empty:
        return []
    peak_date, peak_value = peak_of(data.set_index("date")[daily_col])
    if peak_date is not None and pd.notna(peak_value):
        insights.append(
            Insight(
                kicker="PEAK DAY",
                title=f"{country}: peak {daily_label.lower()} on {format_date(peak_date)}",
                body=(
                    f"The highest single-day value in the selected window is "
                    f"<strong>{format_int(peak_value)}</strong>. Raw daily counts can be distorted by "
                    f"batch releases; the accompanying 7-day average smooths this volatility."
                ),
                source=f"Metric: {daily_col} · {format_date(start)} → {format_date(end)}",
                tone="amber",
            )
        )

    monthly = monthly_summary(data, [daily_col])
    if not monthly.empty:
        row = monthly.loc[monthly[daily_col].idxmax()]
        total = monthly[daily_col].sum(min_count=1)
        share = safe_divide(row[daily_col], total, 100)
        insights.append(
            Insight(
                kicker="HIGHEST MONTH",
                title=f"{format_date(row['month'], '%B %Y')} carried the largest reported load",
                body=(
                    f"The month contributed <strong>{format_int(row[daily_col])}</strong> new reports"
                    + (f" ({format_pct(share)} of the selected period)" if pd.notna(share) else "")
                    + ". Monthly aggregation reduces day-to-day reporting noise but can still reflect "
                    "backlogs and retrospective revisions."
                ),
                source=f"Metric: {daily_col} · summed by calendar month",
                tone="blue",
            )
        )

    trend = trend_change(data, rolling_col, window=28)
    if trend.get("change_pct") is not None:
        insights.append(
            Insight(
                kicker="TREND",
                title=f"Recent reported activity is {trend['direction']}",
                body=(
                    f"The final 28 observations average <strong>{format_compact(trend['recent_mean'])}</strong> "
                    f"per day on the 7-day-average basis, versus <strong>{format_compact(trend['previous_mean'])}</strong> "
                    f"in the preceding block (<strong>{format_signed_pct(trend['change_pct'])}</strong>). "
                    "This is a description of the reported series, not evidence of cause and effect."
                ),
                source=f"Metric: {rolling_col} · last 28 observations vs previous 28",
                tone="green" if trend["change_pct"] < 0 else "amber",
            )
        )

    latest = data.dropna(subset=[metric_col]).sort_values("date")
    if not latest.empty:
        last = latest.iloc[-1]
        insights.append(
            Insight(
                kicker="LATEST VALUE",
                title=f"{metric_label}: {format_compact(last[metric_col])}",
                body=(
                    f"As of <strong>{format_date(last['date'])}</strong>, {esc(country)} reports "
                    f"<strong>{format_compact(last[metric_col])}</strong> on this metric. "
                    "The value is the last non-missing observation inside the selected filter."
                ),
                source=f"Metric: {metric_col} · last non-missing observation",
                tone="violet",
            )
        )
    return _clean(insights)
