"""Comparative Analytics — multi-country trajectories and per-capita comparisons."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from src import utils
from src.analysis import (
    align_by_days_since_threshold,
    continent_summary,
    country_series,
    filter_by_date,
    latest_snapshot,
    window_comparison,
    world_frame,
)
from src.data_loader import OWID_URL, load_owid_data
from src.insights import comparison_insights
from src.visualizations import comparison_lines, ranking_bar, simple_bar


utils.page_setup("Comparative Analytics")

try:
    with st.spinner("Loading the historical panel…"):
        df = load_owid_data()
except Exception as exc:  # pragma: no cover - UI guard
    utils.show_data_error(str(exc), detail="python -m src.data_loader --build")
    st.stop()

full_snapshot = latest_snapshot(df)
default_countries = (
    full_snapshot.dropna(subset=["total_cases"]).nlargest(5, "total_cases")["country"].tolist()
    if not full_snapshot.empty
    else []
)
country_options = sorted(full_snapshot["country"].dropna().astype(str).unique().tolist())

utils.sidebar_intro(
    latest=utils.format_date(df["date"].max()),
    rows=f"{len(df):,}",
    countries=str(df.loc[df["is_country"].astype(bool), "country"].nunique()),
    coverage=f"{utils.format_date(df['date'].min())} →",
)
utils.sidebar_label("Comparison filters")
start, end = utils.date_range_filter(df, key="analysis_date_range")
st.sidebar.caption(utils.date_window_caption(start, end))

countries = st.sidebar.multiselect(
    "Countries",
    country_options,
    default=[country for country in default_countries if country in country_options],
    max_selections=8,
    key="comparison_countries",
    help="Select up to eight countries. The same window and metric apply to all of them.",
)
METRICS = {
    "Confirmed cases": ("total_cases", "blue", False, "Confirmed cases"),
    "Deaths": ("total_deaths", "red", False, "Deaths"),
    "Cases per million": ("total_cases_per_million", "cyan", False, "Cases per million"),
    "Deaths per million": ("total_deaths_per_million", "red", False, "Deaths per million"),
    "Case fatality ratio": ("case_fatality_ratio", "violet", True, "CFR"),
    "First-dose coverage": ("vaccination_rate", "green", True, "At least one dose"),
    "New cases (7-day average)": ("new_cases_7d", "blue", False, "New cases / day"),
    "New deaths (7-day average)": ("new_deaths_7d", "red", False, "New deaths / day"),
}
metric_label = st.sidebar.selectbox("Comparison metric", list(METRICS), key="comparison_metric")
metric, scale, is_percent, axis_label = METRICS[metric_label]
threshold = st.sidebar.selectbox(
    "Alignment threshold",
    [100, 1_000, 10_000, 100_000, 1_000_000],
    index=2,
    format_func=lambda value: f"{value:,} cumulative cases",
    key="comparison_threshold",
)
utils.sidebar_note(
    "Cumulative comparisons are dominated by population size. Use cases or deaths per million for cross-country "
    "burden. Reporting coverage and death registration still differ between countries."
)

filtered = filter_by_date(df, start, end)
if filtered.empty:
    utils.page_header(
        "COMPARATIVE ANALYTICS",
        "Same window, same metric, different countries",
        "No observations fall inside the selected window. Widen the analysis window in the sidebar.",
        badge_label="FILTERED VIEW",
        badge_value="No records",
    )
    st.stop()

if not countries:
    utils.page_header(
        "COMPARATIVE ANALYTICS",
        "Same window, same metric, different countries",
        "Select at least one country in the sidebar to build a comparison.",
        badge_label="NO SELECTION",
        badge_value="Choose a country",
    )
    st.stop()

series = country_series(filtered, countries, metric)
if series.empty:
    utils.page_header(
        "COMPARATIVE ANALYTICS",
        "Same window, same metric, different countries",
        f"{metric_label} has no non-missing observations for the selected countries in this window.",
        badge_label="EMPTY SELECTION",
        badge_value="Try another metric",
    )
    st.stop()

latest_series = series.sort_values("date").groupby("country", as_index=False).tail(1)
leader = latest_series.nlargest(1, metric).iloc[0] if not latest_series.empty else None
trailer = latest_series.nsmallest(1, metric).iloc[0] if not latest_series.empty else None
snapshot = latest_snapshot(filtered)
selected_snapshot = snapshot[snapshot["country"].isin(countries)].copy()
span_days = int((pd.Timestamp(end) - pd.Timestamp(start)).days + 1)
latest_observation = series["date"].max()

utils.page_header(
    "COMPARATIVE ANALYTICS",
    "Same window, same metric, different countries",
    "Compare trajectories directly, align epidemic curves at a common threshold, and separate raw scale from population-adjusted burden.",
    accent="like for like",
    badge_label="COMPARISON WINDOW",
    badge_value=f"{len(countries)} countries · {span_days:,} days",
    badge_foot=f"Through {utils.format_date(latest_observation)} · OWID",
)

cards = [
    utils.kpi_card(
        "COUNTRIES SELECTED",
        f"{len(countries)}",
        sub="Shared metric and date window",
        tone="blue",
        icon_name="compare",
        chip="SCOPE",
    ),
    utils.kpi_card(
        "WINDOW LENGTH",
        f"{span_days:,} days",
        sub=utils.date_window_caption(start, end),
        tone="cyan",
        icon_name="calendar",
        chip="INCLUSIVE",
    ),
    utils.kpi_card(
        f"LEADER · {metric_label.upper()}",
        utils.format_pct(leader[metric], 2) if is_percent else utils.format_compact(leader[metric], 2),
        sub=str(leader["country"]),
        tone="green",
        icon_name="trend_up",
        chip="LATEST IN WINDOW",
    ),
    utils.kpi_card(
        "TRAILING VALUE",
        utils.format_pct(trailer[metric], 2) if is_percent else utils.format_compact(trailer[metric], 2),
        sub=str(trailer["country"]),
        tone="amber",
        icon_name="trend_down",
        chip="LATEST IN WINDOW",
    ),
    utils.kpi_card(
        "REPORTED TOTAL",
        utils.format_compact(selected_snapshot["total_cases"].sum(min_count=1), 2)
        if metric in {"total_cases", "total_deaths"} and "total_cases" in selected_snapshot
        else "—",
        sub="Sum of selected countries" if metric in {"total_cases", "total_deaths"} else "Not additive for this metric",
        tone="violet",
        icon_name="layers",
        chip="CUMULATIVE" if metric in {"total_cases", "total_deaths"} else "PER-COUNTRY",
    ),
    utils.kpi_card(
        "LATEST OBSERVATION",
        utils.format_date(latest_observation),
        sub="Most recent non-missing value shown",
        tone="red",
        icon_name="clock",
        chip="REPORTING DATE",
    ),
]
utils.render_kpis(cards)

# --------------------------------------------------------------------------- #
# Trajectories
# --------------------------------------------------------------------------- #
utils.section_header(
    "TRAJECTORIES",
    f"{metric_label} across the selected countries",
    "Each line uses the same metric and date window. Gaps are missing observations, not zeros.",
    note="No interpolation across missing dates",
)
with st.container(border=True):
    utils.plotly(
        comparison_lines(
            series,
            metric=metric,
            label=axis_label,
            height=410,
            x_title="Reported date",
        )
    )
    st.caption(
        "Vaccination and CFR series can have very different reporting tails. A flat or short line means the source "
        "stopped reporting that metric, not that the underlying value became zero."
    )

utils.section_header(
    "EPIDEMIC ALIGNMENT",
    "Curves aligned at the same cumulative threshold",
    f"Each country starts at its first day with at least {threshold:,} confirmed cases. This removes calendar timing and compares shape, not total size.",
    note="Cumulative confirmed cases",
)
aligned = align_by_days_since_threshold(
    filtered,
    countries,
    metric="total_cases",
    threshold=float(threshold),
)
with st.container(border=True):
    if aligned.empty:
        utils.empty_state(
            "No country crossed this threshold",
            "Choose a lower cumulative-case threshold or include a country with a larger reported outbreak.",
        )
    else:
        utils.plotly(
            comparison_lines(
                aligned,
                metric="total_cases",
                label="Confirmed cases",
                height=390,
                x_col="day",
                x_title="Days since crossing threshold",
            )
        )
        st.caption(
            "Alignment is based on reported cumulative cases. Testing capacity and case definitions affect when a "
            "country appears to cross a threshold, so the aligned curves are descriptive rather than biological."
        )

# --------------------------------------------------------------------------- #
# Rankings and regional context
# --------------------------------------------------------------------------- #
utils.section_header(
    "RANKINGS",
    "Selected countries and regional context",
    "The left chart compares only the selected countries. The right chart aggregates all countries in each continent for context.",
    note="Latest value in window",
)
rank_left, rank_right = st.columns([1, 1], gap="medium")
with rank_left:
    with st.container(border=True):
        st.markdown(
            '<div class="section-eyebrow">SELECTED COUNTRIES</div>'
            f'<div class="section-title">{utils.esc(metric_label)}</div>',
            unsafe_allow_html=True,
        )
        utils.plotly(
            ranking_bar(
                latest_series,
                metric=metric,
                label=axis_label,
                n=len(countries),
                colorscale=scale,
                percent=is_percent,
                height=max(330, 46 * len(countries) + 70),
            )
        )
with rank_right:
    with st.container(border=True):
        st.markdown(
            '<div class="section-eyebrow">CONTINENT CONTEXT</div>'
            f'<div class="section-title">{utils.esc(metric_label)} by continent</div>',
            unsafe_allow_html=True,
        )
        if metric not in snapshot.columns:
            utils.empty_state(
                "Regional aggregation unavailable",
                "This metric is a daily flow and is not aggregated into the country snapshot used for regional context.",
            )
        else:
            regions = continent_summary(snapshot, metric)
            if regions.empty:
                utils.empty_state("No regional values", "The selected metric has no continent-level observations.")
            else:
                utils.plotly(
                    simple_bar(
                        regions["continent"].tolist(),
                        regions["value"].tolist(),
                        label=axis_label,
                        color=utils.COLORS["cyan"],
                        percent=is_percent,
                        height=390,
                    )
                )
                st.caption("Population-aware aggregation where the metric requires it; country counts differ by continent.")

# --------------------------------------------------------------------------- #
# Country sheet
# --------------------------------------------------------------------------- #
utils.section_header(
    "COUNTRY SHEET",
    "Latest cumulative values for the selected countries",
    "The table is cumulative and is kept separate from daily-flow metrics. Latest dates can differ by country.",
    note=f"{len(selected_snapshot)} countries",
)
table_source = selected_snapshot.sort_values("total_cases", ascending=False).copy()
table = pd.DataFrame(
    {
        "Country": table_source["country"],
        "Confirmed": table_source["total_cases"].map(utils.format_int),
        "Deaths": table_source["total_deaths"].map(utils.format_int),
        "Cases / million": table_source["total_cases_per_million"].map(utils.format_compact),
        "Deaths / million": table_source["total_deaths_per_million"].map(utils.format_compact),
        "CFR": table_source["case_fatality_ratio"].map(lambda value: utils.format_pct(value, 2)),
        "First dose": table_source["vaccination_rate"].map(lambda value: utils.format_pct(value, 1)),
        "Population": table_source["population"].map(utils.format_compact),
        "Data as of": table_source["date"].map(utils.format_date),
    }
)
with st.container(border=True):
    st.dataframe(table, width="stretch", hide_index=True, height=min(480, 42 * len(table) + 44))
    st.caption(
        "Cumulative values are the last non-missing observation in the window. “First dose” may have an earlier "
        "reporting date than the case observation; OWID's vaccination series ends in August 2024."
    )

# --------------------------------------------------------------------------- #
# Recent global context
# --------------------------------------------------------------------------- #
utils.section_header(
    "RECENT CONTEXT",
    "The world's last 30 days versus the preceding 30",
    "This context uses the World aggregate and is independent of the country selection. It describes reported activity only.",
    note="Not a forecast",
)
world = world_frame(filtered)
period_days = min(30, max(1, span_days))
recent = window_comparison(world, "new_cases", end, days=period_days)
change_text = utils.format_signed_pct(recent["change_pct"]) if recent["change_pct"] is not None else "—"
utils.render_kpis(
    [
        utils.kpi_card(
            f"LAST {period_days} DAYS",
            utils.format_compact(recent["current"], 1),
            sub="Global new cases, summed",
            tone="cyan",
            icon_name="activity",
            chip="REPORTED",
        ),
        utils.kpi_card(
            "PRECEDING WINDOW",
            utils.format_compact(recent["previous"], 1),
            sub="Equal-length comparison window",
            tone="violet",
            icon_name="clock",
            chip="REPORTED",
        ),
        utils.kpi_card(
            "CHANGE",
            change_text,
            sub="Not a forecast or causal estimate",
            tone="amber" if recent["change_pct"] is None or recent["change_pct"] >= 0 else "green",
            icon_name="trend_up" if recent["change_pct"] is None or recent["change_pct"] >= 0 else "trend_down",
            chip="WINDOW VS WINDOW",
        ),
    ]
)
if recent["change_pct"] is None:
    st.caption("A percentage change is unavailable because the preceding window is missing or sums to zero.")

# --------------------------------------------------------------------------- #
# Insights
# --------------------------------------------------------------------------- #
utils.section_header(
    "OBSERVATIONS",
    "What the comparison supports",
    "Rule-based observations from the selected countries, metric and window. Each card records its date and metric basis.",
    note="No hard-coded statistics",
)
utils.insight_grid(
    comparison_insights(
        filtered,
        countries,
        metric,
        metric_label,
        start=start,
        end=end,
    )
)

utils.source_strip(
    source="Our World in Data — COVID-19 Dataset",
    url=OWID_URL,
    latest=utils.format_date(df["date"].max()),
)
st.caption(
    "All calculations are descriptive. Cumulative totals are sensitive to population and testing; per-capita rates "
    "depend on population denominators. CFR is reported deaths ÷ reported cases, not an infection fatality ratio."
)
