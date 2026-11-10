"""Global Overview — world totals, trends, map and rankings."""
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
    filter_by_date,
    global_totals,
    latest_metric_date,
    latest_snapshot,
    peak_of,
    rank_entities,
    vaccination_coverage,
    world_frame,
)
from src.data_loader import load_owid_data
from src.insights import global_insights
from src.visualizations import daily_chart, dual_axis_trend, ranking_bar, world_map


utils.page_setup("Global Overview")

try:
    with st.spinner("Loading the historical panel…"):
        df = load_owid_data()
except Exception as exc:  # pragma: no cover - UI guard
    utils.show_data_error(str(exc), detail="python -m src.data_loader --build")
    st.stop()

# --------------------------------------------------------------------------- #
# Sidebar filters
# --------------------------------------------------------------------------- #
utils.sidebar_intro(
    latest=utils.format_date(df["date"].max()),
    rows=f"{len(df):,}",
    countries=str(df.loc[df["is_country"].astype(bool), "country"].nunique()),
    coverage=f"{utils.format_date(df['date'].min())} →",
)
utils.sidebar_label("Global filters")
start, end = utils.date_range_filter(df, key="analysis_date_range")
st.sidebar.caption(utils.date_window_caption(start, end))

MAP_METRICS = {
    "Cases per million": ("total_cases_per_million", "blue", False, "Cases per million"),
    "Deaths per million": ("total_deaths_per_million", "red", False, "Deaths per million"),
    "Total confirmed cases": ("total_cases", "blue", False, "Confirmed cases"),
    "Total deaths": ("total_deaths", "red", False, "Deaths"),
    "First-dose coverage": ("vaccination_rate", "green", True, "At least one dose"),
}
map_label = st.sidebar.selectbox("Map metric", list(MAP_METRICS), key="global_map_metric")
RANK_METRICS = {
    "Confirmed cases": ("total_cases", "blue", False, "Confirmed cases"),
    "Deaths": ("total_deaths", "red", False, "Deaths"),
    "Cases per million": ("total_cases_per_million", "cyan", False, "Cases per million"),
    "Deaths per million": ("total_deaths_per_million", "red", False, "Deaths per million"),
    "First-dose coverage": ("vaccination_rate", "green", True, "At least one dose"),
}
rank_label = st.sidebar.selectbox("Ranking metric", list(RANK_METRICS), key="global_rank_metric", index=0)
utils.sidebar_note(
    "Global recoveries are not shown because OWID does not publish a consistent "
    "recovery series. Active cases are therefore not derived at world level."
)

filtered = filter_by_date(df, start, end)
if filtered.empty:
    utils.page_header(
        "GLOBAL OVERVIEW",
        "The world's reported pandemic",
        "No observations fall inside the selected window. Widen the analysis window in the sidebar.",
        badge_label="FILTERED VIEW",
        badge_value="No records",
    )
    st.stop()

world = world_frame(filtered)
snapshot = latest_snapshot(filtered)
totals = global_totals(filtered)
latest_date = totals.get("date") or filtered["date"].max()
vax = vaccination_coverage(snapshot)
countries_reporting = int(snapshot["country"].nunique()) if not snapshot.empty else 0

# --------------------------------------------------------------------------- #
# Header + KPIs
# --------------------------------------------------------------------------- #
utils.page_header(
    "GLOBAL OVERVIEW",
    "The world's reported pandemic",
    "Cumulative burden, daily activity, per-capita geography and country rankings derived from the selected historical window.",
    accent="at a glance",
    badge_label="HISTORICAL ARCHIVE",
    badge_value=f"Through {utils.format_date(latest_date)}",
    badge_foot=f"{countries_reporting} countries · OWID",
)

latest_row = world.sort_values("date").tail(1)
latest_7d = (
    float(latest_row["new_cases_7d"].iloc[0])
    if not latest_row.empty and "new_cases_7d" in latest_row and pd.notna(latest_row["new_cases_7d"].iloc[0])
    else float("nan")
)
peak_date, peak_value = peak_of(world.set_index("date")["new_cases"]) if not world.empty else (None, float("nan"))

cards = [
    utils.kpi_card(
        "TOTAL CONFIRMED",
        utils.format_compact(totals.get("confirmed"), 2),
        sub=f"{utils.format_int(totals.get('confirmed'))} reported cases",
        tone="blue",
        icon_name="activity",
        chip="CUMULATIVE",
    ),
    utils.kpi_card(
        "TOTAL DEATHS",
        utils.format_compact(totals.get("deaths"), 2),
        sub=f"{utils.format_int(totals.get('deaths'))} reported deaths",
        tone="red",
        icon_name="shield",
        chip="REPORTED",
    ),
    utils.kpi_card(
        "CASE FATALITY RATIO",
        utils.format_pct(totals.get("cfr"), 2),
        sub="Reported deaths ÷ reported cases",
        tone="amber",
        icon_name="percent",
        chip="CFR ≠ IFR",
    ),
    utils.kpi_card(
        "FIRST-DOSE COVERAGE",
        utils.format_pct(vax["weighted"], 1) if pd.notna(vax["weighted"]) else "—",
        sub=(
            f"Population-weighted · {vax['countries']} countries"
            if vax["countries"]
            else "Vaccination records unavailable"
        ),
        tone="green",
        icon_name="syringe",
        chip=f"AS OF {utils.format_date(vax['as_of'], '%b %Y').upper()}"
        if vax["as_of"] is not None and pd.notna(vax["as_of"])
        else "HISTORICAL",
    ),
    utils.kpi_card(
        "COUNTRIES REPORTING",
        f"{countries_reporting:,}",
        sub="With cumulative totals in window",
        tone="cyan",
        icon_name="globe",
        chip="COVERAGE",
    ),
    utils.kpi_card(
        "7-DAY AVERAGE",
        utils.format_compact(latest_7d, 1) if pd.notna(latest_7d) else "—",
        sub=(
            f"Peak day: {utils.format_compact(peak_value)} on {utils.format_date(peak_date)}"
            if peak_date is not None
            else "Daily case average unavailable"
        ),
        tone="violet",
        icon_name="trend_up",
        chip="NEW CASES / DAY",
    ),
]
utils.render_kpis(cards)

utils.callout(
    "<strong>Recoveries are intentionally absent.</strong> Our World in Data does not provide a "
    "consistent global recovery series, so active cases cannot be calculated reliably at world level. "
    "The India page shows recoveries and active cases because all three component series exist there.",
    title="A deliberate omission, not a missing chart",
    tone="amber",
    icon_name="info",
)

# --------------------------------------------------------------------------- #
# Trajectory
# --------------------------------------------------------------------------- #
utils.section_header(
    "TRAJECTORY",
    "Cumulative and daily reported activity",
    "Cumulative totals show scale; daily values and the 7-day average show timing and wave structure.",
    note=utils.date_window_caption(start, end),
)
row1, row2 = st.columns([1.45, 1], gap="medium")
with row1:
    with st.container(border=True):
        st.markdown(
            '<div class="section-eyebrow">CUMULATIVE</div>'
            '<div class="section-title">Confirmed cases and deaths over time</div>',
            unsafe_allow_html=True,
        )
        utils.plotly(
            dual_axis_trend(
                world,
                cases_col="total_cases",
                deaths_col="total_deaths",
                cases_label="Confirmed cases",
                deaths_label="Deaths (right axis)",
                height=320,
            )
        )
with row2:
    with st.container(border=True):
        st.markdown(
            '<div class="section-eyebrow">DAILY CASES</div>'
            '<div class="section-title">New cases and 7-day average</div>',
            unsafe_allow_html=True,
        )
        utils.plotly(
            daily_chart(
                world,
                daily_col="new_cases",
                rolling_col="new_cases_7d",
                label="New cases",
                height=320,
            )
        )

row3, row4 = st.columns([1, 1], gap="medium")
with row3:
    with st.container(border=True):
        st.markdown(
            '<div class="section-eyebrow">DAILY DEATHS</div>'
            '<div class="section-title">New deaths and 7-day average</div>',
            unsafe_allow_html=True,
        )
        utils.plotly(
            daily_chart(
                world,
                daily_col="new_deaths",
                rolling_col="new_deaths_7d",
                label="New deaths",
                color=utils.COLORS["red"],
                height=310,
            )
        )
with row4:
    top = rank_entities(snapshot, "total_cases", n=10)
    with st.container(border=True):
        st.markdown(
            '<div class="section-eyebrow">RANKING</div>'
            '<div class="section-title">Top 10 countries by confirmed cases</div>',
            unsafe_allow_html=True,
        )
        utils.plotly(
            ranking_bar(
                top,
                metric="total_cases",
                label="Confirmed cases",
                n=10,
                height=310,
                colorscale="blue",
            )
        )

st.caption(
    "Bars show raw daily reports, including negative values when a country revises its cumulative series downward. "
    "The white line is the 7-day average."
)

# --------------------------------------------------------------------------- #
# Map
# --------------------------------------------------------------------------- #
utils.section_header(
    "GEOGRAPHY",
    "Where the reported burden is concentrated",
    "The default view uses population-adjusted rates, which are more comparable across countries of different sizes.",
    note="Latest available value per country",
)
map_col, map_scale, map_percent, map_title = MAP_METRICS[map_label]
mapped = snapshot.dropna(subset=[map_col]) if map_col in snapshot.columns else snapshot.iloc[0:0]
with st.container(border=True):
    if mapped.empty:
        utils.empty_state(
            "No map data in this window",
            "The selected metric has no non-missing country observations inside the current date filter.",
        )
    else:
        utils.plotly(
            world_map(
                mapped,
                metric=map_col,
                label=map_title,
                colorscale=map_scale,
                height=520,
                percent=map_percent,
            )
        )
        latest_map_date = latest_metric_date(filtered, map_col)
        st.caption(
            f"Latest observation in window: {utils.format_date(latest_map_date)}. "
            "Countries without a reported value are uncoloured. Per-capita rates depend on population denominators."
        )

if map_label == "First-dose coverage":
    utils.callout(
        "OWID's vaccination series has a different reporting tail from the case series. Coverage dates vary by "
        "country and the vaccination data in this archive ends in August 2024, so this map is a historical "
        "coverage snapshot rather than a current vaccination status.",
        title="Different reporting tail",
        tone="green",
        icon_name="syringe",
    )

# --------------------------------------------------------------------------- #
# Country sheet
# --------------------------------------------------------------------------- #
utils.section_header(
    "COUNTRY SHEET",
    "Rankings and latest country values",
    "The table uses each country's last non-missing cumulative observation inside the selected window.",
    note=f"Top 15 by {rank_label.lower()}",
)
rank_col, rank_scale, rank_percent, rank_title = RANK_METRICS[rank_label]
left, right = st.columns([1, 1.3], gap="medium")
with left:
    with st.container(border=True):
        ranked = rank_entities(snapshot, rank_col, n=10)
        utils.plotly(
            ranking_bar(
                ranked,
                metric=rank_col,
                label=rank_title,
                n=10,
                colorscale=rank_scale,
                percent=rank_percent,
                height=440,
            )
        )
with right:
    with st.container(border=True):
        table_source = snapshot.dropna(subset=[rank_col]).sort_values(rank_col, ascending=False).head(15)
        table = pd.DataFrame(
            {
                "Country": table_source["country"],
                "Confirmed": table_source["total_cases"].map(utils.format_int),
                "Deaths": table_source["total_deaths"].map(utils.format_int),
                "CFR": table_source["case_fatality_ratio"].map(lambda value: utils.format_pct(value, 2)),
                "Cases / million": table_source["total_cases_per_million"].map(utils.format_compact),
                "First dose": table_source["vaccination_rate"].map(lambda value: utils.format_pct(value, 1)),
                "Data as of": table_source["date"].map(utils.format_date),
            }
        )
        st.dataframe(
            table,
            width="stretch",
            hide_index=True,
            height=440,
            column_config={
                "Country": st.column_config.TextColumn(width="medium"),
                "Confirmed": st.column_config.TextColumn(width="small"),
                "Deaths": st.column_config.TextColumn(width="small"),
                "CFR": st.column_config.TextColumn(width="small"),
                "Cases / million": st.column_config.TextColumn(width="small"),
                "First dose": st.column_config.TextColumn(width="small"),
                "Data as of": st.column_config.TextColumn(width="small"),
            },
        )
        st.caption("“First dose” is the latest available value and may predate the case observation.")

# --------------------------------------------------------------------------- #
# Insights
# --------------------------------------------------------------------------- #
utils.section_header(
    "OBSERVATIONS",
    "What the selected data actually shows",
    "Rule-based observations generated from the filtered frame. Each card states the metric and date basis.",
    note="No hard-coded statistics",
)
utils.insight_grid(
    global_insights(
        filtered,
        snapshot,
        totals,
        start=start,
        end=end,
    )
)

utils.source_strip(
    source="Our World in Data — COVID-19 Dataset",
    url="https://catalog.ourworldindata.org/garden/covid/latest/compact/compact.csv",
    latest=utils.format_date(df["date"].max()),
)
st.caption(
    "Population-adjusted metrics use the population value supplied by OWID. CFR is reported deaths divided by "
    "reported confirmed cases; it is not an infection fatality ratio."
)
