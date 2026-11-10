"""India Analysis — national waves, state map and state rankings."""
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
    india_totals,
    latest_national_india,
    latest_state_snapshot,
    peak_of,
    rank_entities,
)
from src.data_cleaning import prepare_india_map_data
from src.data_loader import INDIA_URL, load_india_data, load_india_geojson
from src.insights import india_insights
from src.visualizations import daily_chart, dual_axis_trend, india_map, ranking_bar, trend_chart


utils.page_setup("India Analysis")

try:
    with st.spinner("Loading the covid19india state archive…"):
        india = load_india_data()
except Exception as exc:  # pragma: no cover - UI guard
    utils.show_data_error(str(exc), detail="python -m src.data_loader --build")
    st.stop()

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
utils.sidebar_intro(
    latest=utils.format_date(india["date"].max()),
    rows=f"{len(india):,}",
    countries=str(india.loc[india["is_state"].astype(bool), "state"].nunique()),
    entity_label="states/UTs",
    coverage=f"{utils.format_date(india['date'].min())} →",
)
utils.sidebar_label("India archive")
start, end = utils.date_range_filter(
    india,
    key="india_date_range",
    label="India archive window",
)
st.sidebar.caption(utils.date_window_caption(start, end))

MAP_METRICS = {
    "Confirmed cases": ("total_cases", "blue", False, "Confirmed cases"),
    "Deaths": ("total_deaths", "red", False, "Deaths"),
    "Recoveries": ("total_recovered", "green", False, "Recoveries"),
    "Active cases": ("active_cases", "amber", False, "Active cases"),
    "Case fatality ratio": ("case_fatality_ratio", "violet", True, "CFR"),
    "Recovery ratio": ("recovery_ratio", "green", True, "Recovery ratio"),
}
map_label = st.sidebar.selectbox("State map metric", list(MAP_METRICS), key="india_map_metric")
rank_label = st.sidebar.selectbox(
    "State ranking metric",
    [key for key in MAP_METRICS if key != "Case fatality ratio"],
    key="india_rank_metric",
)
utils.sidebar_note(
    "The public covid19india archive ends on <b>1 June 2021</b>. State populations are "
    "not included in the source, so per-million state comparisons are deliberately omitted."
)

filtered = filter_by_date(india, start, end)
if filtered.empty:
    utils.page_header(
        "INDIA ANALYSIS",
        "States, waves and an archive that ends in 2021",
        "No Indian observations fall inside the selected window. Widen the archive window in the sidebar.",
        badge_label="FILTERED VIEW",
        badge_value="No records",
    )
    st.stop()

national = filtered[filtered["entity_type"].eq("National")].sort_values("date")
totals = india_totals(filtered)
states = latest_state_snapshot(filtered)
latest_date = totals.get("date") or filtered["date"].max()
latest_row = latest_national_india(filtered)
peak_date, peak_value = peak_of(national.set_index("date")["new_cases"]) if not national.empty else (None, float("nan"))

# --------------------------------------------------------------------------- #
# Header and KPIs
# --------------------------------------------------------------------------- #
utils.page_header(
    "INDIA ANALYSIS",
    "States, waves and an archive that ends in 2021",
    "National and state/UT analysis from the covid19india.org public archive, with recoveries, active cases and reporting caveats stated explicitly.",
    accent="state by state",
    badge_label="HISTORICAL ARCHIVE",
    badge_value=f"Through {utils.format_date(latest_date)}",
    badge_foot=f"{len(states)} states/UTs · covid19india",
)

cards = [
    utils.kpi_card(
        "TOTAL CONFIRMED",
        utils.format_compact(totals.get("confirmed"), 2),
        sub=f"{utils.format_int(totals.get('confirmed'))} reported cases",
        tone="blue",
        icon_name="activity",
        chip="NATIONAL ROW",
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
        "TOTAL RECOVERIES",
        utils.format_compact(totals.get("recovered"), 2),
        sub="Reported cumulative recoveries",
        tone="green",
        icon_name="heart",
        chip="SOURCE SERIES",
    ),
    utils.kpi_card(
        "ACTIVE CASES",
        utils.format_compact(totals.get("active"), 2),
        sub="Confirmed − recovered − deaths",
        tone="amber",
        icon_name="layers",
        chip="DERIVED · CLIPPED",
    ),
    utils.kpi_card(
        "CASE FATALITY RATIO",
        utils.format_pct(totals.get("cfr"), 2),
        sub="Deaths ÷ confirmed cases",
        tone="violet",
        icon_name="percent",
        chip="CFR ≠ IFR",
    ),
    utils.kpi_card(
        "RECOVERY RATIO",
        utils.format_pct(totals.get("recovery_rate"), 2),
        sub=(
            f"Peak day: {utils.format_compact(peak_value)} on {utils.format_date(peak_date)}"
            if peak_date is not None
            else "No daily peak available"
        ),
        tone="cyan",
        icon_name="trend_up",
        chip="REPORTED SHARE",
    ),
]
utils.render_kpis(cards)

utils.callout(
    "<strong>What this archive can and cannot say.</strong> Recoveries and active cases are available here because "
    "the source reports all three components. Active cases are calculated as confirmed − recovered − deaths and clipped "
    "at zero; they are an estimate, not an independently reported series. The public archive ends on 1 June 2021, so "
    "nothing on this page should be read as a current national status.",
    title="Historical archive · 1 June 2021",
    tone="amber",
    icon_name="alert",
)

# --------------------------------------------------------------------------- #
# National trajectory
# --------------------------------------------------------------------------- #
utils.section_header(
    "NATIONAL TRAJECTORY",
    "The shape of India's reported waves",
    "Cumulative totals, daily reporting, active burden and mortality are shown together so scale and timing are not confused.",
    note=utils.date_window_caption(start, end),
)
left, right = st.columns([1.4, 1], gap="medium")
with left:
    with st.container(border=True):
        st.markdown(
            '<div class="section-eyebrow">CUMULATIVE</div>'
            '<div class="section-title">Confirmed cases and deaths</div>',
            unsafe_allow_html=True,
        )
        utils.plotly(
            dual_axis_trend(
                national,
                cases_col="total_cases",
                deaths_col="total_deaths",
                cases_label="Confirmed cases",
                deaths_label="Deaths (right axis)",
                height=310,
            )
        )
with right:
    with st.container(border=True):
        st.markdown(
            '<div class="section-eyebrow">DAILY REPORTING</div>'
            '<div class="section-title">New cases and 7-day average</div>',
            unsafe_allow_html=True,
        )
        utils.plotly(
            daily_chart(
                national,
                daily_col="new_cases",
                rolling_col="new_cases_7d",
                label="New cases",
                height=310,
            )
        )

left2, right2 = st.columns([1, 1], gap="medium")
with left2:
    with st.container(border=True):
        st.markdown(
            '<div class="section-eyebrow">ACTIVE BURDEN</div>'
            '<div class="section-title">Active cases over time</div>',
            unsafe_allow_html=True,
        )
        utils.plotly(
            trend_chart(
                national,
                y="active_cases",
                label="Active cases",
                color=utils.COLORS["amber"],
                height=300,
            )
        )
with right2:
    with st.container(border=True):
        st.markdown(
            '<div class="section-eyebrow">MORTALITY</div>'
            '<div class="section-title">New deaths and 7-day average</div>',
            unsafe_allow_html=True,
        )
        utils.plotly(
            daily_chart(
                national,
                daily_col="new_deaths",
                rolling_col="new_deaths_7d",
                label="New deaths",
                color=utils.COLORS["red"],
                height=300,
            )
        )

st.caption(
    "Daily values are differences of cumulative totals. Negative values are retained where the source revised a "
    "cumulative series downward; the 7-day average is in white. Reporting batches can make single days misleading."
)

# --------------------------------------------------------------------------- #
# State map
# --------------------------------------------------------------------------- #
utils.section_header(
    "STATE GEOGRAPHY",
    "Where reported state caseloads sit",
    "The map uses each state/UT's latest available cumulative value inside the selected archive window.",
    note="State counts only · populations unavailable",
)
map_col, map_scale, map_percent, map_title = MAP_METRICS[map_label]
snapshot_for_map = states.dropna(subset=[map_col]) if map_col in states.columns else states.iloc[0:0]
with st.container(border=True):
    if snapshot_for_map.empty:
        utils.empty_state(
            "No state map data in this window",
            "The selected metric has no non-missing state observations in the current archive window.",
        )
    else:
        geojson = load_india_geojson()
        if geojson is None:
            utils.empty_state(
                "State boundaries are unavailable",
                "The India GeoJSON could not be loaded. The rankings and table below remain fully usable.",
            )
        else:
            mapped = prepare_india_map_data(snapshot_for_map, map_col)
            utils.plotly(
                india_map(
                    mapped,
                    geojson,
                    metric=map_col,
                    label=map_title,
                    colorscale=map_scale,
                    height=555,
                    percent=map_percent,
                )
            )
            st.caption(
                "Pre-merger boundaries are used. The combined Dadra & Nagar Haveli and Daman & Diu record is drawn "
                "on both historical polygons. Grey states have no matching value."
            )

# --------------------------------------------------------------------------- #
# State rankings
# --------------------------------------------------------------------------- #
utils.section_header(
    "STATE RANKINGS",
    "Largest reported state loads",
    "Rankings use the same metric as the map selector on the left and a fixed confirmed-case table on the right.",
    note="Latest state snapshot in window",
)
rank_col, rank_scale, rank_percent, rank_title = MAP_METRICS[rank_label]
rank_left, rank_right = st.columns([1, 1.3], gap="medium")
with rank_left:
    with st.container(border=True):
        ranked = rank_entities(states, rank_col, entity_col="state", n=10)
        if ranked.empty:
            utils.empty_state("No ranking data", "The selected metric has no state values in this window.")
        else:
            utils.plotly(
                ranking_bar(
                    ranked,
                    metric=rank_col,
                    label=rank_title,
                    n=10,
                    name_col="state",
                    colorscale=rank_scale,
                    percent=rank_percent,
                    height=430,
                )
            )
with rank_right:
    with st.container(border=True):
        table_source = states.dropna(subset=["total_cases"]).sort_values("total_cases", ascending=False).head(15)
        table = pd.DataFrame(
            {
                "State / UT": table_source["state"],
                "Confirmed": table_source["total_cases"].map(utils.format_int),
                "Recovered": table_source["total_recovered"].map(utils.format_int),
                "Deaths": table_source["total_deaths"].map(utils.format_int),
                "Active": table_source["active_cases"].map(utils.format_int),
                "CFR": table_source["case_fatality_ratio"].map(lambda value: utils.format_pct(value, 2)),
                "Recovery": table_source["recovery_ratio"].map(lambda value: utils.format_pct(value, 1)),
                "Data as of": table_source["date"].map(utils.format_date),
            }
        )
        st.dataframe(table, width="stretch", hide_index=True, height=430)
        st.caption(
            "State recovery counts are self-reported and can lag case updates. Active cases are derived from the "
            "three cumulative series and clipped at zero."
        )

# --------------------------------------------------------------------------- #
# Insights
# --------------------------------------------------------------------------- #
utils.section_header(
    "OBSERVATIONS",
    "What the archive supports",
    "Every card is computed from the filtered national/state frame and states its metric and date basis.",
    note="No hard-coded statistics",
)
utils.insight_grid(
    india_insights(
        totals,
        states,
        national_series=national,
    )
)

utils.source_strip(
    source="covid19india.org — State-wise archive",
    url=INDIA_URL,
    latest=utils.format_date(india["date"].max()),
)
st.caption(
    "The covid19india.org project was a volunteer-led aggregator. Its public CSV archive ends on 1 June 2021; "
    "retrospective revisions and reporting gaps are part of the source. This page never claims to be live."
)
