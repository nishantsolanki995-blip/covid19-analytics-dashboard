"""Time Series & Insights — peaks, seasonality, distributions and trend observations."""
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
    detect_peaks,
    filter_by_date,
    monthly_heatmap_matrix,
    monthly_summary,
    peak_of,
    trend_change,
    window_comparison,
    world_frame,
)
from src.data_loader import OWID_URL, load_owid_data
from src.insights import timeseries_insights
from src.visualizations import (
    box_plot,
    daily_chart,
    distribution_plot,
    monthly_bar_chart,
    monthly_heatmap,
    style_matplotlib,
)


utils.page_setup("Time Series & Insights")
style_matplotlib()

try:
    with st.spinner("Loading the historical panel…"):
        df = load_owid_data()
except Exception as exc:  # pragma: no cover - UI guard
    utils.show_data_error(str(exc), detail="python -m src.data_loader --build")
    st.stop()

country_options = ["World"] + sorted(
    df.loc[df["is_country"].astype(bool), "country"].dropna().astype(str).unique().tolist()
)

utils.sidebar_intro(
    latest=utils.format_date(df["date"].max()),
    rows=f"{len(df):,}",
    countries=str(df.loc[df["is_country"].astype(bool), "country"].nunique()),
    coverage=f"{utils.format_date(df['date'].min())} →",
)
utils.sidebar_label("Time-series filters")
start, end = utils.date_range_filter(df, key="analysis_date_range")
st.sidebar.caption(utils.date_window_caption(start, end))
entity = st.sidebar.selectbox("Country or aggregate", country_options, index=0, key="ts_entity")
METRICS = {
    "Daily cases": ("new_cases", "new_cases_7d", "total_cases", "New cases", "Cumulative confirmed cases"),
    "Daily deaths": ("new_deaths", "new_deaths_7d", "total_deaths", "New deaths", "Cumulative deaths"),
}
metric_label = st.sidebar.selectbox("Daily metric", list(METRICS), key="ts_metric")
daily_col, rolling_col, cumulative_col, daily_label, cumulative_label = METRICS[metric_label]
utils.sidebar_note(
    "Peaks use a separated local-maximum rule, not a fitted model. Monthly aggregation reduces daily noise but can "
    "still reflect batch releases and retrospective revisions."
)

filtered = filter_by_date(df, start, end)
if filtered.empty:
    utils.page_header(
        "TIME SERIES & INSIGHTS",
        "Shape, peaks and monthly structure",
        "No observations fall inside the selected window. Widen the analysis window in the sidebar.",
        badge_label="FILTERED VIEW",
        badge_value="No records",
    )
    st.stop()

data = (
    world_frame(filtered)
    if entity == "World"
    else filtered[filtered["country"].eq(entity)].sort_values("date")
)
if data.empty:
    utils.page_header(
        "TIME SERIES & INSIGHTS",
        "Shape, peaks and monthly structure",
        f"No observations are available for {entity} in the selected window.",
        badge_label="EMPTY SERIES",
        badge_value=entity,
    )
    st.stop()

data = data.sort_values("date").copy()
span_days = int((pd.Timestamp(end) - pd.Timestamp(start)).days + 1)
peak_date, peak_value = peak_of(data.set_index("date")[daily_col])
latest_metric = data.dropna(subset=[cumulative_col]).sort_values("date").tail(1)
latest_7d_values = data[rolling_col].dropna()
latest_7d = float(latest_7d_values.iloc[-1]) if not latest_7d_values.empty else float("nan")
trend = trend_change(data, rolling_col, window=28)
latest_date = latest_metric["date"].iloc[0] if not latest_metric.empty else data["date"].max()
reported_days = int(data[daily_col].notna().sum())

utils.page_header(
    "TIME SERIES & INSIGHTS",
    "Shape, peaks and monthly structure",
    f"A single-series investigation of {entity}, from raw daily reports through peaks, monthly seasonality and distributions.",
    accent="one series at a time",
    badge_label="SERIES WINDOW",
    badge_value=f"{entity} · {span_days:,} days",
    badge_foot=f"Through {utils.format_date(latest_date)} · OWID",
)

cards = [
    utils.kpi_card(
        "CUMULATIVE CONFIRMED",
        utils.format_compact(latest_metric["total_cases"].iloc[0] if not latest_metric.empty else float("nan"), 2),
        sub=f"As of {utils.format_date(latest_date)}",
        tone="blue",
        icon_name="activity",
        chip="REPORTED",
    ),
    utils.kpi_card(
        "CUMULATIVE DEATHS",
        utils.format_compact(latest_metric["total_deaths"].iloc[0] if not latest_metric.empty else float("nan"), 2),
        sub="Deaths reported to the source",
        tone="red",
        icon_name="shield",
        chip="REPORTED",
    ),
    utils.kpi_card(
        "LATEST 7-DAY AVERAGE",
        utils.format_compact(latest_7d, 1),
        sub=f"{daily_label} per day",
        tone="cyan",
        icon_name="trend_up",
        chip="SMOOTHED",
    ),
    utils.kpi_card(
        "PEAK DAY",
        utils.format_compact(peak_value, 1),
        sub=f"On {utils.format_date(peak_date)}" if peak_date is not None else "No positive daily values",
        tone="amber",
        icon_name="alert",
        chip="RAW DAILY",
    ),
    utils.kpi_card(
        "REPORTING DAYS",
        f"{reported_days:,}",
        sub=f"of {len(data):,} rows in window",
        tone="violet",
        icon_name="calendar",
        chip="NON-MISSING",
    ),
    utils.kpi_card(
        "28-OBSERVATION TREND",
        trend.get("direction", "unavailable").title(),
        sub=(
            f"{utils.format_signed_pct(trend.get('change_pct'))} vs preceding 28"
            if trend.get("change_pct") is not None
            else "Insufficient baseline for a change estimate"
        ),
        tone="green" if (trend.get("change_pct") or 0) < 0 else "amber",
        icon_name="trend_down" if (trend.get("change_pct") or 0) < 0 else "trend_up",
        chip="DESCRIPTIVE",
    ),
]
utils.render_kpis(cards)

utils.callout(
    "A zero in the daily series means the source reported no change for that day; it does not automatically mean "
    "there was no transmission. OWID's late-period rows are sparse and uneven across countries, and raw daily values "
    "can contain batch releases. The 7-day average is the appropriate line for direction.",
    title="Zero is a report, not a biological fact",
    tone="amber",
    icon_name="info",
)

# --------------------------------------------------------------------------- #
# Daily activity
# --------------------------------------------------------------------------- #
utils.section_header(
    "DAILY ACTIVITY",
    f"{entity}: {daily_label.lower()} and the 7-day average",
    "Bars are raw reported values; the white line is a trailing 7-day mean. Negative bars are reporting revisions.",
    note=utils.date_window_caption(start, end),
)
with st.container(border=True):
    utils.plotly(
        daily_chart(
            data,
            daily_col=daily_col,
            rolling_col=rolling_col,
            label=daily_label,
            color=utils.COLORS["blue"] if daily_col == "new_cases" else utils.COLORS["red"],
            height=380,
        )
    )

monthly = monthly_summary(data, [daily_col])
recent_months = monthly.sort_values("month").tail(24) if not monthly.empty else monthly
left, right = st.columns([1.45, 1], gap="medium")
with left:
    with st.container(border=True):
        st.markdown(
            '<div class="section-eyebrow">MONTHLY LOAD</div>'
            f'<div class="section-title">{utils.esc(daily_label)} summed by month</div>',
            unsafe_allow_html=True,
        )
        if recent_months.empty:
            utils.empty_state("No monthly data", "The selected window does not contain monthly observations.")
        else:
            utils.plotly(
                monthly_bar_chart(
                    recent_months,
                    value_col=daily_col,
                    label=daily_label,
                    color=utils.COLORS["blue"] if daily_col == "new_cases" else utils.COLORS["red"],
                    height=340,
                )
            )
            st.caption("Last 24 months in the window. Monthly totals can still reflect reporting backlogs.")
with right:
    with st.container(border=True):
        st.markdown(
            '<div class="section-eyebrow">PEAKS</div>'
            '<div class="section-title">Separated local maxima</div>',
            unsafe_allow_html=True,
        )
        peaks = detect_peaks(data.set_index("date")[daily_col], window=14, top_n=6)
        if peaks.empty:
            utils.empty_state("No separated peaks", "The series has no positive local maxima under the current rule.")
        else:
            lookup = data.set_index("date")[rolling_col]
            peaks_view = pd.DataFrame(
                {
                    "Peak date": peaks["date"].map(utils.format_date),
                    "Daily value": peaks["value"].map(utils.format_compact),
                    "7-day average": peaks["date"].map(lambda stamp: utils.format_compact(lookup.get(stamp))),
                }
            )
            st.dataframe(peaks_view, width="stretch", hide_index=True, height=260)
            st.caption("A peak must be the maximum in a centred 29-day window and at least 5% of the series maximum; nearby candidates are suppressed.")

# --------------------------------------------------------------------------- #
# Seasonality
# --------------------------------------------------------------------------- #
utils.section_header(
    "SEASONALITY",
    "Monthly heatmap and the recent period comparison",
    "The heatmap shows within-year monthly totals. The comparison uses equal-length windows ending at the selected end date.",
    note="Descriptive, not a forecast",
)
heat_left, heat_right = st.columns([1.45, 1], gap="medium")
with heat_left:
    with st.container(border=True):
        matrix = monthly_heatmap_matrix(data, daily_col)
        if matrix.empty:
            utils.empty_state("No heatmap data", "Monthly aggregation produced no rows for this series.")
        else:
            utils.plotly(monthly_heatmap(matrix, label=daily_label, height=360))
            st.caption("Years with no observations are blank. Values are sums of reported daily counts.")
with heat_right:
    with st.container(border=True):
        period_days = min(30, max(1, span_days))
        recent = window_comparison(data, daily_col, end, days=period_days)
        st.markdown(
            '<div class="section-eyebrow">LAST WINDOW</div>'
            f'<div class="section-title">Last {period_days} days vs preceding {period_days}</div>',
            unsafe_allow_html=True,
        )
        utils.render_kpis(
            [
                utils.kpi_card(
                    "CURRENT WINDOW",
                    utils.format_compact(recent["current"], 1),
                    sub=f"{daily_label} summed",
                    tone="cyan",
                    icon_name="activity",
                    chip="REPORTED",
                ),
                utils.kpi_card(
                    "PRECEDING WINDOW",
                    utils.format_compact(recent["previous"], 1),
                    sub="Equal length",
                    tone="violet",
                    icon_name="clock",
                    chip="REPORTED",
                ),
                utils.kpi_card(
                    "CHANGE",
                    utils.format_signed_pct(recent["change_pct"]) if recent["change_pct"] is not None else "—",
                    sub="Percentage change" if recent["change_pct"] is not None else "Missing or zero baseline",
                    tone="amber",
                    icon_name="trend_up",
                    chip="NOT CAUSAL",
                ),
            ]
        )
        st.caption("The preceding window may fall outside the selected filter when the window is short.")

# --------------------------------------------------------------------------- #
# Distribution
# --------------------------------------------------------------------------- #
utils.section_header(
    "DISTRIBUTION",
    "How daily reporting is distributed",
    "Daily values are strongly skewed and contain reporting artefacts. Distributions are shown rather than reduced to a single average.",
    note="Raw daily values",
)
dist_left, dist_right = st.columns([1, 1], gap="medium")
with dist_left:
    with st.container(border=True):
        st.markdown(
            '<div class="section-eyebrow">HISTOGRAM</div>'
            f'<div class="section-title">{utils.esc(daily_label)} distribution</div>',
            unsafe_allow_html=True,
        )
        utils.pyplot(
            distribution_plot(
                data[daily_col],
                label=daily_label,
                log_scale=False,
                color=utils.COLORS["cyan"] if daily_col == "new_cases" else utils.COLORS["red"],
            )
        )
        st.caption("The median is marked. Negative revisions are included if present in the selected series.")
with dist_right:
    with st.container(border=True):
        st.markdown(
            '<div class="section-eyebrow">BY YEAR</div>'
            '<div class="section-title">Within-year variation</div>',
            unsafe_allow_html=True,
        )
        year_data = data.assign(year=data["date"].dt.year)
        utils.pyplot(
            box_plot(
                year_data,
                x="year",
                y=daily_col,
                label=daily_label,
                log_scale=False,
            )
        )
        st.caption("Boxes show the interquartile range; points are outliers. Reporting practices differ by year.")

# --------------------------------------------------------------------------- #
# Insights
# --------------------------------------------------------------------------- #
utils.section_header(
    "OBSERVATIONS",
    "What this series supports",
    "Rule-based observations from the selected entity, metric and window. Each card states its metric and date basis.",
    note="No hard-coded statistics",
)
utils.insight_grid(
    timeseries_insights(
        data,
        country=entity,
        daily_col=daily_col,
        rolling_col=rolling_col,
        daily_label=daily_label,
        metric_col=cumulative_col,
        metric_label=cumulative_label,
        start=start,
        end=end,
    )
)
utils.pills(["Raw daily reports", "7-day average", "Separated peaks", "Monthly aggregation", "No forecast"])

utils.source_strip(
    source="Our World in Data — COVID-19 Dataset",
    url=OWID_URL,
    latest=utils.format_date(df["date"].max()),
)
st.caption(
    "This page describes a reported time series. Peak detection and trend comparisons are deterministic rules, not "
    "statistical models, and they do not establish causation or predict future values."
)
