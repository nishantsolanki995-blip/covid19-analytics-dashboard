"""Exploratory Data Analysis — schema, quality, distributions and relationships."""
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
    correlation_matrix,
    coverage_summary,
    describe_metrics,
    filter_by_date,
    iqr_outliers,
    latest_snapshot,
)
from src.data_loader import OWID_URL, ensure_quality_report, load_owid_data
from src.visualizations import (
    box_plot,
    correlation_plot,
    distribution_plot,
    scatter_plot,
    simple_bar,
    style_matplotlib,
)


utils.page_setup("Exploratory Data Analysis")
style_matplotlib()

try:
    with st.spinner("Loading the historical panel…"):
        df = load_owid_data()
except Exception as exc:  # pragma: no cover - UI guard
    utils.show_data_error(str(exc), detail="python -m src.data_loader --build")
    st.stop()

with st.spinner("Reading the data-quality report…"):
    quality = ensure_quality_report() or {}

utils.sidebar_intro(
    latest=utils.format_date(df["date"].max()),
    rows=f"{len(df):,}",
    countries=str(df.loc[df["is_country"].astype(bool), "country"].nunique()),
    coverage=f"{utils.format_date(df['date'].min())} →",
)
utils.sidebar_label("Analysis window")
start, end = utils.date_range_filter(df, key="analysis_date_range")
st.sidebar.caption(utils.date_window_caption(start, end))
utils.sidebar_note(
    "EDA is descriptive. Missing values are never silently converted to zero, and correlation is not treated as "
    "causation. The quality report describes the full source file; the charts below use the selected window."
)

filtered = filter_by_date(df, start, end)
if filtered.empty:
    utils.page_header(
        "EXPLORATORY DATA ANALYSIS",
        "What is in the panel, and what is missing",
        "No observations fall inside the selected window. Widen the analysis window in the sidebar.",
        badge_label="FILTERED VIEW",
        badge_value="No records",
    )
    st.stop()

snapshot = latest_snapshot(filtered)
if snapshot.empty:
    utils.page_header(
        "EXPLORATORY DATA ANALYSIS",
        "What is in the panel, and what is missing",
        "The window has no country with a usable cumulative case total.",
        badge_label="EMPTY SNAPSHOT",
        badge_value="No records",
    )
    st.stop()

coverage = coverage_summary(filtered.loc[filtered["is_country"].astype(bool)], "country")
owid_q = quality.get("owid", {})
india_q = quality.get("india", {})

utils.page_header(
    "EXPLORATORY DATA ANALYSIS",
    "What is in the panel, and what is missing",
    "A transparent audit of schema, missingness, distributions, outliers and ecological relationships before any interpretation.",
    accent="before the charts",
    badge_label="QUALITY AUDIT",
    badge_value=f"Through {utils.format_date(coverage.get('date_max'))}",
    badge_foot=f"{coverage.get('entities', 0)} countries · {quality.get('generated_at', 'report unavailable')[:10] if quality.get('generated_at') else 'processed panel'}",
)

# --------------------------------------------------------------------------- #
# Quality scorecards
# --------------------------------------------------------------------------- #
utils.section_header(
    "DATA QUALITY",
    "The shape of the source and the processed panel",
    "The full raw OWID file is described first; the filtered window drives the charts later on this page.",
    note="Missing ≠ zero",
)
quality_cards = [
    utils.kpi_card(
        "ROWS IN WINDOW",
        f"{len(filtered):,}",
        sub=f"of {int(owid_q.get('processed_rows') or len(df)):,} processed rows",
        tone="blue",
        icon_name="table",
        chip="FILTERED",
    ),
    utils.kpi_card(
        "COUNTRIES IN WINDOW",
        f"{coverage.get('entities', 0):,}",
        sub="Countries with at least one row",
        tone="cyan",
        icon_name="globe",
        chip="COVERAGE",
    ),
    utils.kpi_card(
        "RAW MISSING CELLS",
        f"{float(owid_q.get('missing_pct') or 0):.1f}%",
        sub=f"{int(owid_q.get('missing_cells') or 0):,} of {int(owid_q.get('rows') or 0) * int(owid_q.get('columns') or 0):,} cells",
        tone="amber",
        icon_name="alert",
        chip="SCHEMA-WIDE",
    ),
    utils.kpi_card(
        "DUPLICATE ROWS",
        f"{int(owid_q.get('duplicate_rows') or 0):,}",
        sub="Country + date duplicates in raw file",
        tone="violet",
        icon_name="layers",
        chip="CHECKED",
    ),
    utils.kpi_card(
        "NEGATIVE DAILY VALUES",
        f"{int(owid_q.get('negative_daily_values') or 0):,}",
        sub="Retained and flagged as revisions",
        tone="red",
        icon_name="trend_down",
        chip="NOT HIDDEN",
    ),
    utils.kpi_card(
        "PROCESSED MISSING",
        f"{float(owid_q.get('processed_missing_pct') or 0):.1f}%",
        sub="After the compact column selection",
        tone="green",
        icon_name="check",
        chip="PROCESSED",
    ),
]
utils.render_kpis(quality_cards)

utils.callout(
    "Missing values are shown as missing, never as zero. The OWID compact file is wide and sparse: testing, "
    "vaccination and contextual indicators have different reporting tails. A high missing share is expected and does "
    "not mean the case series is wrong. The India archive has its own schema and a much shorter reporting period.",
    title="Read missingness before reading the numbers",
    tone="amber",
    icon_name="info",
)

# --------------------------------------------------------------------------- #
# Tabs: global and India
# --------------------------------------------------------------------------- #
tab_global, tab_india = st.tabs(["Global panel · OWID", "India archive · covid19india"])

with tab_global:
    left, right = st.columns([1.05, 1], gap="medium")
    with left:
        with st.container(border=True):
            st.markdown(
                '<div class="section-eyebrow">SCHEMA</div>'
                '<div class="section-title">Columns, coverage and missingness</div>',
                unsafe_allow_html=True,
            )
            schema = pd.DataFrame(owid_q.get("schema", []))
            if schema.empty:
                utils.empty_state("Schema report unavailable", "Rebuild the quality report with the data-loader CLI.")
            else:
                schema_view = pd.DataFrame(
                    {
                        "Column": schema["column"],
                        "Type": schema["dtype"],
                        "Non-null": schema["non_null"].map(utils.format_int),
                        "Missing": schema["missing"].map(utils.format_int),
                        "Missing %": schema["missing_pct"].map(lambda value: f"{float(value):.1f}%"),
                        "Unique": schema["unique"].map(utils.format_int),
                    }
                )
                st.dataframe(schema_view, width="stretch", hide_index=True, height=320)
            st.caption("Raw OWID columns. The processed panel persists a documented subset used by the dashboard.")
    with right:
        with st.container(border=True):
            st.markdown(
                '<div class="section-eyebrow">MISSINGNESS</div>'
                '<div class="section-title">The sparsest columns in the raw file</div>',
                unsafe_allow_html=True,
            )
            if schema.empty:
                utils.empty_state("Missingness report unavailable", "Rebuild the quality report to populate this chart.")
            else:
                sparse = schema.sort_values("missing_pct", ascending=True).tail(16)
                utils.plotly(
                    simple_bar(
                        [utils.truncate(column, 30) for column in sparse["column"]],
                        sparse["missing_pct"].tolist(),
                        label="Missing cells",
                        color=utils.COLORS["violet"],
                        percent=True,
                        height=430,
                    )
                )
                st.caption("Share of all raw rows in which the column is blank. A value of 100% means no usable observations.")

    utils.section_header(
        "DISTRIBUTIONS",
        "Country-level burden is heavy-tailed and not normal",
        "The latest country snapshot inside the window is used so each country contributes one observation, not one per day.",
        note=f"{len(snapshot)} countries",
    )
    metric_options = {
        "Cases per million": ("total_cases_per_million", "Cases per million", True),
        "Deaths per million": ("total_deaths_per_million", "Deaths per million", True),
        "Case fatality ratio": ("case_fatality_ratio", "CFR (%)", False),
        "First-dose coverage": ("vaccination_rate", "At least one dose (%)", False),
        "GDP per capita": ("gdp_per_capita", "GDP per capita (international $)", True),
        "Median age": ("median_age", "Median age (years)", False),
    }
    metric_label = st.selectbox("Distribution metric", list(metric_options), key="eda_metric")
    metric_col, metric_title, metric_log = metric_options[metric_label]

    dist_left, dist_right = st.columns([1, 1], gap="medium")
    with dist_left:
        with st.container(border=True):
            st.markdown(
                f'<div class="section-eyebrow">HISTOGRAM</div>'
                f'<div class="section-title">{utils.esc(metric_title)} across countries</div>',
                unsafe_allow_html=True,
            )
            if metric_col not in snapshot.columns:
                utils.empty_state("Metric unavailable", "This metric is not present in the processed panel.")
            else:
                utils.pyplot(
                    distribution_plot(
                        snapshot[metric_col],
                        label=metric_title,
                        log_scale=metric_log,
                        color=utils.COLORS["cyan"],
                    )
                )
                st.caption("Median is marked. A log scale compresses the long right tail without discarding large values.")
    with dist_right:
        with st.container(border=True):
            st.markdown(
                f'<div class="section-eyebrow">BY CONTINENT</div>'
                f'<div class="section-title">{utils.esc(metric_title)} by region</div>',
                unsafe_allow_html=True,
            )
            if metric_col not in snapshot.columns:
                utils.empty_state("Metric unavailable", "This metric is not present in the processed panel.")
            else:
                utils.pyplot(
                    box_plot(
                        snapshot,
                        x="continent",
                        y=metric_col,
                        label=metric_title,
                        log_scale=metric_log,
                    )
                )
                st.caption("Boxes show the interquartile range; points are outliers. Regions have different country counts.")

    utils.section_header(
        "OUTLIERS & SUMMARY",
        "Extreme values are observations, not errors to delete",
        "The IQR rule flags values outside 1.5 × the interquartile range. Flags are shown, not removed.",
        note="Descriptive statistics",
    )
    outliers = iqr_outliers(snapshot[metric_col]) if metric_col in snapshot.columns else {"count": 0, "lower": float("nan"), "upper": float("nan"), "share_pct": 0.0}
    summary_cols = [
        "total_cases_per_million",
        "total_deaths_per_million",
        "case_fatality_ratio",
        "vaccination_rate",
        "median_age",
        "life_expectancy",
        "gdp_per_capita",
        "hospital_beds_per_thousand",
    ]
    summary = describe_metrics(snapshot, summary_cols)
    out_left, out_right = st.columns([1, 1.2], gap="medium")
    with out_left:
        with st.container(border=True):
            st.markdown(
                '<div class="section-eyebrow">IQR CHECK</div>'
                '<div class="section-title">Flagged values on the selected metric</div>',
                unsafe_allow_html=True,
            )
            utils.render_kpis(
                [
                    utils.kpi_card("FLAGGED COUNT", f"{outliers['count']:,}", sub=f"{outliers['share_pct']:.1f}% of countries", tone="amber", icon_name="alert", chip="IQR"),
                    utils.kpi_card("LOWER FENCE", utils.format_decimal(outliers["lower"], 2), sub=metric_title, tone="cyan", icon_name="trend_down", chip="Q1 − 1.5×IQR"),
                    utils.kpi_card("UPPER FENCE", utils.format_decimal(outliers["upper"], 2), sub=metric_title, tone="violet", icon_name="trend_up", chip="Q3 + 1.5×IQR"),
                ]
            )
            st.caption("For heavy-tailed counts, many countries can legitimately fall above the fence. The flag is a prompt to inspect, not evidence of a data error.")
    with out_right:
        with st.container(border=True):
            st.markdown(
                '<div class="section-eyebrow">SUMMARY</div>'
                '<div class="section-title">Distribution of structural indicators</div>',
                unsafe_allow_html=True,
            )
            if summary.empty:
                utils.empty_state("Summary unavailable", "No structural metrics are present in this window.")
            else:
                st.dataframe(summary.round(2), width="stretch", height=360)
                st.caption("Count is the number of non-missing country observations; quartiles use only those observations.")

    utils.section_header(
        "RELATIONSHIPS",
        "Correlation structure and an ecological scatter",
        "Spearman correlation is used because most indicators are skewed. Association is not causation and country-level patterns need not hold for individuals.",
        note="Spearman · country snapshot",
    )
    corr_cols = [
        "total_cases_per_million",
        "total_deaths_per_million",
        "case_fatality_ratio",
        "vaccination_rate",
        "population_density",
        "median_age",
        "life_expectancy",
        "gdp_per_capita",
        "hospital_beds_per_thousand",
        "human_development_index",
    ]
    corr = correlation_matrix(snapshot, corr_cols, method="spearman", min_periods=20)
    corr_left, corr_right = st.columns([1, 1.25], gap="medium")
    with corr_left:
        with st.container(border=True):
            st.markdown(
                '<div class="section-eyebrow">CORRELATION</div>'
                '<div class="section-title">Structural indicators</div>',
                unsafe_allow_html=True,
            )
            if corr.empty or corr.isna().all().all():
                utils.empty_state("Not enough paired observations", "At least two structural metrics with 20 paired countries are required.")
            else:
                utils.pyplot(correlation_plot(corr, title="Spearman correlation"))
                st.caption("Pairwise complete observations differ by pair, so the matrix is descriptive rather than a single model.")
    with corr_right:
        with st.container(border=True):
            st.markdown(
                '<div class="section-eyebrow">SCATTER</div>'
                '<div class="section-title">GDP per capita vs reported cases per million</div>',
                unsafe_allow_html=True,
            )
            if "gdp_per_capita" not in snapshot.columns or "total_cases_per_million" not in snapshot.columns:
                utils.empty_state("Scatter unavailable", "The required metrics are not present.")
            else:
                utils.plotly(
                    scatter_plot(
                        snapshot,
                        x="gdp_per_capita",
                        y="total_cases_per_million",
                        x_label="GDP per capita (log scale)",
                        y_label="Cases per million (log scale)",
                        height=440,
                    )
                )
                st.caption("Marker size is population; colour is continent. Both axes are logarithmic. This is an ecological comparison, not an individual-level result.")

with tab_india:
    st.markdown(
        '<div class="section-sub">The India archive is a separate source with a different schema, a shorter period and '
        'recoveries that OWID does not provide consistently. Its quality is reported here rather than mixed into the global panel.</div>',
        unsafe_allow_html=True,
    )
    if not india_q:
        utils.empty_state("India quality report unavailable", "Run python -m src.data_loader --build to generate the combined report.")
    else:
        india_cards = [
            utils.kpi_card("RAW ROWS", f"{int(india_q.get('rows') or 0):,}", sub="covid19india state archive", tone="blue", icon_name="table", chip="SOURCE"),
            utils.kpi_card("STATES / UTS", f"{int(india_q.get('states') or 0):,}", sub="Real state/UT entities", tone="cyan", icon_name="map", chip="COVERAGE"),
            utils.kpi_card("DUPLICATE ROWS", f"{int(india_q.get('duplicate_rows') or 0):,}", sub="State + date duplicates", tone="violet", icon_name="layers", chip="CHECKED"),
            utils.kpi_card("NEGATIVE DAILY", f"{int(india_q.get('negative_daily_rows') or 0):,}", sub="Flagged revisions", tone="red", icon_name="trend_down", chip="RETAINED"),
            utils.kpi_card("PROCESSED ROWS", f"{int(india_q.get('processed_rows') or 0):,}", sub="Complete daily grid", tone="green", icon_name="check", chip="DERIVED"),
            utils.kpi_card("ARCHIVE END", utils.format_date(india_q.get("date_max")), sub="Historical public CSV", tone="amber", icon_name="calendar", chip="NOT LIVE"),
        ]
        utils.render_kpis(india_cards)
        schema = pd.DataFrame(india_q.get("schema", []))
        if not schema.empty:
            schema_view = pd.DataFrame(
                {
                    "Column": schema["column"],
                    "Type": schema["dtype"],
                    "Non-null": schema["non_null"].map(utils.format_int),
                    "Missing": schema["missing"].map(utils.format_int),
                    "Missing %": schema["missing_pct"].map(lambda value: f"{float(value):.1f}%"),
                    "Unique": schema["unique"].map(utils.format_int),
                }
            )
            st.dataframe(schema_view, width="stretch", hide_index=True, height=240)
        st.caption(india_q.get("note", ""))

utils.source_strip(
    source="Our World in Data — COVID-19 Dataset",
    url=OWID_URL,
    secondary="covid19india.org archive",
    secondary_url="https://api.covid19india.org/csv/latest/states.csv",
    latest=utils.format_date(coverage.get("date_max")),
)
st.caption(
    "All summaries are computed inside the selected window unless labelled as a full-file quality metric. "
    "Missing data is never imputed for headline statistics, and outliers are reported rather than deleted."
)
