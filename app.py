"""COVID-19 Pandemic Trends & Comparative Data Analytics Dashboard.

Entry point and landing page.  Uses Streamlit's native multi-page navigation
(the numbered files in ``pages/``) and a shared design system from ``src``.
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from src import utils
from src.analysis import (
    global_totals,
    latest_snapshot,
    latest_world,
    rank_entities,
    vaccination_coverage,
    world_frame,
)
from src.data_loader import dataset_status, load_owid_data
from src.visualizations import daily_chart, ranking_bar


utils.page_setup("Overview")

# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
df: pd.DataFrame | None = None
load_error: str | None = None
try:
    with st.spinner("Loading the historical COVID-19 panel…"):
        df = load_owid_data()
except Exception as exc:  # pragma: no cover - UI guard
    load_error = str(exc)

status = dataset_status()
latest_label = utils.format_date(status.get("owid_date_max"))

if df is not None:
    world = world_frame(df)
    snapshot = latest_snapshot(df)
    totals = global_totals(df)
    latest_row = latest_world(df)
    vax = vaccination_coverage(snapshot)
    latest_date = totals.get("date") or status.get("owid_date_max")
    utils.sidebar_intro(
        latest=utils.format_date(latest_date),
        rows=f"{len(df):,}",
        countries=str(totals.get("countries_reporting", "—")),
        coverage="2020 → latest",
    )
else:
    utils.sidebar_intro(
        latest=latest_label or "not loaded",
        coverage="Data files unavailable",
    )

utils.sidebar_label("About this build")
utils.sidebar_note(
    "MCA Semester III Data Science project.<br><br>"
    "All figures are <b>reported historical values</b>. The dashboard is not a live feed. "
    "Use the pages above to move between analytical workspaces."
)

# --------------------------------------------------------------------------- #
# Hero
# --------------------------------------------------------------------------- #
if df is not None:
    confirmed = totals.get("confirmed")
    deaths = totals.get("deaths")
    cfr = totals.get("cfr")
    hero_value = utils.format_compact(confirmed, 2)
    hero_deaths = utils.format_compact(deaths, 2)
    hero_cfr = utils.format_pct(cfr, 2)
    hero_countries = str(totals.get("countries_reporting", "—"))
    latest_series = world[world["date"] <= pd.Timestamp(latest_date)] if not world.empty else world
else:
    hero_value = "Data unavailable"
    hero_deaths = "—"
    hero_cfr = "—"
    hero_countries = "—"
    latest_series = pd.DataFrame()

st.markdown(
    f"""
    <div class="hero">
      <div class="hero-grid">
        <div>
          <div class="eyebrow"><span class="eyebrow-dot"></span>MCA DATA SCIENCE · SEMESTER III</div>
          <h1>COVID-19 Pandemic<br><span class="accent">Analytics Dashboard</span></h1>
          <p>
            An interactive historical analysis platform for global and Indian pandemic trends.
            Explore reported cases, mortality, per-capita burden, vaccination coverage and
            state-level patterns through cleaned data, transparent methods and rule-based insights.
          </p>
          <div class="hero-meta">
            <span>Our World in Data</span>
            <span>covid19india archive</span>
            <span>Python · Streamlit · Plotly</span>
            <span>Historical, not live</span>
          </div>
        </div>
        <div class="hero-panel">
          <div class="hp-label">LATEST GLOBAL TOTAL</div>
          <div class="hp-value">{utils.esc(hero_value)}</div>
          <div class="hp-sub">confirmed cases · through {utils.esc(utils.format_date(latest_date))}</div>
          <div class="hp-rule"></div>
          <div class="hp-label">REPORTED DEATHS · CFR</div>
          <div class="hp-value" style="font-size:1.35rem">{utils.esc(hero_deaths)} <span style="color:#94A3B8;font-size:0.8rem;font-weight:500">· {utils.esc(hero_cfr)}</span></div>
          <div class="hp-sub">{utils.esc(hero_countries)} countries with cumulative totals</div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if load_error:
    utils.callout(
        "The historical files could not be loaded. Run <code>python -m src.data_loader --build</code> "
        "from the project folder, or place the OWID CSV in <code>data/raw/</code>. "
        "The navigation below remains available; data pages will show their own instructions.",
        title="Dataset preparation required",
        tone="amber",
        icon_name="alert",
    )
else:
    cards = [
        utils.kpi_card(
            "TOTAL CONFIRMED",
            utils.format_compact(confirmed, 2),
            sub=f"{utils.format_int(confirmed)} reported cases",
            tone="blue",
            icon_name="activity",
            chip=f"WORLD · {utils.format_date(latest_date, '%b %Y').upper()}",
        ),
        utils.kpi_card(
            "TOTAL DEATHS",
            utils.format_compact(deaths, 2),
            sub=f"{utils.format_int(deaths)} reported deaths",
            tone="red",
            icon_name="shield",
            chip="REPORTED",
        ),
        utils.kpi_card(
            "CASE FATALITY RATIO",
            utils.format_pct(cfr, 2),
            sub="Reported deaths ÷ reported cases",
            tone="amber",
            icon_name="percent",
            chip="NOT IFR",
        ),
        utils.kpi_card(
            "FIRST-DOSE COVERAGE",
            utils.format_pct(vax["weighted"], 1) if pd.notna(vax["weighted"]) else "—",
            sub=(
                f"Population-weighted · {vax['countries']} countries"
                if vax["countries"]
                else "Vaccination data not available"
            ),
            tone="green",
            icon_name="syringe",
            chip=(
                f"AS OF {utils.format_date(vax['as_of'], '%b %Y').upper()}"
                if vax["as_of"] is not None and pd.notna(vax["as_of"])
                else "HISTORICAL"
            ),
        ),
        utils.kpi_card(
            "COUNTRIES REPORTING",
            f"{totals.get('countries_reporting', 0):,}",
            sub="Latest cumulative totals available",
            tone="cyan",
            icon_name="globe",
            chip="COVERAGE",
        ),
    ]
    utils.render_kpis(cards)

# --------------------------------------------------------------------------- #
# Analytical preview
# --------------------------------------------------------------------------- #
if df is not None and not latest_series.empty:
    utils.section_header(
        "GLOBAL PULSE",
        "The pandemic in two views",
        "Reported daily activity and the countries carrying the largest cumulative caseloads.",
        note="Historical OWID panel",
    )
    left, right = st.columns([1.55, 1], gap="medium")
    with left:
        with st.container(border=True):
            st.markdown(
                '<div class="section-eyebrow">DAILY REPORTING</div>'
                '<div class="section-title">Global new cases and 7-day average</div>',
                unsafe_allow_html=True,
            )
            utils.plotly(
                daily_chart(
                    latest_series,
                    daily_col="new_cases",
                    rolling_col="new_cases_7d",
                    label="New cases",
                    height=300,
                )
            )
            st.caption("Raw daily reports include batch releases; the white line is the 7-day average.")
    with right:
        with st.container(border=True):
            st.markdown(
                '<div class="section-eyebrow">CUMULATIVE BURDEN</div>'
                '<div class="section-title">Top 10 countries by confirmed cases</div>',
                unsafe_allow_html=True,
            )
            top = rank_entities(snapshot, "total_cases", n=10)
            utils.plotly(
                ranking_bar(
                    top,
                    metric="total_cases",
                    label="Confirmed cases",
                    n=10,
                    height=300,
                    colorscale="blue",
                )
            )

# --------------------------------------------------------------------------- #
# Workspaces
# --------------------------------------------------------------------------- #
utils.section_header(
    "WORKSPACES",
    "Seven connected analytical views",
    "Each page shares the same cleaned data, typography and metric definitions. Date filters persist between analytical pages.",
    note="Use the sidebar or the links below",
)

workspaces = [
    (
        "Global Overview",
        "pages/1_Global_Overview.py",
        "globe",
        "cyan",
        "World totals, a per-capita choropleth, top-10 rankings and a dated analytical summary.",
    ),
    (
        "India Analysis",
        "pages/2_India_Analysis.py",
        "map_pin",
        "amber",
        "National waves, state map, state rankings and an explicit account of archive limitations.",
    ),
    (
        "Exploratory Data Analysis",
        "pages/3_EDA.py",
        "database",
        "violet",
        "Schema, missingness, data-quality checks, distributions, outliers and correlation structure.",
    ),
    (
        "Comparative Analytics",
        "pages/4_Comparative_Analytics.py",
        "compare",
        "blue",
        "Multi-country trajectories, rankings and population-adjusted comparisons over a shared window.",
    ),
    (
        "Time Series & Insights",
        "pages/5_Time_Series_Insights.py",
        "clock",
        "green",
        "Peak detection, monthly heatmaps, period comparisons and traceable rule-based observations.",
    ),
    (
        "Data Explorer",
        "pages/6_Data_Explorer.py",
        "table",
        "cyan",
        "Search, filter, sort, paginate and export exactly the rows currently on screen.",
    ),
    (
        "Project Information",
        "pages/7_Project_Information.py",
        "file",
        "violet",
        "Problem statement, methodology, sources, limitations and future scope for academic evaluation.",
    ),
]

for start in range(0, len(workspaces), 3):
    columns = st.columns(3, gap="small")
    for column, workspace in zip(columns, workspaces[start : start + 3]):
        title, path, icon_name, tone, description = workspace
        with column:
            with st.container(border=True):
                st.markdown(
                    f"""
                    <div class="feature tone-{utils.esc(tone)}">
                      <div class="feature-icon">{utils.icon(icon_name, 15)}</div>
                      <div class="feature-title">{utils.esc(title)}</div>
                      <div class="feature-desc">{description}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.page_link(path, label=f"Open {title}", width="stretch")

# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #
if df is not None:
    utils.section_header(
        "DATA PROVENANCE",
        "What is behind the numbers",
        "The dashboard reads local CSV archives, validates the schema and builds a separate processed dataset.",
    )
    info_left, info_right = st.columns([1.15, 1], gap="medium")
    with info_left:
        with st.container(border=True):
            st.markdown(
                f"""
                <div class="doc-card" style="border:0;background:transparent;padding:0">
                  <h4>Our World in Data — country panel</h4>
                  <p>
                    Country/date records for confirmed cases, deaths, testing, vaccination and
                    contextual indicators. The raw file is approximately 180 MB; this project ships
                    a compact processed derivative and rebuilds it automatically when the raw file changes.
                  </p>
                  <p style="margin-top:8px">
                    <b>Coverage:</b> {utils.esc(utils.format_date(status.get('owid_date_min')))} →
                    {utils.esc(utils.format_date(status.get('owid_date_max')))} ·
                    <b>Rows:</b> {int(status.get('owid_rows') or 0):,} ·
                    <b>Countries:</b> {status.get('owid_countries') or '—'}
                  </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<a href="{utils.esc(status["owid_source"]["url"])}" target="_blank" rel="noopener">Open the OWID source CSV ↗</a>',
                unsafe_allow_html=True,
            )
    with info_right:
        with st.container(border=True):
            st.markdown(
                f"""
                <div class="doc-card" style="border:0;background:transparent;padding:0">
                  <h4>covid19india.org — state archive</h4>
                  <p>
                    State/UT cumulative confirmed, recovered and deceased counts. This is the
                    historical public archive used for the India page; it ends on
                    <b>1 June 2021</b> and is not presented as current data.
                  </p>
                  <p style="margin-top:8px">
                    <b>Coverage:</b> 30 Jan 2020 → 1 Jun 2021 ·
                    <b>States/UTs:</b> {status.get('india_states') or '—'} ·
                    <b>Status:</b> historical archive
                  </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<a href="{utils.esc(status["india_source"]["url"])}" target="_blank" rel="noopener">Open the covid19india archive ↗</a>',
                unsafe_allow_html=True,
            )

utils.source_strip(
    source=status["owid_source"]["name"],
    url=status["owid_source"]["url"],
    secondary="covid19india.org archive",
    secondary_url=status["india_source"]["url"],
    latest=utils.format_date(status.get("owid_date_max")),
)
st.caption(
    "Unless stated otherwise, all values are cumulative or daily reported figures from the selected source. "
    "Missing data is never replaced with zero for headline KPIs."
)
