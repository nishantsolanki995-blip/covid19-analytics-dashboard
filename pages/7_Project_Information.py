"""Project Information — brief, methodology, sources, limitations and reproduction."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from src import utils
from src.data_loader import dataset_status


utils.page_setup("Project Information")
status = dataset_status()
owid = status.get("owid_source", {})
india = status.get("india_source", {})
quality = status.get("quality", {}) or {}
owid_q = quality.get("owid", {}) or {}
india_q = quality.get("india", {}) or {}

utils.sidebar_intro(
    latest=utils.format_date(status.get("owid_date_max")),
    rows=f"{int(status.get('owid_rows') or 0):,}" if status.get("owid_rows") else "processed panel",
    countries=str(status.get("owid_countries") or "—"),
    coverage="OWID + covid19india",
)
utils.sidebar_label("About this build")
utils.sidebar_note(
    "MCA Semester III Data Science project. The dashboard uses archived historical data and is explicitly "
    "not a live feed. Recoveries and active cases are only shown where the source supports them."
)

utils.page_header(
    "PROJECT INFORMATION",
    "A transparent analytical build, not a black box",
    "Problem statement, objectives, method, sources, metric definitions, limitations and reproduction steps for academic review.",
    accent="method & provenance",
    badge_label="MCA · SEMESTER III",
    badge_value="Data Science project",
    badge_foot="Python · Streamlit · Plotly · Pandas",
)

cards = [
    utils.kpi_card("INDEPENDENT SOURCES", "2", sub="OWID country panel + India state archive", tone="blue", icon_name="database", chip="PROVENANCE"),
    utils.kpi_card("COUNTRIES", f"{int(status.get('owid_countries') or 0):,}", sub="OWID country entities in the panel", tone="cyan", icon_name="globe", chip="GLOBAL"),
    utils.kpi_card("STATES / UTS", f"{int(status.get('india_states') or 0):,}", sub="covid19india state entities", tone="amber", icon_name="map_pin", chip="INDIA"),
    utils.kpi_card("OWID THROUGH", utils.format_date(status.get("owid_date_max")), sub="Latest raw observation", tone="violet", icon_name="calendar", chip="HISTORICAL"),
    utils.kpi_card("INDIA THROUGH", utils.format_date(status.get("india_date_max")), sub="Public archive endpoint", tone="red", icon_name="clock", chip="NOT LIVE"),
    utils.kpi_card(
        "PROCESSED ROWS",
        f"{int((owid_q.get('processed_rows') or 0) + (india_q.get('processed_rows') or 0)):,}",
        sub="Both processed panels combined",
        tone="green",
        icon_name="table",
        chip="DERIVED",
    ),
]
utils.render_kpis(cards)

# --------------------------------------------------------------------------- #
# Brief
# --------------------------------------------------------------------------- #
utils.section_header(
    "BRIEF",
    "Problem statement and objectives",
    "The project answers descriptive questions about a reported pandemic using two independent historical archives.",
    note="MCA Semester III",
)
brief_left, brief_right = st.columns([1.35, 1], gap="medium")
with brief_left:
    with st.container(border=True):
        st.markdown(
            """
            <div class="doc-card">
              <h4>Problem statement</h4>
              <p>
                COVID-19 data is abundant but uneven. Confirmed cases, deaths, recoveries and vaccination
                coverage arrive at different cadences, cover different periods and are revised retrospectively.
                A dashboard that hides those differences can produce confident-looking but misleading conclusions.
                This project builds a reproducible analytical workspace that keeps the reported history, the
                transformation steps and the interpretation limits visible together.
              </p>
              <h4 style="margin-top:14px">Objectives</h4>
              <ul>
                <li>Acquire and validate two independent historical sources without presenting them as live data.</li>
                <li>Clean and document schema, duplicates, missingness and negative reporting revisions.</li>
                <li>Analyse global and Indian trends, rankings, per-capita burden and vaccination coverage.</li>
                <li>Compare countries over a shared window and at a shared epidemic threshold.</li>
                <li>Provide EDA, time-series insights, a filterable explorer and exportable rows for review.</li>
                <li>Make every headline number traceable to a metric definition and an observation date.</li>
              </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
with brief_right:
    with st.container(border=True):
        st.markdown(
            """
            <div class="doc-card">
              <h4>In scope</h4>
              <ul>
                <li>Historical global country/date panel (OWID).</li>
                <li>Historical Indian state/UT archive through 1 June 2021.</li>
                <li>Cases, deaths, CFR, per-capita rates, recoveries/active (India only), vaccination, testing context.</li>
                <li>Descriptive analytics, EDA, period comparison and rule-based observations.</li>
                <li>CSV export of the visible explorer table.</li>
              </ul>
              <h4 style="margin-top:14px">Out of scope</h4>
              <ul>
                <li>Live feeds, forecasting and causal inference.</li>
                <li>Global recovery/active estimates where no reliable source exists.</li>
                <li>Individual-level or clinical risk prediction.</li>
                <li>State-level per-million comparisons where populations are unavailable.</li>
              </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

utils.callout(
    "The dashboard follows a simple rule: if the source does not support a metric, the metric is omitted or clearly "
    "labelled as derived. Global recoveries and active cases are omitted because OWID does not publish a consistent "
    "recovery series. Indian active cases are derived from three reported cumulative series and clipped at zero.",
    title="The central design rule",
    tone="blue",
    icon_name="info",
)

# --------------------------------------------------------------------------- #
# Method
# --------------------------------------------------------------------------- #
utils.section_header(
    "METHOD",
    "From raw archive to a dated observation",
    "Every page reads the same processed panel. The pipeline is deterministic, logged and rebuildable from the raw files.",
    note="Acquire → validate → clean → derive → analyse → communicate",
)
timeline, method_side = st.columns([1.3, 1], gap="medium")
with timeline:
    with st.container(border=True):
        st.markdown(
            """
            <div class="timeline">
              <div class="timeline-item">
                <div class="t-step">01 · Acquire</div>
                <div class="t-title">Download or use local archives</div>
                <div class="t-body">OWID compact CSV, covid19india states CSV and an India state GeoJSON. Files are checked for size and schema before use.</div>
              </div>
              <div class="timeline-item">
                <div class="t-step">02 · Validate</div>
                <div class="t-title">Check columns, dates and types</div>
                <div class="t-body">Required columns are asserted, dates parsed, numbers coerced, duplicate country/date or state/date rows counted.</div>
              </div>
              <div class="timeline-item">
                <div class="t-step">03 · Clean</div>
                <div class="t-title">Standardise entities and fill only what is safe</div>
                <div class="t-body">State spellings are normalised. India gets a complete daily grid with cumulative forward-fill. Missing values are not replaced with zero for headline KPIs.</div>
              </div>
              <div class="timeline-item">
                <div class="t-step">04 · Derive</div>
                <div class="t-title">Calculate documented fields</div>
                <div class="t-body">Daily differences, 7-day averages, CFR, recovery ratio and Indian active cases. Negative daily values remain visible as revisions.</div>
              </div>
              <div class="timeline-item">
                <div class="t-step">05 · Analyse</div>
                <div class="t-title">Use one consistent analytical vocabulary</div>
                <div class="t-body">Latest snapshots, rankings, period comparisons, peaks, monthly aggregation, correlations and coverage summaries are shared by the pages.</div>
              </div>
              <div class="timeline-item">
                <div class="t-step">06 · Communicate</div>
                <div class="t-title">State the basis, date and limits</div>
                <div class="t-body">Each insight card records its metric and date basis. The UI distinguishes historical archives from live feeds and correlation from causation.</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
with method_side:
    with st.container(border=True):
        st.markdown(
            """
            <div class="doc-card">
              <h4>Metric definitions</h4>
              <ul>
                <li><b>Confirmed cases:</b> cumulative reported cases.</li>
                <li><b>Deaths:</b> cumulative reported deaths.</li>
                <li><b>CFR:</b> reported deaths ÷ reported confirmed cases × 100. Not an infection fatality ratio.</li>
                <li><b>Cases/deaths per million:</b> cumulative count ÷ population × 1,000,000.</li>
                <li><b>7-day average:</b> OWID smoothed series where supplied, else a trailing mean.</li>
                <li><b>Active cases (India):</b> confirmed − recovered − deaths, clipped at zero.</li>
                <li><b>Recovery ratio (India):</b> reported recoveries ÷ confirmed × 100.</li>
                <li><b>Vaccination:</b> people with at least one dose per hundred population.</li>
              </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
        utils.callout(
            "Population-adjusted rates depend on the population denominator supplied by OWID. They are more comparable "
            "than raw totals but still reflect testing and reporting differences.",
            title="Rates are not truth",
            tone="amber",
            icon_name="percent",
        )

# --------------------------------------------------------------------------- #
# Sources
# --------------------------------------------------------------------------- #
utils.section_header(
    "SOURCES",
    "Two archives, two reporting tails",
    "The global and Indian views are deliberately separate because their schemas, coverage and end dates differ.",
    note="Historical downloads",
)
src_left, src_right = st.columns([1, 1], gap="medium")
with src_left:
    with st.container(border=True):
        st.markdown(
            f"""
            <div class="doc-card">
              <h4>Our World in Data — COVID-19 Dataset</h4>
              <p>{utils.esc(owid.get('description', ''))}</p>
              <p style="margin-top:8px">
                <b>Coverage:</b> {utils.esc(utils.format_date(status.get('owid_date_min')))} →
                {utils.esc(utils.format_date(status.get('owid_date_max')))}<br>
                <b>Processed rows:</b> {int(owid_q.get('processed_rows') or 0):,}<br>
                <b>Countries:</b> {int(status.get('owid_countries') or 0):,}<br>
                <b>Licence:</b> CC BY 4.0
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<a href="{utils.esc(owid.get("url", ""))}" target="_blank" rel="noopener">Open the OWID source ↗</a>',
            unsafe_allow_html=True,
        )
with src_right:
    with st.container(border=True):
        st.markdown(
            f"""
            <div class="doc-card">
              <h4>covid19india.org — State-wise archive</h4>
              <p>{utils.esc(india.get('description', ''))}</p>
              <p style="margin-top:8px">
                <b>Coverage:</b> {utils.esc(utils.format_date(status.get('india_date_min')))} →
                {utils.esc(utils.format_date(status.get('india_date_max')))}<br>
                <b>States/UTs:</b> {int(status.get('india_states') or 0):,}<br>
                <b>Processed rows:</b> {int(india_q.get('processed_rows') or 0):,}<br>
                <b>Status:</b> historical public archive
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<a href="{utils.esc(india.get("url", ""))}" target="_blank" rel="noopener">Open the covid19india archive ↗</a>',
            unsafe_allow_html=True,
        )

utils.pills(
    [
        "OWID compact CSV",
        "covid19india states CSV",
        "India state GeoJSON",
        "No live API",
        "No fabricated statistics",
        "No hard-coded KPI values",
    ]
)
st.caption(
    "The India GeoJSON is used only for boundary drawing; it does not supply health data. The loader downloads the "
    "source files if they are absent, but the dashboard never presents the archive as a current live dataset."
)

# --------------------------------------------------------------------------- #
# Limitations
# --------------------------------------------------------------------------- #
utils.section_header(
    "LIMITATIONS",
    "What this project deliberately does not claim",
    "Limitations are part of the result, not a disclaimer added at the end.",
    note="Read before interpreting",
)
lim_left, lim_right = st.columns([1, 1], gap="medium")
with lim_left:
    with st.container(border=True):
        st.markdown(
            """
            <div class="doc-card">
              <h4>Data and measurement</h4>
              <ul>
                <li>Reported cases depend on testing volume, access and case definitions.</li>
                <li>Reported deaths depend on attribution and registration practice.</li>
                <li>CFR is not an infection fatality ratio and is sensitive to testing coverage.</li>
                <li>Negative daily values are retained as revisions; they can distort short windows.</li>
                <li>OWID's vaccination series ends in August 2024, while case rows extend further.</li>
                <li>Late-period rows are sparse and uneven across countries.</li>
              </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
with lim_right:
    with st.container(border=True):
        st.markdown(
            """
            <div class="doc-card">
              <h4>Scope and interpretation</h4>
              <ul>
                <li>The India archive ends on 1 June 2021; it is not a current status.</li>
                <li>State populations are not in the source, so state per-million rates are not shown.</li>
                <li>Active cases are derived (confirmed − recovered − deaths) and clipped at zero.</li>
                <li>Global recoveries and active cases are omitted for lack of a reliable series.</li>
                <li>Correlation is descriptive; no causal claim or forecast is made.</li>
                <li>Rule-based insights describe the reported series, not true transmission intensity.</li>
              </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

utils.callout(
    "<strong>India archive caveat.</strong> covid19india.org was a volunteer-led aggregator. Its state archive is a "
    "valuable historical record, but reporting gaps, late revisions and the 1 June 2021 endpoint limit what can be "
    "inferred about later waves. The dashboard states the endpoint on the India page, the landing page and here.",
    title="Historical endpoint, not a live status",
    tone="amber",
    icon_name="alert",
)

# --------------------------------------------------------------------------- #
# Reproduce
# --------------------------------------------------------------------------- #
utils.section_header(
    "REPRODUCE",
    "Run the project from a clean checkout",
    "The processed files are deterministic derivatives. Rebuild them after changing the raw archives.",
    note="Windows paths shown",
)
run_left, run_right = st.columns([1.15, 1], gap="medium")
with run_left:
    st.code(
        "python -m venv venv\n"
        "venv\\Scripts\\activate\n"
        "pip install -r requirements.txt\n"
        "python -m src.data_loader --build\n"
        "streamlit run app.py\n"
        "pytest",
        language="bash",
    )
    st.caption(
        "On macOS/Linux use `source venv/bin/activate`. The download step needs network access; if it is blocked, "
        "place the source files in data/raw/ using the paths documented in README.md."
    )
with run_right:
    st.code(
        "covid-dashboard/\n"
        "├── app.py\n"
        "├── pages/\n"
        "│   ├── 1_Global_Overview.py\n"
        "│   ├── 2_India_Analysis.py\n"
        "│   ├── 3_EDA.py\n"
        "│   ├── 4_Comparative_Analytics.py\n"
        "│   ├── 5_Time_Series_Insights.py\n"
        "│   ├── 6_Data_Explorer.py\n"
        "│   └── 7_Project_Information.py\n"
        "├── src/\n"
        "│   ├── data_loader.py\n"
        "│   ├── data_cleaning.py\n"
        "│   ├── analysis.py\n"
        "│   ├── visualizations.py\n"
        "│   ├── insights.py\n"
        "│   └── utils.py\n"
        "├── tests/\n"
        "├── data/raw · data/processed\n"
        "├── assets/\n"
        "├── requirements.txt\n"
        "└── README.md",
        language="text",
    )

# --------------------------------------------------------------------------- #
# Quality assurance / future scope
# --------------------------------------------------------------------------- #
utils.section_header(
    "ASSURANCE & NEXT STEPS",
    "Tests, verification and a realistic future scope",
    "The test suite targets the pure transformation and analysis layers; the UI is verified against the local historical files.",
    note="Engineering notes",
)
qa_left, qa_right = st.columns([1, 1], gap="medium")
with qa_left:
    with st.container(border=True):
        st.markdown(
            """
            <div class="doc-card">
              <h4>Quality assurance</h4>
              <ul>
                <li>Schema validation before cleaning; unsupported files fail with a clear message.</li>
                <li>Unit tests for rolling averages, daily differences, CFR, state normalisation and quality reports.</li>
                <li>Tests for snapshots, rankings, period comparisons, peaks and outlier detection.</li>
                <li>Loader tests build processed frames in memory without network access.</li>
                <li>Streamlit caching keys include processed-file modification times.</li>
                <li>No chart invents observations; empty filters produce an explicit empty state.</li>
              </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
with qa_right:
    with st.container(border=True):
        st.markdown(
            """
            <div class="doc-card">
              <h4>Future scope</h4>
              <ul>
                <li>Extend the India panel with a post-2021 source and reconcile definitions.</li>
                <li>Add uncertainty intervals and sensitivity analysis for reporting lags.</li>
                <li>Add vaccination-to-case and mobility analyses with explicit ecological caveats.</li>
                <li>Add a downloadable methods appendix and reproducible figure exports.</li>
                <li>Add automated data freshness checks without turning the dashboard into a live feed.</li>
                <li>Add accessibility and localisation reviews for a wider academic audience.</li>
              </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

# --------------------------------------------------------------------------- #
# Credits
# --------------------------------------------------------------------------- #
utils.section_header(
    "ACKNOWLEDGEMENTS",
    "Data and tools",
    "This project is built on public historical archives and open-source Python tooling.",
)
st.markdown(
    """
    <div class="doc-card" style="border:1px solid #22314A;background:#111827;padding:16px 18px">
      <p>
        Data: <b>Our World in Data</b> COVID-19 dataset (CC BY 4.0) and the <b>covid19india.org</b> state-wise public
        archive. India state boundaries: the publicly available <b>datta07/INDIAN-SHAPEFILES</b> GeoJSON. The
        dashboard is an academic project and is not affiliated with either data provider.
      </p>
      <p style="margin-top:8px">
        Tooling: Python, Streamlit, Pandas, NumPy, Plotly, Matplotlib, Seaborn, Requests and Pytest. The visual design
        is original to this build and uses a dark analytical palette with a consistent type and chart system.
      </p>
      <p style="margin-top:8px">
        All figures are reported historical values. The dashboard is <b>not a live feed</b>, does not forecast, and
        does not make causal claims.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

utils.source_strip(
    source=owid.get("name", "Our World in Data — COVID-19 Dataset"),
    url=owid.get("url", ""),
    secondary="covid19india.org archive",
    secondary_url=india.get("url", ""),
    latest=utils.format_date(status.get("owid_date_max")),
)
st.caption(
    "Academic project · MCA Semester III · Data Science. Source files remain subject to their providers' terms and "
    "retrospective revisions."
)
