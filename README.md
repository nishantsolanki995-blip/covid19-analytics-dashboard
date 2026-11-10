# COVID-19 Pandemic Trends & Comparative Data Analytics Dashboard

An MCA Semester III Data Science project built with **Python, Streamlit, Pandas, NumPy, Plotly, Matplotlib and Seaborn**.

The dashboard is an **interactive historical analysis workspace** for global and Indian COVID-19 data. It combines country-level data from **Our World in Data (OWID)** with the **covid19india.org state archive**, documents the cleaning steps, and presents descriptive analytics with explicit provenance and limitations.

> **This is not a live feed.** Every figure is a reported historical value from the selected source, through the observation date shown in the app. Missing values are never silently converted to zero for headline KPIs.

---

## Table of contents

- [Overview](#overview)
- [Features](#features)
- [Dashboard pages](#dashboard-pages)
- [Project structure](#project-structure)
- [Quick start](#quick-start)
- [Data sources](#data-sources)
- [Data pipeline and methodology](#data-pipeline-and-methodology)
- [Metric definitions](#metric-definitions)
- [Data quality and transparency](#data-quality-and-transparency)
- [Testing](#testing)
- [Limitations](#limitations)
- [Academic context and future scope](#academic-context-and-future-scope)
- [Troubleshooting](#troubleshooting)
- [Attribution](#attribution)

---

## Overview

COVID-19 data is abundant but uneven. Cases, deaths, recoveries and vaccination coverage arrive at different cadences, cover different periods, and are revised retrospectively. A dashboard that hides those differences can produce confident-looking but misleading conclusions.

This project builds a reproducible analytical workspace that keeps the following visible together:

- the original source and its coverage window;
- the cleaning and transformation rules;
- the difference between missing data and a reported zero;
- cumulative totals versus daily activity;
- reported metrics versus derived metrics;
- what the data can and cannot support.

The app is designed for academic review and exploratory analysis. It does not forecast, infer causality, or present itself as an official public-health surveillance system.

---

## Features

- **Global overview:** world totals, daily cases and deaths, per-capita choropleth, country rankings, and date-aware analytical notes.
- **India analysis:** national and state/UT trends from the historical covid19india archive, state map, rankings, recoveries and active cases with caveats.
- **Exploratory data analysis:** schema inspection, missingness, duplicate checks, negative-revision checks, distributions, IQR outliers, and Spearman correlation.
- **Comparative analytics:** multi-country trajectories, per-capita comparisons, shared time windows, and epidemic-day alignment at a chosen threshold.
- **Time series and insights:** peak detection, monthly heatmaps, adjacent-period comparisons, and rule-based observations generated from the selected data.
- **Data explorer:** dataset switch, date filter, entity search, visible-column selection, sorting, pagination, and CSV export of the visible page or all filtered rows.
- **Project information:** problem statement, objectives, sources, metric definitions, methodology, limitations, and reproduction steps.
- **Premium dark interface:** custom CSS design system, consistent Plotly and Matplotlib themes, responsive cards, source strips, and accessible data tables.
- **Transparent engineering:** deterministic processed datasets, a generated data-quality report, and automated tests for cleaning, analysis, and loading.

---

## Dashboard pages

| Page | Purpose |
|---|---|
| **Overview** (`app.py`) | Landing page with latest global snapshot, a two-chart analytical preview, workspace links, and data provenance. |
| **Global Overview** | World trend charts, per-capita map, top-10 rankings, and KPI cards from the OWID `World` aggregate. |
| **India Analysis** | National and state/UT analysis using the covid19india archive through 1 June 2021. Includes state map and recoveries/active-case disclosure. |
| **EDA** | Data-quality scorecards, column-level schema tables, missingness, distributions, outlier review, and correlation structure. |
| **Comparative Analytics** | Country comparison over a shared date window, ranking metrics, per-capita rates, and epidemic-day alignment. |
| **Time Series & Insights** | Peaks, 7-day activity, monthly patterns, period changes, and traceable rule-based insight cards. |
| **Data Explorer** | Filter, sort, paginate, inspect, and export the processed global or India panel. |
| **Project Information** | Academic brief, sources, pipeline, definitions, limitations, and reproducibility instructions. |

The analytical pages share date-filter state:

- `analysis_date_range` is shared by Global Overview, EDA, Comparative Analytics, Time Series & Insights, and the global Data Explorer.
- `india_date_range` is shared by India Analysis and the India Data Explorer.

---

## Project structure

```text
covid-dashboard/
├── app.py                         # Entry point and landing page
├── README.md
├── requirements.txt
├── pytest.ini
├── .gitignore
├── .streamlit/
│   └── config.toml                # Dark Streamlit theme
├── assets/
│   ├── logo.png
│   └── dashboard_assets/
│       ├── logo_mark.png
│       └── styles.css             # Custom dark design system
├── data/
│   ├── raw/                       # Historical source files (large files gitignored)
│   │   ├── owid_covid_compact.csv
│   │   ├── india_states.csv
│   │   └── india_states.geojson
│   └── processed/
│       ├── owid_country_daily.csv.gz
│       ├── india_states_daily.csv.gz
│       └── data_quality.json
├── pages/
│   ├── 1_Global_Overview.py
│   ├── 2_India_Analysis.py
│   ├── 3_EDA.py
│   ├── 4_Comparative_Analytics.py
│   ├── 5_Time_Series_Insights.py
│   ├── 6_Data_Explorer.py
│   └── 7_Project_Information.py
├── src/
│   ├── __init__.py
│   ├── data_loader.py             # Download, cache, build and status helpers
│   ├── data_cleaning.py           # Schema validation, cleaning, India grid, quality reports
│   ├── analysis.py                # Snapshots, rankings, comparisons, peaks, EDA helpers
│   ├── visualizations.py          # Plotly and Matplotlib/Seaborn charts and maps
│   ├── insights.py                # Deterministic rule-based insight generators
│   └── utils.py                   # Page setup, CSS, formatting, filters and UI components
└── tests/
    ├── test_data_cleaning.py
    ├── test_analysis.py
    └── test_data_loader.py
```

---

## Quick start

### Prerequisites

- **Python 3.10+** recommended. The project was built and tested with **Python 3.12**.
- Internet access for the first data download. If the source endpoints are unavailable, the raw files can be placed manually in `data/raw/`.
- Approximately **1 GB of free disk space** for the virtual environment, raw OWID file, processed panels and caches.

### Windows PowerShell

```powershell
cd covid-dashboard
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m src.data_loader --build
streamlit run app.py
```

The app opens at `http://localhost:8501`.

> If PowerShell blocks activation, use `Set-ExecutionPolicy -Scope Process Bypass` for the current session, or invoke the virtual environment directly:
>
> ```powershell
> .\venv\Scripts\python.exe -m streamlit run app.py
> ```

### macOS / Linux

```bash
cd covid-dashboard
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m src.data_loader --build
streamlit run app.py
```

### Data preparation

The first run can download and build the datasets automatically. To do it explicitly:

```bash
python -m src.data_loader --build
```

Useful loader options:

```bash
python -m src.data_loader --build --force   # re-download raw files and rebuild
python -m src.data_loader --download        # download the OWID raw CSV only
```

The processed build writes:

- `data/processed/owid_country_daily.csv.gz`
- `data/processed/india_states_daily.csv.gz`
- `data/processed/data_quality.json`

The raw OWID compact CSV is approximately **185 MB** and is intentionally gitignored. The processed files are deterministic derivatives and can be rebuilt.

### Run the dashboard

```bash
streamlit run app.py
```

The landing page and all seven analytical pages are available from the Streamlit sidebar.

---

## Data sources

| Source | Role | Coverage in the current build | Link |
|---|---|---|---|
| **Our World in Data — COVID-19 Dataset** (`compact.csv`) | Global country/date panel for cases, deaths, testing, vaccination and contextual indicators | 2020-01-01 → 2026-10-18; 238 country entities in the processed panel | <https://catalog.ourworldindata.org/garden/covid/latest/compact/compact.csv> |
| **covid19india.org state archive** (`states.csv`) | Historical Indian state/UT cumulative confirmed, recovered and deceased counts | 2020-01-30 → 2021-06-01; 36 states/UTs plus national and unassigned rows | <https://api.covid19india.org/csv/latest/states.csv> |
| **India state boundaries** | GeoJSON used for the India choropleth | State/UT polygons | <https://raw.githubusercontent.com/datta07/INDIAN-SHAPEFILES/master/INDIA/INDIA_STATES.geojson> |

The figures above describe the current generated build. Re-running the loader refreshes them from the source files.

### Source notes

- **OWID** compiles country-level data from national authorities and other aggregators. Country definitions, testing regimes and reporting practices vary.
- **covid19india.org** was a volunteer-led aggregator. Its public CSV archive available for this project ends on **1 June 2021**; the dashboard treats it as a historical archive, not current data.
- **OWID vaccination coverage** currently ends in **August 2024**, while case rows extend further. Late-period case rows are sparse and uneven across countries.
- The India GeoJSON is used for visualisation only. Boundaries are not an authoritative political statement.

---

## Data pipeline and methodology

### 1. Acquisition

`src/data_loader.py` downloads source files to `data/raw/` using streamed, atomic writes (`.part` then rename) and validates minimum file size. Downloads are skipped when a valid local file already exists.

### 2. Schema validation

`src/data_cleaning.py` validates required OWID columns before cleaning. A missing or incompatible schema raises an explicit error with a recovery instruction instead of failing silently.

### 3. Cleaning and standardisation

- Dates are parsed and invalid dates are counted in the quality report.
- Country and state names are normalised.
- OWID country entities are flagged with `is_country`; the `World` aggregate and other non-country entities are kept for aggregate analysis but excluded from country snapshots.
- Duplicate `country + date` rows are checked and reported.
- Negative daily values and backward revisions are retained and flagged rather than erased.
- A compact analytical column set is selected for the processed panel.

### 4. Global snapshot

`latest_snapshot()` creates one latest-available row per country:

- cumulative fields are forward-filled within each country to carry forward the last reported cumulative value;
- entities without a usable `total_cases` value are excluded from the cumulative snapshot;
- vaccination recency is carried separately as `vaccination_as_of`;
- per-capita metrics are recomputed only when the required numerator and population are available.

### 5. India panel

The India archive is expanded to a complete daily grid for each state/UT:

- cumulative confirmed, recovered and deceased values are forward-filled within each state;
- daily new values are derived from cumulative changes;
- active cases are derived as `total_cases - total_recovered - total_deaths` and clipped at zero;
- recovery and fatality ratios are computed from reported cumulative values;
- state names are mapped to the GeoJSON for visualisation;
- the national `India` row and the `State Unassigned` row are retained for analysis but excluded from state-only rankings.

### 6. Derived analytics

`src/analysis.py` provides reusable functions for:

- latest snapshots and world aggregates;
- rankings and country/state comparisons;
- adjacent-window and period-over-period changes;
- epidemic-day alignment at a chosen cumulative threshold;
- peak detection on ordered daily series;
- IQR outlier detection;
- Spearman correlation matrices;
- coverage and missingness summaries.

### 7. Visualisation

`src/visualizations.py` builds Plotly charts and Matplotlib/Seaborn figures with a shared dark theme. All maps use explicit data traces and graceful empty states. No chart injects synthetic observations.

### 8. Insight generation

`src/insights.py` contains deterministic rules that describe what the selected data shows. Insights are labelled as descriptive observations, cite the metric and window, and never claim causation or live status.

---

## Metric definitions

| Metric | Definition | Notes |
|---|---|---|
| **Total confirmed** | Cumulative reported confirmed cases. | Reported, not estimated. |
| **Total deaths** | Cumulative reported deaths. | Reported, not estimated. |
| **New cases / new deaths** | Daily change in reported cumulative values. | Negative revisions are retained and flagged. |
| **7-day average** | OWID smoothed series where supplied; otherwise a trailing 7-day mean. | Raw daily reports may include batch releases. |
| **Case fatality ratio (CFR)** | `total_deaths ÷ total_cases`. | This is a reported-case ratio, **not** the infection fatality rate (IFR). |
| **Cases / deaths per million** | `cumulative count ÷ population × 1,000,000`. | Depends on the population denominator supplied by OWID. |
| **First-dose coverage** | Share of the population with at least one vaccine dose, from OWID's `people_vaccinated`/population. | Displayed with the latest available vaccination date. |
| **Fully vaccinated / booster rate** | OWID's fully vaccinated and booster shares. | Coverage tails differ by country. |
| **Recoveries** | India only: cumulative reported recovered. | Not shown globally because OWID does not publish a consistent recovery series. |
| **Active cases** | India only: `total_cases - total_recovered - total_deaths`, clipped at zero. | Derived; sensitive to reporting revisions. |
| **Epidemic day** | Days since a country first reached a chosen cumulative-cases threshold. | Used for shape comparison, not calendar comparison. |
| **Adjacent-window change** | Current window sum versus the immediately preceding equal-length window. | A missing denominator returns `None`, not infinity. |

### Missing data rule

A missing value is **not** a zero. Headline KPIs use reported or explicitly forward-filled cumulative values. Missing vaccination, testing or contextual indicators remain missing in the data and in the quality report.

---

## Data quality and transparency

The app generates `data/processed/data_quality.json` on build. The EDA and Project Information pages expose:

- raw row and column counts;
- date coverage;
- duplicate `country + date` rows;
- invalid dates;
- negative cumulative cases/deaths;
- rows where deaths exceed cases;
- negative daily values;
- missing-cell counts and percentages;
- processed-panel missingness;
- entity counts;
- India recovery coverage and negative daily rows.

### Current build snapshot

| Dataset | Raw rows | Processed rows | Date coverage | Entities |
|---|---:|---:|---|---:|
| OWID global panel | 629,868 | 620,753 | 2020-01-01 → 2026-10-18 | 238 countries |
| covid19india archive | 16,060 | 18,582 | 2020-01-30 → 2021-06-01 | 36 states/UTs + national/unassigned |

The India processed row count is higher than the raw count because the state panel is expanded to a complete daily grid.

### Reporting revisions

Negative daily values and backward revisions are not hidden. They are retained, flagged with `revision_flag`, and described in the quality report. This matters for cumulative totals and for period comparisons.

---

## Testing

The test suite covers the cleaning rules, analytical functions and loader contract.

```bash
pytest
```

Current result:

```text
34 passed
```

The tests include:

- schema validation and cleaning behaviour;
- India daily-grid construction and derived active cases;
- latest-snapshot and ranking logic;
- window and period comparisons;
- trend direction, peak detection and vaccination coverage;
- quality-report contracts;
- atomic download behaviour and incomplete-download rejection;
- dataset-status keys used by the UI.

The Streamlit UI was also smoke-tested in a browser across all seven analytical pages, including pagination and the global/India Data Explorer switch.

---

## Limitations

- **Historical, not live.** The app reads local archived sources. OWID case rows extend to October 2026, but the India archive ends on 1 June 2021.
- **Uneven reporting.** Countries and Indian states differ in testing, case definitions, death attribution, reporting frequency and revision practices.
- **Global recoveries and active cases are omitted.** OWID does not provide a consistent global recovery series. Deriving active cases globally would create a false sense of precision.
- **Indian recoveries and active cases are derived** from three reported cumulative series and clipped at zero. They should be read with the archive's reporting caveats.
- **No state-level per-million comparisons.** The India archive does not include state populations, so state comparisons use counts and shares, not population-adjusted rates.
- **Vaccination data have a different tail.** OWID vaccination coverage ends in August 2024; case rows extend further. Coverage is not a live vaccination monitor.
- **Sparse late-period rows.** Late-period case data are uneven and sometimes zero-filled by reporting practice. Trends near the end of the series should be interpreted cautiously.
- **CFR is not IFR.** The case fatality ratio uses reported cases and deaths and is sensitive to testing and attribution.
- **Correlation is not causation.** EDA relationships are ecological and descriptive.
- **No forecasting or causal inference.** The project deliberately stays within descriptive analytics.
- **Map boundaries are visual only.** The GeoJSON is not an authoritative statement on disputed boundaries.

---

## Academic context and future scope

This dashboard was built as an **MCA Semester III Data Science project**. Its purpose is to demonstrate:

- reproducible data acquisition and cleaning;
- transparent data-quality reporting;
- comparative and time-series analytics;
- interactive visualisation with Streamlit and Plotly;
- testing and documentation of an analytical application.

Possible future extensions:

- integrate a maintained live source with explicit freshness monitoring;
- add state-level population denominators for per-million comparisons;
- add uncertainty intervals and revision-aware trend estimates;
- add mobility, policy-stringency or genomic-surveillance context;
- add spatial autocorrelation and clustering;
- add forecasting as a clearly separated, evaluated module;
- containerise the app and add automated UI tests.

---

## Troubleshooting

### `pandas` fails to install or import on Windows

This project pins **pandas 2.2.3** because Windows Smart App Control may block later pandas C extensions. Use a clean Python 3.12 virtual environment and install the pinned requirements.

### The OWID download fails

Download the compact CSV manually and save it as:

```text
data/raw/owid_covid_compact.csv
```

Then rebuild:

```bash
python -m src.data_loader --build
```

### The India CSV endpoint is unavailable

Place a compatible `states.csv` at:

```text
data/raw/india_states.csv
```

The loader also expects `data/raw/india_states.geojson` for the India map. If the GeoJSON is missing, the rest of the India page still works and the map degrades gracefully.

### Data looks stale or the app does not reflect a rebuild

Streamlit caches data reads by file modification time. Rebuild with `python -m src.data_loader --build`, then reload the page or clear the app cache from the Streamlit menu.

### The sidebar says a processed file is missing

Run the build command from the project root, not from inside `src/` or `pages/`:

```bash
python -m src.data_loader --build
```

---

## Attribution

- **Our World in Data** — COVID-19 dataset, CC BY 4.0. Source: <https://ourworldindata.org/coronavirus>.
- **covid19india.org** — volunteer-led historical state archive. Source and terms: <https://data.covid19india.org/>.
- **India state GeoJSON** — public shapefile repository used for map visualisation: <https://github.com/datta07/INDIAN-SHAPEFILES>.

This repository is an academic project. The source datasets remain subject to their original licences and terms of use. The dashboard does not claim ownership of the underlying data and does not present it as an official public-health source.
