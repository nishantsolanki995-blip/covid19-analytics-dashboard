"""Data Explorer — filter, sort, paginate and export the visible rows."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from src import utils
from src.analysis import filter_dataframe, paginate
from src.data_loader import (
    INDIA_RAW_PATH,
    INDIA_URL,
    OWID_RAW_PATH,
    OWID_URL,
    load_india_data,
    load_owid_data,
    load_raw_preview,
)


utils.page_setup("Data Explorer")

DATASETS = {
    "Global · OWID": "global",
    "India · covid19india": "india",
}
dataset_label = st.sidebar.radio("Dataset", list(DATASETS), key="explorer_dataset")
dataset = DATASETS[dataset_label]

try:
    with st.spinner("Loading the selected historical dataset…"):
        if dataset == "global":
            data = load_owid_data()
        else:
            data = load_india_data()
except Exception as exc:  # pragma: no cover - UI guard
    utils.show_data_error(str(exc), detail="python -m src.data_loader --build")
    st.stop()

entity_col = "country" if dataset == "global" else "state"
utils.sidebar_intro(
    latest=utils.format_date(data["date"].max()),
    rows=f"{len(data):,}",
    countries=(
        str(data.loc[data["is_country"].astype(bool), "country"].nunique())
        if dataset == "global"
        else str(data.loc[data["is_state"].astype(bool), "state"].nunique())
    ),
    entity_label="countries" if dataset == "global" else "states/UTs",
    coverage=f"{utils.format_date(data['date'].min())} →",
)
utils.sidebar_label("Explorer filters")

if dataset == "global":
    start, end = utils.date_range_filter(data, key="analysis_date_range")
    st.sidebar.caption(utils.date_window_caption(start, end))
    countries_only = st.sidebar.checkbox("Countries only", value=True, key="explorer_countries_only")
    if countries_only:
        data = data[data["is_country"].astype(bool)].copy()
else:
    start, end = utils.date_range_filter(
        data,
        key="india_date_range",
        label="India archive window",
    )
    st.sidebar.caption(utils.date_window_caption(start, end))
    scope = st.sidebar.radio(
        "Row scope",
        ["States/UTs only", "National only", "All rows"],
        index=0,
        key="explorer_india_scope",
    )
    if scope == "States/UTs only":
        data = data[data["is_state"].astype(bool)].copy()
    elif scope == "National only":
        data = data[data["entity_type"].eq("National")].copy()

utils.sidebar_note(
    "The explorer reads the processed panel. The raw source is available in the preview at the bottom of the page. "
    "No rows are reordered or edited for display."
)

filtered_dates = data
filtered_dates = filtered_dates[
    (pd.to_datetime(filtered_dates["date"]) >= pd.Timestamp(start))
    & (pd.to_datetime(filtered_dates["date"]) <= pd.Timestamp(end))
]
entity_options = sorted(filtered_dates[entity_col].dropna().astype(str).unique().tolist())
entities = st.sidebar.multiselect(
    "Entity filter",
    entity_options,
    default=[],
    key=f"explorer_entities_{dataset}",
    help="Leave empty to include every entity.",
)
search = st.sidebar.text_input(
    "Search entity name",
    key=f"explorer_search_{dataset}",
    placeholder="e.g. India, Kerala, Brazil",
)

utils.page_header(
    "DATA EXPLORER",
    "Inspect and export exactly what you see",
    "Search, filter, sort and paginate the processed historical panel. The download button exports the rows currently displayed, not a hidden full table.",
    accent="row by row",
    badge_label="PROCESSED PANEL",
    badge_value=dataset_label,
    badge_foot=f"{len(data):,} processed rows · through {utils.format_date(data['date'].max())}",
)

working = filter_dataframe(
    data,
    entities=entities or None,
    entity_col=entity_col,
    start=start,
    end=end,
    search=search or None,
)

# --------------------------------------------------------------------------- #
# Display controls
# --------------------------------------------------------------------------- #
all_columns = working.columns.tolist()
default_columns = (
    [
        "country",
        "iso_code",
        "continent",
        "date",
        "total_cases",
        "new_cases",
        "new_cases_7d",
        "total_deaths",
        "new_deaths",
        "total_cases_per_million",
        "total_deaths_per_million",
        "case_fatality_ratio",
        "vaccination_rate",
        "population",
    ]
    if dataset == "global"
    else [
        "date",
        "state",
        "entity_type",
        "total_cases",
        "new_cases",
        "new_cases_7d",
        "total_deaths",
        "new_deaths",
        "total_recovered",
        "active_cases",
        "case_fatality_ratio",
        "recovery_ratio",
        "revision_flag",
    ]
)
default_columns = [column for column in default_columns if column in all_columns]

control_left, control_right = st.columns([1.4, 1], gap="medium")
with control_left:
    columns = st.multiselect(
        "Visible columns",
        all_columns,
        default=default_columns or all_columns[:8],
        key=f"explorer_columns_{dataset}",
        help="Choose the fields shown in the table and exported to CSV.",
    )
with control_right:
    sort_col = st.selectbox(
        "Sort by",
        all_columns,
        index=all_columns.index("date") if "date" in all_columns else 0,
        key=f"explorer_sort_{dataset}",
    )
    sort_ascending = st.checkbox("Ascending order", value=False, key=f"explorer_sort_dir_{dataset}")
    page_size = st.selectbox("Rows per page", [25, 50, 100, 250], index=1, key=f"explorer_page_size_{dataset}")

if not columns:
    columns = default_columns or all_columns[:8]
if sort_col in working.columns:
    working = working.sort_values(sort_col, ascending=sort_ascending, na_position="last")

# --------------------------------------------------------------------------- #
# Pagination
# --------------------------------------------------------------------------- #
meta = paginate(working, 1, page_size)[1]
page_widget_key = f"explorer_page_input_{dataset}"
if page_widget_key not in st.session_state:
    st.session_state[page_widget_key] = 1
# Clamp before the widget is created; changing a widget's state after it has
# been instantiated in the same run is not allowed by Streamlit.
st.session_state[page_widget_key] = min(
    max(int(st.session_state[page_widget_key]), 1),
    max(1, meta["total_pages"]),
)


def _step_page(key: str, delta: int, total_pages: int) -> None:
    st.session_state[key] = min(max(int(st.session_state[key]) + delta, 1), max(1, total_pages))


nav_left, nav_mid, nav_right = st.columns([1, 2.2, 1])
with nav_left:
    st.button(
        "← Previous",
        width="stretch",
        disabled=meta["total_pages"] <= 1,
        key=f"explorer_prev_{dataset}",
        on_click=_step_page,
        args=(page_widget_key, -1, meta["total_pages"]),
    )
with nav_mid:
    page_number = st.number_input(
        "Page",
        min_value=1,
        max_value=max(1, meta["total_pages"]),
        step=1,
        key=page_widget_key,
    )
with nav_right:
    st.button(
        "Next →",
        width="stretch",
        disabled=meta["total_pages"] <= 1,
        key=f"explorer_next_{dataset}",
        on_click=_step_page,
        args=(page_widget_key, 1, meta["total_pages"]),
    )

page_df, meta = paginate(working, int(page_number), page_size)
start_row = (meta["page"] - 1) * meta["page_size"] + 1 if meta["total_rows"] else 0
end_row = min(meta["page"] * meta["page_size"], meta["total_rows"])

st.markdown(
    f'<div class="explorer-status"><span><b>{meta["total_rows"]:,}</b> matching rows</span>'
    f"<span>Showing <b>{start_row:,}–{end_row:,}</b></span>"
    f'<span>Page <b>{meta["page"]:,}</b> of <b>{meta["total_pages"]:,}</b></span>'
    f'<span>Sorted by <b>{utils.esc(sort_col)}</b></span></div>',
    unsafe_allow_html=True,
)

if working.empty:
    utils.empty_state(
        "No rows match these filters",
        "Clear the entity selection, widen the window or change the search term. Missing data is never filled in to create rows.",
    )
else:
    table = page_df[columns].copy()
    st.dataframe(
        table,
        width="stretch",
        hide_index=True,
        height=min(640, 36 * len(table) + 48),
    )
    st.caption(
        "Dates are shown as stored. Missing values are blank, not zero. Use the column selector above to inspect "
        "source identifiers or diagnostic flags."
    )

    export_left, export_right = st.columns([1, 1], gap="small")
    file_stem = f"{dataset}_{start:%Y%m%d}_{end:%Y%m%d}_page{meta['page']}"
    with export_left:
        st.download_button(
            "Download visible page · CSV",
            data=table.to_csv(index=False).encode("utf-8"),
            file_name=f"{file_stem}.csv",
            mime="text/csv",
            width="stretch",
            key=f"explorer_download_page_{dataset}",
        )
    with export_right:
        st.download_button(
            "Download all filtered rows · CSV",
            data=working[columns].to_csv(index=False).encode("utf-8"),
            file_name=f"{dataset}_{start:%Y%m%d}_{end:%Y%m%d}_filtered.csv",
            mime="text/csv",
            width="stretch",
            key=f"explorer_download_all_{dataset}",
        )

# --------------------------------------------------------------------------- #
# Dictionary and raw preview
# --------------------------------------------------------------------------- #
utils.section_header(
    "REFERENCE",
    "Column dictionary and source format",
    "The processed panel is a documented derivative. The raw files are shown separately so the transformation is auditable.",
    note="Processed vs raw",
)
dict_left, dict_right = st.columns([1, 1], gap="medium")
with dict_left:
    with st.container(border=True):
        st.markdown(
            '<div class="section-eyebrow">PROCESSED FIELDS</div>'
            '<div class="section-title">What the columns mean</div>',
            unsafe_allow_html=True,
        )
        if dataset == "global":
            dictionary = pd.DataFrame(
                [
                    ("country", "Country or aggregate name as supplied by OWID."),
                    ("iso_code", "ISO-3 code where available; used for the world map."),
                    ("date", "Observation date."),
                    ("total_cases", "Cumulative confirmed cases reported to date."),
                    ("new_cases", "Daily difference in cumulative confirmed cases."),
                    ("new_cases_7d", "OWID smoothed daily cases, or a trailing 7-day mean."),
                    ("total_deaths", "Cumulative reported deaths."),
                    ("total_cases_per_million", "Cumulative cases divided by population, times one million."),
                    ("case_fatality_ratio", "Deaths ÷ confirmed × 100; not an infection fatality ratio."),
                    ("vaccination_rate", "People with at least one dose per hundred population."),
                    ("population", "Population denominator supplied by OWID."),
                    ("is_country", "True when the row represents a country rather than an aggregate."),
                ],
                columns=["Column", "Definition"],
            )
        else:
            dictionary = pd.DataFrame(
                [
                    ("date", "Observation date in the covid19india archive."),
                    ("state", "State/UT, India national row, or State Unassigned."),
                    ("entity_type", "State/UT, National, or Unassigned."),
                    ("total_cases", "Cumulative confirmed cases."),
                    ("new_cases", "Daily difference in cumulative confirmed cases."),
                    ("total_deaths", "Cumulative reported deaths."),
                    ("total_recovered", "Cumulative reported recoveries."),
                    ("active_cases", "Confirmed − recovered − deaths, clipped at zero."),
                    ("case_fatality_ratio", "Deaths ÷ confirmed × 100."),
                    ("recovery_ratio", "Recovered ÷ confirmed × 100."),
                    ("revision_flag", "True when a daily difference is negative due to a revision."),
                ],
                columns=["Column", "Definition"],
            )
        st.dataframe(dictionary, width="stretch", hide_index=True, height=330)
with dict_right:
    with st.container(border=True):
        st.markdown(
            '<div class="section-eyebrow">RAW PREVIEW</div>'
            '<div class="section-title">First rows of the source file</div>',
            unsafe_allow_html=True,
        )
        try:
            if dataset == "global" and OWID_RAW_PATH.exists():
                raw_preview = load_raw_preview(30)
                preview_note = "A head preview of the raw OWID file, not a random sample. Raw files are not edited in place."
            elif dataset == "global":
                raw_preview = data.head(30)
                preview_note = (
                    "The 185 MB raw OWID file is not bundled in this deployment. "
                    "Showing the first processed rows instead; run the data loader locally for the raw preview."
                )
            elif INDIA_RAW_PATH.exists():
                raw_preview = pd.read_csv(INDIA_RAW_PATH, nrows=30)
                preview_note = "A head preview of the raw covid19india file, not a random sample. Raw files are not edited in place."
            else:
                raw_preview = data.head(30)
                preview_note = "The raw covid19india CSV is unavailable; showing the first processed rows instead."
            st.dataframe(raw_preview, width="stretch", hide_index=True, height=330)
            st.caption(preview_note)
        except Exception as exc:
            utils.empty_state("Raw preview unavailable", utils.esc(str(exc)))

utils.source_strip(
    source="Our World in Data — COVID-19 Dataset" if dataset == "global" else "covid19india.org — State-wise archive",
    url=OWID_URL if dataset == "global" else INDIA_URL,
    latest=utils.format_date(data["date"].max()),
)
st.caption(
    "Exports contain exactly the selected columns and the rows currently on screen. The date filter applies to the "
    "whole explorer, not just the visible page."
)
