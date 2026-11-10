"""Shared UI, formatting and configuration helpers.

The module deliberately contains no data logic.  It owns the *presentation*
layer: colour tokens, HTML components, formatting helpers and Streamlit
configuration so that every page looks and behaves consistently.
"""
from __future__ import annotations

import html
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------- #
# Paths and design tokens
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = ROOT / "assets"
DASHBOARD_ASSETS = ASSETS_DIR / "dashboard_assets"
LOGO_PATH = ASSETS_DIR / "logo.png"
LOGO_MARK_PATH = DASHBOARD_ASSETS / "logo_mark.png"
CSS_PATH = DASHBOARD_ASSETS / "styles.css"
DATA_RAW_DIR = ROOT / "data" / "raw"
DATA_PROCESSED_DIR = ROOT / "data" / "processed"

COLORS: dict[str, str] = {
    "bg": "#0B1220",
    "bg_soft": "#0E1728",
    "card": "#182235",
    "card_2": "#111827",
    "border": "#22314A",
    "text": "#F8FAFC",
    "muted": "#94A3B8",
    "blue": "#3B82F6",
    "cyan": "#06B6D4",
    "green": "#22C55E",
    "red": "#EF4444",
    "amber": "#F59E0B",
    "violet": "#8B5CF6",
}

COLOR_SEQUENCE: list[str] = [
    "#3B82F6", "#06B6D4", "#22C55E", "#F59E0B", "#EF4444",
    "#8B5CF6", "#EC4899", "#14B8A6", "#F97316", "#A3E635",
]

PLOTLY_CONFIG: dict[str, Any] = {
    "displayModeBar": False,
    "displaylogo": False,
    "responsive": True,
    "scrollZoom": False,
    "doubleClick": False,
}

TONE_ICON = {
    "blue": "activity",
    "cyan": "globe",
    "green": "heart",
    "red": "shield",
    "amber": "alert",
    "violet": "chart",
}

# Minimal Lucide-style line icons (inline SVG keeps the app fully self-contained).
_ICON_PATHS: dict[str, str] = {
    "activity": '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>',
    "globe": '<circle cx="12" cy="12" r="10"/><path d="M2 12h20"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/>',
    "heart": '<path d="M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4.05 3 5.5l7 7Z"/><path d="M3.22 12H9.5l.5-1 2 4.5 2-7 1.5 3.5h5.27"/>',
    "shield": '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>',
    "percent": '<line x1="19" y1="5" x2="5" y2="19"/><circle cx="6.5" cy="6.5" r="2.5"/><circle cx="17.5" cy="17.5" r="2.5"/>',
    "syringe": '<path d="m18 2 4 4"/><path d="m17 7 3-3"/><path d="M19 9 8.7 19.3c-1 1-2.5 1-3.4 0l-.6-.6c-1-1-1-2.5 0-3.4L15 5"/><path d="m9 11 4 4"/><path d="m5 19-3 3"/><path d="m14 4 6 6"/>',
    "table": '<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18"/><path d="M3 15h18"/><path d="M9 3v18"/>',
    "chart": '<path d="M3 3v18h18"/><path d="M18 17V9"/><path d="M13 17V5"/><path d="M8 17v-3"/>',
    "clock": '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
    "info": '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>',
    "database": '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v14c0 1.66 3.58 3 8 3s8-1.34 8-3V5"/><path d="M4 12c0 1.66 3.58 3 8 3s8-1.34 8-3"/>',
    "map": '<path d="m3 6 6-3 6 3 6-3v15l-6 3-6-3-6 3z"/><path d="M9 3v15"/><path d="M15 6v15"/>',
    "arrow": '<path d="M7 17 17 7"/><path d="M7 7h10v10"/>',
    "alert": '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><path d="M12 9v4"/><path d="M12 17h.01"/>',
    "layers": '<path d="m12 2 9 5-9 5-9-5 9-5Z"/><path d="m3 12 9 5 9-5"/><path d="m3 17 9 5 9-5"/>',
    "compare": '<path d="M12 3v18"/><path d="M5 7h14"/><path d="m8 7-3 6h6L8 7Z"/><path d="m16 7-3 6h6l-3-6Z"/>',
    "file": '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8Z"/><path d="M14 2v6h6"/><path d="M8 13h8"/><path d="M8 17h6"/>',
    "home": '<path d="m3 10 9-7 9 7v10a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2Z"/><path d="M9 22V12h6v10"/>',
    "trend_up": '<path d="m3 17 6-6 4 4 8-8"/><path d="M15 7h6v6"/>',
    "trend_down": '<path d="m3 7 6 6 4-4 8 8"/><path d="M15 17h6v-6"/>',
    "users": '<path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>',
    "calendar": '<rect x="3" y="4" width="18" height="18" rx="2"/><path d="M16 2v4"/><path d="M8 2v4"/><path d="M3 10h18"/>',
    "download": '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="M7 10l5 5 5-5"/><path d="M12 15V3"/>',
    "search": '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>',
    "check": '<path d="M20 6 9 17l-5-5"/>',
    "map_pin": '<path d="M20 10c0 6-8 12-8 12s-8-6-8-12a8 8 0 0 1 16 0Z"/><circle cx="12" cy="10" r="3"/>',
    "help": '<circle cx="12" cy="12" r="10"/><path d="M9.1 9a3 3 0 0 1 5.8 1c0 2-3 2-3 4"/><path d="M12 17h.01"/>',
}


def icon(name: str, size: int = 16) -> str:
    """Return an inline SVG string for a named icon."""
    paths = _ICON_PATHS.get(name, _ICON_PATHS["activity"])
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" '
        f'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{paths}</svg>'
    )


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #
def format_compact(value: Any, decimals: int = 1) -> str:
    """Compact human-readable number: 1,234,567 -> 1.2M."""
    if value is None:
        return "—"
    try:
        if pd.isna(value):
            return "—"
    except (TypeError, ValueError):
        pass
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if number == 0:
        return "0"
    sign = "-" if number < 0 else ""
    number = abs(number)
    for divisor, suffix in ((1_000_000_000, "B"), (1_000_000, "M"), (1_000, "K")):
        if number >= divisor:
            text = f"{number / divisor:,.{decimals}f}".rstrip("0").rstrip(".")
            return f"{sign}{text}{suffix}"
    if number >= 100:
        return f"{sign}{number:,.0f}"
    if number >= 1:
        text = f"{sign}{number:,.{min(decimals, 2)}f}".rstrip("0").rstrip(".")
        return text
    return f"{sign}{number:.2f}"


def format_int(value: Any) -> str:
    if value is None:
        return "—"
    try:
        if pd.isna(value):
            return "—"
        return f"{float(value):,.0f}"
    except (TypeError, ValueError):
        return "—"


def format_decimal(value: Any, digits: int = 2) -> str:
    if value is None:
        return "—"
    try:
        if pd.isna(value):
            return "—"
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def format_pct(value: Any, digits: int = 1) -> str:
    if value is None:
        return "—"
    try:
        if pd.isna(value):
            return "—"
        return f"{float(value):.{digits}f}%"
    except (TypeError, ValueError):
        return "—"


def format_signed_pct(value: Any, digits: int = 1) -> str:
    if value is None:
        return "—"
    try:
        if pd.isna(value):
            return "—"
        return f"{float(value):+.{digits}f}%"
    except (TypeError, ValueError):
        return "—"


def format_date(value: Any, fmt: str = "%d %b %Y") -> str:
    if value is None:
        return "—"
    try:
        stamp = pd.Timestamp(value)
        if pd.isna(stamp):
            return "—"
        return stamp.strftime(fmt)
    except (TypeError, ValueError):
        return "—"


def format_month(value: Any) -> str:
    return format_date(value, "%b %Y")


def truncate(text: Any, length: int = 60) -> str:
    text = str(text)
    return text if len(text) <= length else text[: length - 1] + "…"


def pct_change(current: Any, previous: Any) -> float | None:
    """Percentage change with safe handling of zero and missing values."""
    try:
        current_f, previous_f = float(current), float(previous)
    except (TypeError, ValueError):
        return None
    if pd.isna(current_f) or pd.isna(previous_f) or previous_f == 0:
        return None
    return (current_f - previous_f) / abs(previous_f) * 100.0


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


# --------------------------------------------------------------------------- #
# Page configuration
# --------------------------------------------------------------------------- #
def inject_css() -> None:
    try:
        css = CSS_PATH.read_text(encoding="utf-8")
    except OSError:
        css = ""
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def page_setup(title: str, *, icon_path: Path | None = None) -> None:
    """Configure every page identically (must be the first Streamlit call)."""
    st.set_page_config(
        page_title=f"{title} · COVID Analytics",
        page_icon=str(icon_path or (LOGO_MARK_PATH if LOGO_MARK_PATH.exists() else LOGO_PATH)),
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()
    try:
        st.logo(
            str(LOGO_PATH),
            size="large",
            icon_image=str(LOGO_MARK_PATH) if LOGO_MARK_PATH.exists() else None,
        )
    except Exception:
        # Logo is decorative; never let it break the app.
        pass


def plotly(fig: Any, *, key: str | None = None) -> None:
    """Render a Plotly figure with the project theme and no mode bar."""
    st.plotly_chart(
        fig,
        width="stretch",
        theme=None,
        config=PLOTLY_CONFIG,
        key=key,
    )


def pyplot(fig: Any, *, key: str | None = None) -> None:
    st.pyplot(fig, clear_figure=True, width="stretch")


# --------------------------------------------------------------------------- #
# Composite HTML components
# --------------------------------------------------------------------------- #
def page_header(
    eyebrow: str,
    title: str,
    subtitle: str,
    *,
    accent: str | None = None,
    badge_label: str = "HISTORICAL DATA",
    badge_value: str = "",
    badge_foot: str = "",
) -> None:
    if accent:
        title_html = f"{esc(title)} <span class='accent'>{esc(accent)}</span>"
    else:
        title_html = esc(title)
    badge = ""
    if badge_value:
        badge = f"""
        <div class="head-card">
          <div class="hc-label"><span class="live-dot"></span>{esc(badge_label)}</div>
          <div class="hc-value">{esc(badge_value)}</div>
          <div class="hc-foot">{esc(badge_foot)}</div>
        </div>
        """
    st.markdown(
        f"""
        <div class="page-head">
          <div>
            <div class="eyebrow"><span class="eyebrow-dot"></span>{esc(eyebrow)}</div>
            <div class="page-title">{title_html}</div>
            <p class="page-sub">{subtitle}</p>
          </div>
          {badge}
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(
    kicker: str,
    title: str,
    subtitle: str | None = None,
    *,
    note: str | None = None,
) -> None:
    sub = f'<p class="section-sub">{subtitle}</p>' if subtitle else ""
    note_html = f'<div class="section-note">{esc(note)}</div>' if note else ""
    st.markdown(
        f"""
        <div class="section-head">
          <div>
            <div class="section-eyebrow">{esc(kicker)}</div>
            <div class="section-title">{esc(title)}</div>
            {sub}
          </div>
          {note_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(
    label: str,
    value: str,
    *,
    sub: str = "",
    tone: str = "blue",
    icon_name: str | None = None,
    chip: str = "",
    delta: str = "",
    delta_tone: str = "neutral",
    title: str = "",
) -> dict[str, str]:
    return {
        "label": label,
        "value": value,
        "sub": sub,
        "tone": tone,
        "icon": icon_name or TONE_ICON.get(tone, "activity"),
        "chip": chip,
        "delta": delta,
        "delta_tone": delta_tone,
        "title": title or value,
    }


def render_kpis(cards: Sequence[Mapping[str, str]]) -> None:
    if not cards:
        return
    parts: list[str] = []
    for card in cards:
        tone = card.get("tone", "blue")
        delta = card.get("delta", "")
        delta_html = ""
        if delta:
            sign = delta.strip()[:1]
            arrow = "▲" if sign == "+" else "▼" if sign == "-" else "•"
            delta_html = (
                f'<span class="kpi-delta {esc(card.get("delta_tone", "neutral"))}">'
                f"{arrow} {esc(delta)}</span>"
            )
        chip = f'<span class="kpi-chip">{esc(card["chip"])}</span>' if card.get("chip") else ""
        parts.append(
            f"""
            <div class="kpi-card tone-{esc(tone)}">
              <div class="kpi-top">
                <span class="kpi-label">{esc(card["label"])}</span>
                <span class="kpi-icon">{icon(card.get("icon", "activity"), 14)}</span>
              </div>
              <div class="kpi-value" title="{esc(card.get('title', card['value']))}">{esc(card["value"])}</div>
              <div class="kpi-sub">{esc(card.get("sub", ""))}</div>
              <div class="kpi-foot">{chip}{delta_html}</div>
            </div>
            """
        )
    st.markdown(
        f'<div class="kpi-grid">{"".join(part.strip() for part in parts)}</div>',
        unsafe_allow_html=True,
    )


def callout(
    body: str,
    *,
    title: str = "",
    tone: str = "blue",
    icon_name: str | None = None,
) -> None:
    icon_name = icon_name or TONE_ICON.get(tone, "info")
    title_html = f'<div class="callout-title">{esc(title)}</div>' if title else ""
    st.markdown(
        f"""
        <div class="callout tone-{esc(tone)}">
          <span class="callout-icon">{icon(icon_name, 15)}</span>
          <div>{title_html}<div class="callout-body">{body}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def insight_grid(insights: Iterable[Mapping[str, str]]) -> None:
    items = list(insights)
    if not items:
        return
    parts = []
    for item in items:
        parts.append(
            f"""
            <div class="insight-card tone-{esc(item.get('tone', 'blue'))}">
              <div class="insight-kicker">{esc(item.get('kicker', 'INSIGHT'))}</div>
              <div class="insight-title">{esc(item.get('title', ''))}</div>
              <div class="insight-body">{item.get('body', '')}</div>
              <div class="insight-source">{esc(item.get('source', ''))}</div>
            </div>
            """
        )
    st.markdown(
        f'<div class="insight-grid">{"".join(part.strip() for part in parts)}</div>',
        unsafe_allow_html=True,
    )


def empty_state(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="empty-state">
          <div class="es-title">{esc(title)}</div>
          <div class="es-body">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def pills(items: Iterable[str]) -> None:
    html_items = "".join(f'<span class="pill">{esc(item)}</span>' for item in items)
    st.markdown(f'<div class="pill-row">{html_items}</div>', unsafe_allow_html=True)


def source_strip(
    *,
    source: str,
    url: str,
    secondary: str = "",
    secondary_url: str = "",
    latest: str = "",
) -> None:
    secondary_html = ""
    if secondary:
        link = f'<a href="{esc(secondary_url)}" target="_blank" rel="noopener">{esc(secondary)}</a>'
        secondary_html = f"<span>India states: {link}</span>"
    latest_html = f"<span>Latest record: <b>{esc(latest)}</b></span>" if latest else ""
    st.markdown(
        f"""
        <div class="source-strip">
          <span>Source: <a href="{esc(url)}" target="_blank" rel="noopener">{esc(source)}</a></span>
          {secondary_html}
          {latest_html}
          <span><b>Historical archive</b> · not a live feed</span>
          <span>Correlation ≠ causation</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_intro(
    *,
    latest: str,
    rows: str = "",
    countries: str = "",
    coverage: str = "",
    entity_label: str = "countries",
) -> None:
    meta_bits = []
    if coverage:
        meta_bits.append(coverage)
    if countries:
        meta_bits.append(f"{countries} {entity_label}")
    if rows:
        meta_bits.append(f"{rows} rows")
    meta = "<br>".join(meta_bits) if meta_bits else "Global + India historical data"
    st.sidebar.markdown(
        f"""
        <div class="side-intro">
          <div class="side-kicker">DATA SNAPSHOT</div>
          <div class="side-title">Historical archive</div>
          <div class="side-meta">Through <b>{esc(latest)}</b><br>{meta}</div>
          <div class="side-badges"><span>NOT LIVE</span><span>OWID</span><span>COVID19INDIA</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_label(text: str) -> None:
    st.sidebar.markdown(f'<div class="sidebar-label">{esc(text)}</div>', unsafe_allow_html=True)


def sidebar_note(text: str) -> None:
    st.sidebar.markdown(f'<div class="sidebar-note">{text}</div>', unsafe_allow_html=True)


def date_range_filter(
    df: pd.DataFrame,
    *,
    key: str = "analysis_date_range",
    label: str = "Analysis window",
    sidebar: bool = True,
    default_start: Any | None = None,
    default_end: Any | None = None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Shared date-range widget.

    A single session key is used by the Global, EDA, Comparative and Time Series
    pages so a window chosen on one page carries over to the others.  India has
    its own key because its archive covers a different period.
    """
    if df.empty or "date" not in df.columns:
        today = pd.Timestamp.today().normalize()
        return today - pd.Timedelta(days=365), today
    dates = pd.to_datetime(df["date"], errors="coerce").dropna()
    if dates.empty:
        today = pd.Timestamp.today().normalize()
        return today - pd.Timedelta(days=365), today
    min_date = dates.min().normalize().date()
    max_date = dates.max().normalize().date()
    start_default = pd.Timestamp(default_start).date() if default_start is not None else min_date
    end_default = pd.Timestamp(default_end).date() if default_end is not None else max_date
    start_default = max(min_date, min(start_default, max_date))
    end_default = max(start_default, min(end_default, max_date))

    target = st.sidebar if sidebar else st
    value = target.date_input(
        label,
        value=(start_default, end_default),
        min_value=min_date,
        max_value=max_date,
        key=key,
        format="DD/MM/YYYY",
        help="Applies to every analytical page that shares this window.",
    )
    if isinstance(value, (tuple, list)) and len(value) == 2:
        start, end = value
    elif isinstance(value, (tuple, list)) and len(value) == 1:
        start, end = value[0], max_date
    else:
        start, end = min_date, max_date
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    if start_ts > end_ts:
        start_ts, end_ts = end_ts, start_ts
    return start_ts, end_ts


def date_window_caption(start: Any, end: Any) -> str:
    return f"{format_date(start)} → {format_date(end)}"


def show_data_error(message: str, *, detail: str = "") -> None:
    st.error("The historical dataset is not available yet.")
    st.markdown(
        f"""
        <div class="callout tone-amber">
          <span class="callout-icon">{icon("alert", 15)}</span>
          <div>
            <div class="callout-title">Data file missing</div>
            <div class="callout-body">{esc(message)}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if detail:
        st.code(detail, language="bash")
    st.stop()
