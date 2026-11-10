"""Plotly and Matplotlib/Seaborn chart builders.

Charts share one visual language: transparent plot areas, muted grid lines, a
single accent colour per message and restrained hover labels.  No chart invents
data; empty filters are handled by the pages before calling these helpers.
"""
from __future__ import annotations

from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots

from src.utils import COLORS, COLOR_SEQUENCE, format_compact, format_date, format_pct

FONT_FAMILY = "Inter, Segoe UI, system-ui, sans-serif"
GRID = "#1B2940"
BG = "#0F1A2E"

COLORSCALES = {
    "blue": [[0.0, "#0E1B2E"], [0.18, "#123B63"], [0.45, "#1D4ED8"], [0.75, "#06B6D4"], [1.0, "#A5F3FC"]],
    "red": [[0.0, "#1A1118"], [0.3, "#7F1D1D"], [0.68, "#DC2626"], [1.0, "#FCA5A5"]],
    "green": [[0.0, "#0D1A17"], [0.35, "#14532D"], [0.7, "#22C55E"], [1.0, "#BBF7D0"]],
    "amber": [[0.0, "#1D160D"], [0.35, "#78350F"], [0.7, "#F59E0B"], [1.0, "#FDE68A"]],
    "violet": [[0.0, "#15102A"], [0.4, "#4C1D95"], [0.75, "#8B5CF6"], [1.0, "#DDD6FE"]],
}

METRIC_COLOR = {
    "total_cases": COLORS["blue"],
    "new_cases": COLORS["blue"],
    "new_cases_7d": COLORS["cyan"],
    "total_deaths": COLORS["red"],
    "new_deaths": COLORS["red"],
    "new_deaths_7d": COLORS["amber"],
    "total_cases_per_million": COLORS["cyan"],
    "total_deaths_per_million": COLORS["red"],
    "case_fatality_ratio": COLORS["amber"],
    "vaccination_rate": COLORS["green"],
    "fully_vaccinated_rate": COLORS["green"],
    "booster_rate": COLORS["violet"],
    "total_recovered": COLORS["green"],
    "active_cases": COLORS["amber"],
    "recovery_ratio": COLORS["green"],
}


def metric_color(metric: str) -> str:
    return METRIC_COLOR.get(metric, COLORS["blue"])


# --------------------------------------------------------------------------- #
# Plotly layout
# --------------------------------------------------------------------------- #
def apply_layout(
    fig: go.Figure,
    *,
    height: int = 360,
    hovermode: str = "x unified",
    showlegend: bool = False,
    legend_title: str | None = None,
    x_title: str | None = None,
    y_title: str | None = None,
    margin: dict[str, int] | None = None,
) -> go.Figure:
    fig.update_layout(
        template=None,
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_FAMILY, color=COLORS["text"], size=11),
        margin=margin or dict(l=8, r=14, t=18, b=8),
        hovermode=hovermode,
        showlegend=showlegend,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            x=0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=10, color="#A8B6CD"),
            title=dict(text=legend_title or "", font=dict(size=10)),
        ),
        hoverlabel=dict(
            bgcolor="#0D1728",
            bordercolor="#2C3D5B",
            font=dict(family=FONT_FAMILY, color=COLORS["text"], size=11),
            align="left",
        ),
        colorway=COLOR_SEQUENCE,
        separators=".,",
    )
    axis_style = dict(
        gridcolor=GRID,
        zeroline=False,
        linecolor=GRID,
        tickfont=dict(color="#8FA0B8", size=10),
        title_font=dict(color="#8FA0B8", size=10),
        showline=False,
    )
    fig.update_xaxes(**axis_style)
    fig.update_yaxes(**axis_style)
    if x_title:
        fig.update_xaxes(title_text=x_title)
    if y_title:
        fig.update_yaxes(title_text=y_title)
    return fig


def _date_ticks(fig: go.Figure) -> None:
    fig.update_xaxes(
        tickformat="%b %Y",
        hoverformat="%d %b %Y",
        ticks="outside",
        ticklen=3,
        tickcolor=GRID,
    )


# --------------------------------------------------------------------------- #
# Time-series charts
# --------------------------------------------------------------------------- #
def trend_chart(
    df: pd.DataFrame,
    *,
    y: str,
    label: str,
    color: str | None = None,
    height: int = 330,
    fill: bool = True,
    show_legend: bool = False,
    x: str = "date",
) -> go.Figure:
    color = color or metric_color(y)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df[x],
            y=df[y],
            mode="lines",
            name=label,
            line=dict(color=color, width=2.2),
            fill="tozeroy" if fill else None,
            fillcolor=_rgba(color, 0.10) if fill else None,
            hovertemplate=f"<b>%{{x|%d %b %Y}}</b><br>{label}: %{{y:,.0f}}<extra></extra>",
        )
    )
    apply_layout(fig, height=height, showlegend=show_legend, y_title=label)
    _date_ticks(fig)
    return fig


def dual_axis_trend(
    df: pd.DataFrame,
    *,
    cases_col: str = "total_cases",
    deaths_col: str = "total_deaths",
    cases_label: str = "Confirmed cases",
    deaths_label: str = "Deaths",
    height: int = 360,
) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df[cases_col],
            name=cases_label,
            mode="lines",
            line=dict(color=COLORS["blue"], width=2.2),
            fill="tozeroy",
            fillcolor=_rgba(COLORS["blue"], 0.08),
            hovertemplate=f"<b>%{{x|%d %b %Y}}</b><br>{cases_label}: %{{y:,.0f}}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df[deaths_col],
            name=deaths_label,
            mode="lines",
            line=dict(color=COLORS["red"], width=2.0, dash="dot"),
            hovertemplate=f"<b>%{{x|%d %b %Y}}</b><br>{deaths_label}: %{{y:,.0f}}<extra></extra>",
        ),
        secondary_y=True,
    )
    apply_layout(fig, height=height, showlegend=True, hovermode="x unified")
    _date_ticks(fig)
    fig.update_yaxes(title_text=cases_label, secondary_y=False, title_font=dict(size=10, color="#8FA0B8"))
    fig.update_yaxes(title_text=deaths_label, secondary_y=True, title_font=dict(size=10, color="#8FA0B8"))
    fig.update_yaxes(showgrid=False, secondary_y=True)
    return fig


def daily_chart(
    df: pd.DataFrame,
    *,
    daily_col: str,
    rolling_col: str,
    label: str,
    rolling_label: str = "7-day average",
    color: str | None = None,
    height: int = 340,
    revision_color: str = COLORS["red"],
) -> go.Figure:
    color = color or metric_color(daily_col)
    values = pd.to_numeric(df[daily_col], errors="coerce")
    bar_colors = np.where(values.fillna(0) < 0, revision_color, color)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df["date"],
            y=values,
            name=label,
            marker=dict(color=bar_colors, line=dict(width=0)),
            opacity=0.58,
            hovertemplate=f"<b>%{{x|%d %b %Y}}</b><br>{label}: %{{y:,.0f}}<extra></extra>",
        )
    )
    if rolling_col in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df[rolling_col],
                name=rolling_label,
                mode="lines",
                line=dict(color="#F8FAFC", width=2.0),
                hovertemplate=f"<b>%{{x|%d %b %Y}}</b><br>{rolling_label}: %{{y:,.0f}}<extra></extra>",
            )
        )
    fig.add_hline(y=0, line_color="#334155", line_width=1)
    apply_layout(fig, height=height, showlegend=True, hovermode="x unified")
    _date_ticks(fig)
    fig.update_yaxes(tickformat="~s", title_text=label)
    return fig


def monthly_bar_chart(
    monthly: pd.DataFrame,
    *,
    value_col: str,
    label: str,
    color: str = COLORS["blue"],
    height: int = 330,
) -> go.Figure:
    """Monthly totals as a restrained vertical bar chart."""
    fig = go.Figure(
        go.Bar(
            x=monthly["month"],
            y=monthly[value_col],
            marker_color=color,
            marker_line_width=0,
            opacity=0.82,
            hovertemplate=f"<b>%{{x|%B %Y}}</b><br>{label}: %{{y:,.0f}}<extra></extra>",
        )
    )
    apply_layout(fig, height=height, hovermode="x unified", y_title=label)
    _date_ticks(fig)
    fig.update_xaxes(tickformat="%b %Y")
    fig.update_yaxes(tickformat="~s")
    return fig


def sparkline(df: pd.DataFrame, y: str, *, color: str = COLORS["cyan"], height: int = 84) -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            x=df["date"],
            y=df[y],
            mode="lines",
            line=dict(color=color, width=2),
            fill="tozeroy",
            fillcolor=_rgba(color, 0.12),
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=4, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False, fixedrange=True),
        yaxis=dict(visible=False, fixedrange=True),
        showlegend=False,
    )
    return fig


# --------------------------------------------------------------------------- #
# Ranking / comparison bars
# --------------------------------------------------------------------------- #
def ranking_bar(
    snapshot: pd.DataFrame,
    *,
    metric: str,
    label: str,
    n: int = 10,
    name_col: str = "country",
    colorscale: str = "blue",
    height: int | None = None,
    value_suffix: str = "",
    percent: bool = False,
) -> go.Figure:
    # Plotly draws the first category at the bottom of a horizontal bar chart.
    # Ascending order therefore places the largest bar at the top.
    data = snapshot.dropna(subset=[metric]).sort_values(metric, ascending=True).tail(n)
    values = data[metric].astype(float).tolist()
    names = data[name_col].astype(str).tolist()
    if percent:
        text = [format_pct(v, 2) for v in values]
        hover = "<b>%{customdata[0]}</b><br>" + label + ": %{customdata[1]}<extra></extra>"
        custom = [[name, format_pct(value, 2)] for name, value in zip(names, values)]
    else:
        text = [format_compact(v) for v in values]
        hover = "<b>%{customdata[0]}</b><br>" + label + ": %{customdata[1]}<extra></extra>"
        custom = [[name, format_compact(value)] for name, value in zip(names, values)]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=[_shorten(name, 25) for name in names],
            orientation="h",
            marker=dict(
                color=values,
                colorscale=COLORSCALES.get(colorscale, COLORSCALES["blue"]),
                line=dict(color="rgba(0,0,0,0)", width=0),
            ),
            text=text,
            textposition="outside",
            textfont=dict(size=9, color="#C8D5E8"),
            cliponaxis=False,
            customdata=custom,
            hovertemplate=hover,
        )
    )
    apply_layout(
        fig,
        height=height or max(330, 33 * len(values) + 75),
        hovermode="closest",
        margin=dict(l=8, r=54, t=8, b=8),
        x_title=label,
    )
    fig.update_xaxes(tickformat="~s", showgrid=True)
    fig.update_yaxes(showgrid=False, tickfont=dict(size=10, color="#CBD5E1"))
    fig.update_traces(marker_line_width=0)
    return fig


def comparison_lines(
    df: pd.DataFrame,
    *,
    metric: str,
    label: str,
    name_col: str = "country",
    height: int = 380,
    x_col: str = "date",
    x_title: str | None = None,
) -> go.Figure:
    fig = go.Figure()
    for index, (name, group) in enumerate(df.groupby(name_col, sort=False)):
        color = COLOR_SEQUENCE[index % len(COLOR_SEQUENCE)]
        x_hover = "%{x|%d %b %Y}" if x_col == "date" else "Day %{x}"
        fig.add_trace(
            go.Scatter(
                x=group[x_col],
                y=group[metric],
                mode="lines",
                name=str(name),
                line=dict(color=color, width=2.1),
                hovertemplate=(
                    f"<b>{name}</b><br>{x_hover}<br>{label}: %{{y:,.0f}}<extra></extra>"
                ),
            )
        )
    apply_layout(
        fig,
        height=height,
        showlegend=True,
        hovermode="x unified",
        y_title=label,
        x_title=x_title,
    )
    if x_col == "date":
        _date_ticks(fig)
    else:
        fig.update_xaxes(title_text=x_title or "Days since threshold")
    return fig


def simple_bar(
    labels: Sequence[str],
    values: Sequence[float],
    *,
    label: str,
    color: str = COLORS["blue"],
    height: int = 340,
    percent: bool = False,
    horizontal: bool = True,
) -> go.Figure:
    text = [format_pct(v, 2) if percent else format_compact(v) for v in values]
    if horizontal:
        fig = go.Figure(
            go.Bar(
                x=list(values),
                y=list(labels),
                orientation="h",
                marker_color=color,
                text=text,
                textposition="outside",
                cliponaxis=False,
                textfont=dict(size=9, color="#C8D5E8"),
                hovertemplate="<b>%{y}</b><br>" + label + ": %{x:,.0f}<extra></extra>",
            )
        )
        fig.update_xaxes(tickformat="~s")
        fig.update_yaxes(showgrid=False)
    else:
        fig = go.Figure(
            go.Bar(
                x=list(labels),
                y=list(values),
                marker_color=color,
                text=text,
                textposition="outside",
                cliponaxis=False,
                textfont=dict(size=9, color="#C8D5E8"),
                hovertemplate="<b>%{x}</b><br>" + label + ": %{y:,.0f}<extra></extra>",
            )
        )
        fig.update_yaxes(tickformat="~s")
    apply_layout(fig, height=height, hovermode="closest", y_title=label if horizontal else None)
    return fig


# --------------------------------------------------------------------------- #
# Maps
# --------------------------------------------------------------------------- #
def world_map(
    snapshot: pd.DataFrame,
    *,
    metric: str,
    label: str,
    colorscale: str = "blue",
    height: int = 520,
    percent: bool = False,
) -> go.Figure:
    data = snapshot.copy()
    data["iso3"] = data["iso_code"].astype(str).str.upper()
    data = data[data["iso3"].str.fullmatch(r"[A-Z]{3}", na=False)]
    data = data.dropna(subset=[metric])
    if percent:
        formatted = data[metric].map(lambda value: format_pct(value, 2))
        unit = "% of population"
        zmax = 100
    else:
        formatted = data[metric].map(format_compact)
        unit = label
        zmax = None
    fig = go.Figure(
        go.Choropleth(
            locations=data["iso3"],
            z=data[metric],
            locationmode="ISO-3",
            colorscale=COLORSCALES.get(colorscale, COLORSCALES["blue"]),
            zmin=0,
            zmax=zmax,
            marker=dict(line=dict(color="#0B1220", width=0.45)),
            customdata=np.column_stack([data["country"], formatted]),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>" + label + ": %{customdata[1]}<extra></extra>"
            ),
            colorbar=dict(
                title=dict(text=unit, font=dict(size=9, color="#8FA0B8")),
                thickness=9,
                len=0.66,
                outlinewidth=0,
                tickfont=dict(size=8, color="#8FA0B8"),
                bgcolor="rgba(0,0,0,0)",
            ),
        )
    )
    apply_layout(
        fig,
        height=height,
        hovermode="closest",
        margin=dict(l=0, r=0, t=0, b=0),
    )
    fig.update_geos(
        projection_type="natural earth",
        showframe=False,
        showcoastlines=False,
        showcountries=True,
        countrycolor="#22314A",
        countrywidth=0.45,
        showland=True,
        landcolor="#111827",
        showocean=True,
        oceancolor="#0B1220",
        showlakes=False,
        bgcolor="rgba(0,0,0,0)",
    )
    return fig


def india_map(
    mapped: pd.DataFrame,
    geojson: dict[str, Any],
    *,
    metric: str,
    label: str,
    colorscale: str = "blue",
    height: int = 545,
    percent: bool = False,
) -> go.Figure:
    features = [feature["properties"]["STNAME"] for feature in geojson.get("features", [])]
    if percent:
        formatted = mapped[metric].map(lambda value: format_pct(value, 2))
    else:
        formatted = mapped[metric].map(format_compact)

    fig = go.Figure()
    # Neutral base so states without a matching value are still visible.
    fig.add_trace(
        go.Choropleth(
            geojson=geojson,
            locations=features,
            z=[0] * len(features),
            featureidkey="properties.STNAME",
            colorscale=[[0, "#17243A"], [1, "#17243A"]],
            showscale=False,
            marker=dict(line=dict(color="#2B3E5C", width=0.7)),
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Choropleth(
            geojson=geojson,
            locations=mapped["geo_state"],
            z=mapped[metric],
            featureidkey="properties.STNAME",
            colorscale=COLORSCALES.get(colorscale, COLORSCALES["blue"]),
            zmin=0,
            marker=dict(line=dict(color="#0B1220", width=0.7)),
            customdata=np.column_stack([mapped["state"], formatted]),
            hovertemplate="<b>%{customdata[0]}</b><br>" + label + ": %{customdata[1]}<extra></extra>",
            colorbar=dict(
                title=dict(text=label, font=dict(size=9, color="#8FA0B8")),
                thickness=9,
                len=0.62,
                outlinewidth=0,
                tickfont=dict(size=8, color="#8FA0B8"),
                bgcolor="rgba(0,0,0,0)",
            ),
        )
    )
    apply_layout(fig, height=height, hovermode="closest", margin=dict(l=0, r=0, t=0, b=0))
    fig.update_geos(
        fitbounds="locations",
        visible=False,
        bgcolor="rgba(0,0,0,0)",
    )
    return fig


# --------------------------------------------------------------------------- #
# Heatmaps
# --------------------------------------------------------------------------- #
def monthly_heatmap(matrix: pd.DataFrame, *, label: str, height: int = 300) -> go.Figure:
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    columns = [month for month in range(1, 13) if month in matrix.columns]
    values = matrix.reindex(columns=columns)
    text = [
        [("" if pd.isna(value) else format_compact(value, 1)) for value in row]
        for row in values.to_numpy()
    ]
    fig = go.Figure(
        go.Heatmap(
            z=values.to_numpy(),
            x=[month_names[month - 1] for month in columns],
            y=[str(year) for year in values.index],
            colorscale=COLORSCALES["blue"],
            showscale=False,
            text=text,
            texttemplate="%{text}",
            textfont=dict(size=8, color="#D7E3F4"),
            hovertemplate="<b>%{y} %{x}</b><br>" + label + ": %{z:,.0f}<extra></extra>",
            xgap=3,
            ygap=3,
            hoverongaps=False,
        )
    )
    apply_layout(fig, height=height, hovermode="closest", margin=dict(l=8, r=8, t=8, b=8))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False, autorange="reversed")
    return fig


def _mpl_figure(width: float = 8.2, height: float = 4.4) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(width, height), facecolor=BG)
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_color("#22314A")
        spine.set_linewidth(0.7)
    ax.tick_params(colors="#8FA0B8", labelsize=8)
    ax.grid(color=GRID, linewidth=0.6, alpha=0.55)
    ax.set_axisbelow(True)
    return fig, ax


def distribution_plot(
    series: pd.Series,
    *,
    label: str,
    log_scale: bool = False,
    bins: int = 40,
    color: str = COLORS["blue"],
) -> plt.Figure:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if log_scale:
        clean = clean[clean > 0]
    fig, ax = _mpl_figure(8.0, 4.0)
    if clean.empty:
        ax.text(0.5, 0.5, "No valid observations", ha="center", va="center", color="#94A3B8")
        ax.set_axis_off()
        return fig
    sns.histplot(
        clean,
        bins=min(bins, max(10, int(np.sqrt(len(clean))))),
        kde=len(clean) > 20,
        color=color,
        edgecolor="#0B1220",
        linewidth=0.4,
        log_scale=log_scale,
        alpha=0.75,
        ax=ax,
    )
    median = clean.median()
    ax.axvline(median, color=COLORS["amber"], linewidth=1.2, linestyle="--", label=f"Median: {format_compact(median)}")
    ax.set_xlabel(label, color="#94A3B8", fontsize=9)
    ax.set_ylabel("Frequency", color="#94A3B8", fontsize=9)
    legend = ax.legend(frameon=False, fontsize=8, loc="upper right")
    for text in legend.get_texts():
        text.set_color("#CBD5E1")
    if log_scale:
        ax.set_xlabel(f"{label} (log scale)", color="#94A3B8", fontsize=9)
    fig.tight_layout()
    return fig


def box_plot(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    label: str,
    log_scale: bool = False,
    height: float = 4.2,
) -> plt.Figure:
    data = df[[x, y]].copy()
    data[y] = pd.to_numeric(data[y], errors="coerce")
    data = data.dropna()
    fig, ax = _mpl_figure(8.4, height)
    if data.empty:
        ax.text(0.5, 0.5, "No valid observations", ha="center", va="center", color="#94A3B8")
        ax.set_axis_off()
        return fig
    order = data.groupby(x)[y].median().sort_values(ascending=False).index.tolist()
    palette = COLOR_SEQUENCE + COLOR_SEQUENCE
    sns.boxplot(
        data=data,
        x=x,
        y=y,
        order=order,
        hue=x,
        palette=palette[: len(order)],
        legend=False,
        linewidth=0.8,
        fliersize=2,
        ax=ax,
    )
    if log_scale:
        positive = data[y][data[y] > 0]
        if not positive.empty:
            ax.set_yscale("log")
            ax.set_ylabel(f"{label} (log scale)", color="#94A3B8", fontsize=9)
        else:
            ax.set_ylabel(label, color="#94A3B8", fontsize=9)
    else:
        ax.set_ylabel(label, color="#94A3B8", fontsize=9)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=30)
    for label_obj in ax.get_xticklabels():
        label_obj.set_horizontalalignment("right")
    fig.tight_layout()
    return fig


def correlation_plot(corr: pd.DataFrame, *, title: str = "") -> plt.Figure:
    size = max(5.0, 0.62 * len(corr.columns) + 2.4)
    fig, ax = _mpl_figure(size, size * 0.82)
    if corr.empty:
        ax.text(0.5, 0.5, "Not enough observations", ha="center", va="center", color="#94A3B8")
        ax.set_axis_off()
        return fig
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    cmap = sns.diverging_palette(8, 220, as_cmap=True)
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmin=-1,
        vmax=1,
        center=0,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 8, "color": "#E2E8F0"},
        linewidths=0.5,
        linecolor=BG,
        square=True,
        cbar_kws={"shrink": 0.62, "pad": 0.02},
        ax=ax,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=38, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    ax.tick_params(colors="#8FA0B8")
    cbar = ax.collections[0].colorbar
    if cbar is not None:
        cbar.ax.tick_params(colors="#8FA0B8", labelsize=7)
        cbar.outline.set_edgecolor("#22314A")
    fig.tight_layout()
    return fig


def scatter_plot(
    snapshot: pd.DataFrame,
    *,
    x: str,
    y: str,
    x_label: str,
    y_label: str,
    size: str = "population",
    color: str = "continent",
    log_x: bool = True,
    log_y: bool = True,
    height: int = 430,
) -> go.Figure:
    needed = [x, y]
    data = snapshot.dropna(subset=needed).copy()
    if size not in data.columns:
        data[size] = 1
    data[size] = pd.to_numeric(data[size], errors="coerce").fillna(1).clip(lower=1)
    if color not in data.columns:
        data[color] = "Country"
    data[color] = data[color].fillna("Other").astype(str)
    if log_x:
        data = data[data[x] > 0]
    if log_y:
        data = data[data[y] > 0]
    if data.empty:
        return apply_layout(go.Figure(), height=height)
    fig = px.scatter(
        data,
        x=x,
        y=y,
        size=size,
        color=color,
        size_max=42,
        color_discrete_sequence=COLOR_SEQUENCE,
        custom_data=["country", "population", "total_cases"],
        log_x=log_x,
        log_y=log_y,
    )
    fig.update_traces(
        marker=dict(line=dict(width=0.5, color="#0B1220"), opacity=0.82),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>" + x_label + ": %{x:,.2f}<br>"
            + y_label + ": %{y:,.2f}<extra></extra>"
        ),
    )
    apply_layout(
        fig,
        height=height,
        hovermode="closest",
        showlegend=True,
        x_title=x_label,
        y_title=y_label,
        margin=dict(l=8, r=10, t=26, b=8),
    )
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    return fig


def period_comparison_bars(
    labels: Sequence[str],
    current: Sequence[float],
    previous: Sequence[float],
    *,
    height: int = 320,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=list(labels),
            y=list(previous),
            name="Previous period",
            marker_color="#2B3E5C",
            hovertemplate="<b>%{x}</b><br>Previous: %{y:,.0f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=list(labels),
            y=list(current),
            name="Selected period",
            marker_color=COLORS["cyan"],
            hovertemplate="<b>%{x}</b><br>Selected: %{y:,.0f}<extra></extra>",
        )
    )
    apply_layout(fig, height=height, hovermode="x unified", showlegend=True)
    fig.update_layout(barmode="group", bargap=0.32, bargroupgap=0.12)
    fig.update_yaxes(tickformat="~s")
    return fig


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _rgba(hex_color: str, alpha: float) -> str:
    value = hex_color.lstrip("#")
    if len(value) != 6:
        return f"rgba(59,130,246,{alpha})"
    r, g, b = (int(value[index : index + 2], 16) for index in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _shorten(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def style_matplotlib() -> None:
    """Apply the Matplotlib defaults used by every EDA figure."""
    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": BG,
            "savefig.facecolor": BG,
            "text.color": COLORS["text"],
            "axes.labelcolor": "#94A3B8",
            "xtick.color": "#8FA0B8",
            "ytick.color": "#8FA0B8",
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#22314A",
            "grid.color": GRID,
        }
    )
