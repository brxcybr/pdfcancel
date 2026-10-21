"""Structured chart and table extraction helpers for figure descriptions.

The vision model is allowed to return a strict JSON block alongside the prose
figure description. This module parses, validates, renders, and derives
Vega-Lite specs from that JSON without making network calls itself.
"""

from __future__ import annotations

import json
import re
from typing import Any


CHART_JSON_START = "CHART_JSON:"
_CHART_COMMENT_RE = re.compile(
    r"<!--\s*pdfcancel-chart-data:\s*(.*?)\s*-->",
    re.DOTALL,
)


def chart_prompt_instructions() -> str:
    """Return prompt instructions for structured chart extraction."""
    return (
        "\n\nIf the figure contains a chart, graph, plotted metric, confusion "
        "matrix, statistical table, or other structured visual data, append a "
        "section starting with CHART_JSON: on a new line. The section must be "
        "valid JSON only, with this shape:\n"
        "{\n"
        '  "figure_type": "bar_chart|line_chart|scatter_plot|table|matrix|diagram|other",\n'
        '  "title": "visible or inferred title",\n'
        '  "x_axis": {"label": "", "unit": ""},\n'
        '  "y_axis": {"label": "", "unit": ""},\n'
        '  "series": [{"name": "", "points": [{"x": "", "y": 0, "label": ""}]}],\n'
        '  "table": {"columns": ["Column"], "rows": [{"Column": "Value"}]},\n'
        '  "confidence": "exact|estimated|low",\n'
        '  "notes": "brief caveats about approximate values"\n'
        "}\n"
        "Use null for unknown numeric values. Use strings for categorical "
        "labels. Omit CHART_JSON entirely if there is no extractable "
        "structured data. Do not wrap the JSON in markdown fences."
    )


def split_chart_json(description: str) -> tuple[str, dict[str, Any] | None]:
    """Split a model response into text and optional structured chart JSON."""
    parts = re.split(
        rf"\n\s*(?:---+\s*\n)?\s*\*{{0,2}}{CHART_JSON_START}\*{{0,2}}\s*\n",
        description,
        maxsplit=1,
        flags=re.IGNORECASE,
    )
    if len(parts) == 1:
        return description.strip(), None

    text = parts[0].strip()
    json_text = _extract_json_object(parts[1].strip())
    if not json_text:
        return text, None

    try:
        raw = json.loads(json_text)
    except json.JSONDecodeError:
        return text, None

    chart = normalize_chart_data(raw)
    return text, chart if chart else None


def normalize_chart_data(raw: Any) -> dict[str, Any]:
    """Validate and normalize model-supplied chart data."""
    if not isinstance(raw, dict):
        return {}

    chart: dict[str, Any] = {
        "figure_type": _clean_string(raw.get("figure_type")) or "other",
        "title": _clean_string(raw.get("title")),
        "x_axis": _normalize_axis(raw.get("x_axis")),
        "y_axis": _normalize_axis(raw.get("y_axis")),
        "series": _normalize_series(raw.get("series")),
        "table": _normalize_table(raw.get("table")),
        "confidence": _normalize_confidence(raw.get("confidence")),
        "notes": _clean_string(raw.get("notes")),
    }

    if not chart["series"] and not chart["table"]["rows"]:
        return {}
    return chart


def render_chart_data_markdown(chart: dict[str, Any] | None) -> str:
    """Render structured chart data as searchable blockquoted markdown."""
    if not chart:
        return ""

    lines = ["> **Structured chart data:**"]
    summary_parts = []
    if chart.get("figure_type"):
        summary_parts.append(str(chart["figure_type"]))
    if chart.get("title"):
        summary_parts.append(str(chart["title"]))
    if chart.get("confidence"):
        summary_parts.append(f"confidence={chart['confidence']}")
    if summary_parts:
        lines.append("> " + "; ".join(summary_parts))

    table = chart.get("table") or {}
    rows = table.get("rows") or []
    columns = table.get("columns") or []
    if rows and columns:
        lines.append(">")
        lines.extend(f"> {line}" for line in _markdown_table(columns, rows))

    series_rows = _series_rows(chart.get("series") or [])
    if series_rows:
        lines.append(">")
        lines.extend(
            f"> {line}"
            for line in _markdown_table(
                ["Series", "X", "Y", "Label"],
                series_rows,
            )
        )

    if chart.get("notes"):
        lines.append(f"> Notes: {chart['notes']}")

    return "\n".join(lines)


def build_vega_lite_spec(chart: dict[str, Any] | None) -> dict[str, Any] | None:
    """Build an approximate Vega-Lite spec from normalized chart data."""
    if not chart:
        return None

    values = _vega_values(chart)
    if not values:
        return None

    figure_type = str(chart.get("figure_type") or "").lower()
    mark = "bar"
    if "line" in figure_type:
        mark = "line"
    elif "scatter" in figure_type:
        mark = "point"
    elif "area" in figure_type:
        mark = "area"
    elif "matrix" in figure_type or "heat" in figure_type:
        mark = "rect"

    spec: dict[str, Any] = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "description": chart.get("title") or "Extracted chart data",
        "data": {"values": values},
        "mark": mark,
        "encoding": {
            "x": {"field": "x", "type": "nominal", "title": _axis_title(chart, "x_axis")},
            "y": {"field": "y", "type": "quantitative", "title": _axis_title(chart, "y_axis")},
        },
        "usermeta": {
            "pdfcancel": {
                "confidence": chart.get("confidence", "low"),
                "notes": chart.get("notes", ""),
                "approximate": chart.get("confidence") != "exact",
            }
        },
    }
    if any("series" in row for row in values):
        spec["encoding"]["color"] = {"field": "series", "type": "nominal"}
    return spec


def chart_data_to_metadata(chart: dict[str, Any] | None) -> dict[str, Any]:
    """Return chunk metadata additions for extracted chart data."""
    if not chart:
        return {}
    spec = build_vega_lite_spec(chart)
    return {
        "has_structured_chart_data": True,
        "chart_data": chart,
        "vega_lite_spec": spec,
    }


def chart_data_comment(chart: dict[str, Any] | None) -> str:
    """Serialize chart data as a hidden markdown comment for chunk metadata."""
    if not chart:
        return ""
    payload = {
        "chart_data": chart,
        "vega_lite_spec": build_vega_lite_spec(chart),
    }
    return f"<!-- pdfcancel-chart-data: {json.dumps(payload, ensure_ascii=False)} -->"


def extract_chart_metadata(text: str) -> dict[str, Any]:
    """Extract hidden chart metadata from a markdown chunk."""
    charts = []
    specs = []
    for match in _CHART_COMMENT_RE.finditer(text):
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        chart = payload.get("chart_data")
        spec = payload.get("vega_lite_spec")
        if isinstance(chart, dict):
            charts.append(chart)
        if isinstance(spec, dict):
            specs.append(spec)
    if not charts:
        return {}
    return {
        "has_structured_chart_data": True,
        "chart_data": charts[0] if len(charts) == 1 else charts,
        "vega_lite_spec": specs[0] if len(specs) == 1 else specs,
    }


def _extract_json_object(text: str) -> str:
    start = text.find("{")
    if start == -1:
        return ""
    depth = 0
    in_string = False
    escaped = False
    for idx, char in enumerate(text[start:], start=start):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return ""


def _clean_string(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _normalize_axis(raw: Any) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {"label": "", "unit": ""}
    return {
        "label": _clean_string(raw.get("label")),
        "unit": _clean_string(raw.get("unit")),
    }


def _normalize_confidence(raw: Any) -> str:
    value = _clean_string(raw).lower()
    return value if value in {"exact", "estimated", "low"} else "low"


def _normalize_series(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    series = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        points = item.get("points")
        if not isinstance(points, list):
            continue
        norm_points = []
        for point in points:
            if not isinstance(point, dict):
                continue
            norm_points.append({
                "x": _clean_string(point.get("x")),
                "y": _normalize_number(point.get("y")),
                "label": _clean_string(point.get("label")),
            })
        if norm_points:
            series.append({
                "name": _clean_string(item.get("name")),
                "points": norm_points,
            })
    return series


def _normalize_table(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {"columns": [], "rows": []}
    rows = raw.get("rows")
    if not isinstance(rows, list):
        return {"columns": [], "rows": []}

    columns = raw.get("columns")
    if not isinstance(columns, list):
        columns = []
    norm_columns = [_clean_string(col) for col in columns if _clean_string(col)]

    norm_rows = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        clean_row = {str(k): _normalize_cell(v) for k, v in row.items()}
        if clean_row:
            norm_rows.append(clean_row)
            for col in clean_row:
                if col not in norm_columns:
                    norm_columns.append(col)

    return {"columns": norm_columns, "rows": norm_rows}


def _normalize_number(value: Any) -> int | float | str | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    text = _clean_string(value).replace(",", "")
    if not text:
        return ""
    approximate = text.startswith("~")
    raw = text[1:] if approximate else text
    try:
        num: int | float
        num = float(raw) if "." in raw else int(raw)
        return f"~{num}" if approximate else num
    except ValueError:
        return text


def _normalize_cell(value: Any) -> str | int | float | None:
    number = _normalize_number(value)
    if number is None:
        return None
    return number


def _markdown_table(columns: list[str], rows: list[dict[str, Any]]) -> list[str]:
    safe_columns = [col or "Value" for col in columns]
    lines = [
        "| " + " | ".join(safe_columns) + " |",
        "| " + " | ".join("---" for _ in safe_columns) + " |",
    ]
    for row in rows:
        values = [_format_cell(row.get(col, "")) for col in safe_columns]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _series_rows(series: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in series:
        name = item.get("name", "")
        for point in item.get("points", []):
            rows.append({
                "Series": name,
                "X": point.get("x", ""),
                "Y": point.get("y", ""),
                "Label": point.get("label", ""),
            })
    return rows


def _format_cell(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("|", "\\|")


def _vega_values(chart: dict[str, Any]) -> list[dict[str, Any]]:
    values = []
    for item in chart.get("series") or []:
        for point in item.get("points", []):
            y = point.get("y")
            if isinstance(y, str) and y.startswith("~"):
                y = y[1:]
            try:
                y_value = float(y)
            except (TypeError, ValueError):
                continue
            row = {"x": point.get("x") or point.get("label") or "", "y": y_value}
            if item.get("name"):
                row["series"] = item["name"]
            if point.get("label"):
                row["label"] = point["label"]
            values.append(row)
    if values:
        return values

    table = chart.get("table") or {}
    columns = table.get("columns") or []
    if len(columns) < 2:
        return []
    x_col, y_col = columns[0], columns[1]
    for row in table.get("rows") or []:
        try:
            y_value = float(str(row.get(y_col, "")).lstrip("~"))
        except ValueError:
            continue
        values.append({"x": row.get(x_col, ""), "y": y_value})
    return values


def _axis_title(chart: dict[str, Any], key: str) -> str:
    axis = chart.get(key) or {}
    label = axis.get("label") or ("X" if key == "x_axis" else "Y")
    unit = axis.get("unit")
    return f"{label} ({unit})" if unit else label
