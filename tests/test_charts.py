from __future__ import annotations

from pdfcancel.charts import (
    build_vega_lite_spec,
    extract_chart_metadata,
    render_chart_data_markdown,
    split_chart_json,
)


def test_split_chart_json_normalizes_series_and_table():
    description = """A bar chart compares model accuracy.

CHART_JSON:
{
  "figure_type": "bar_chart",
  "title": "Model accuracy",
  "x_axis": {"label": "Model", "unit": ""},
  "y_axis": {"label": "Accuracy", "unit": "%"},
  "series": [
    {"name": "Accuracy", "points": [{"x": "CNN", "y": "~91.5", "label": "CNN"}]}
  ],
  "table": {
    "columns": ["Model", "Accuracy"],
    "rows": [{"Model": "CNN", "Accuracy": "~91.5"}]
  },
  "confidence": "estimated",
  "notes": "Values estimated from the chart"
}
"""

    prose, chart = split_chart_json(description)

    assert prose == "A bar chart compares model accuracy."
    assert chart is not None
    assert chart["figure_type"] == "bar_chart"
    assert chart["table"]["rows"][0]["Model"] == "CNN"
    assert chart["series"][0]["points"][0]["y"] == "~91.5"


def test_render_chart_data_markdown_and_extract_metadata_comment_roundtrip():
    _, chart = split_chart_json("""Chart.

CHART_JSON:
{"figure_type":"line_chart","title":"Loss","series":[{"name":"loss","points":[{"x":"1","y":0.5}]}],"confidence":"exact"}
""")

    rendered = render_chart_data_markdown(chart)
    spec = build_vega_lite_spec(chart)

    assert "> **Structured chart data:**" in rendered
    assert "| Series | X | Y | Label |" in rendered
    assert spec is not None
    assert spec["mark"] == "line"
    assert spec["data"]["values"][0]["y"] == 0.5

    text = (
        "<!-- pdfcancel-chart-data: "
        '{"chart_data":{"figure_type":"line_chart","series":[{"name":"loss","points":[{"x":"1","y":0.5}]}],"table":{"columns":[],"rows":[]},"confidence":"exact"},'
        '"vega_lite_spec":{"mark":"line"}} -->'
    )
    meta = extract_chart_metadata(text)

    assert meta["has_structured_chart_data"] is True
    assert meta["chart_data"]["figure_type"] == "line_chart"
    assert meta["vega_lite_spec"]["mark"] == "line"

