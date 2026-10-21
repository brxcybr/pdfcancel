from __future__ import annotations

from pdfcancel.multimodal import _split_description_data, inject_descriptions


def test_split_description_data_accepts_bold_marker_and_rule():
    prose, data = _split_description_data(
        "A ROC chart with an AUC value.\n\n---\n**DATA:**\n| Metric | Value |\n| --- | --- |\n| AUC | 0.94 |"
    )

    assert prose == "A ROC chart with an AUC value."
    assert "| AUC | 0.94 |" in data


def test_inject_descriptions_renders_structured_chart_json():
    markdown = "![chart](img-0.jpeg)\n"
    description = """A bar chart shows model accuracy.

DATA:
| Model | Accuracy |
| --- | --- |
| CNN | ~91.5 |

CHART_JSON:
{"figure_type":"bar_chart","title":"Model accuracy","x_axis":{"label":"Model"},"y_axis":{"label":"Accuracy","unit":"%"},"series":[{"name":"Accuracy","points":[{"x":"CNN","y":"~91.5","label":"CNN"}]}],"confidence":"estimated"}
"""

    result = inject_descriptions(markdown, {"img-0.jpeg": description})

    assert "> **Figure description:** A bar chart shows model accuracy." in result
    assert "> **Structured chart data:**" in result
    assert "| Series | X | Y | Label |" in result
    assert "pdfcancel-chart-data" in result
