from __future__ import annotations

from pdfcancel.multimodal import _split_description_data


def test_split_description_data_accepts_bold_marker_and_rule():
    prose, data = _split_description_data(
        "A ROC chart with an AUC value.\n\n---\n**DATA:**\n| Metric | Value |\n| --- | --- |\n| AUC | 0.94 |"
    )

    assert prose == "A ROC chart with an AUC value."
    assert "| AUC | 0.94 |" in data
