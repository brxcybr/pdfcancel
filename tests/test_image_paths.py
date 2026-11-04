"""Image links written for --images must survive spaces and parentheses in PDF names."""

from __future__ import annotations

import re

import pytest

from pdfcancel.images import process_images_from_raw
from pdfcancel.multimodal import inject_descriptions

STEMS = [
    "plain",
    "Cascavilla - 2021 - Cybercrime Threat Intelligence",   # spaces
    "978-3-319-18302-2_1 (1)",                              # parentheses
    "Strom - 2017 - ATT&CK #2 [draft] 50%",                 # &, #, brackets, %
]


def _link_target(md: str) -> str:
    m = re.search(r"!\[[^\]]*\]\(([^)\s]+)\)", md)
    assert m, f"no well-formed image link in {md!r}"
    return m.group(1)


@pytest.mark.parametrize("stem", STEMS)
def test_extracted_link_is_well_formed_and_resolves(tmp_path, stem):
    md = process_images_from_raw(
        "Before\n\n![img-0.png](img-0.png)\n\nAfter",
        images={"img-0.png": b"\x89PNG fake"},
        stem=stem, output_dir=tmp_path, extract=True,
    )
    target = _link_target(md)
    from urllib.parse import unquote
    assert (tmp_path / unquote(target)).read_bytes() == b"\x89PNG fake"


@pytest.mark.parametrize("stem", STEMS)
def test_descriptions_inject_after_extraction(tmp_path, stem):
    md = process_images_from_raw(
        "![img-0.png](img-0.png)\n\nFigure 1.",
        images={"img-0.png": b"x"}, stem=stem, output_dir=tmp_path, extract=True,
    )
    out = inject_descriptions(md, {"img-0.png": "A bar chart."})
    assert "> **Figure description:** A bar chart." in out
