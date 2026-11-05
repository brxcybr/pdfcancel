"""Heading levels from section numbering."""

from __future__ import annotations

from pdfcancel.clean import clean_markdown
from pdfcancel.headings import normalize_headings


def _heads(md: str) -> list[str]:
    return [ln for ln in md.splitlines() if ln.startswith("#")]


def test_decimal_levels_follow_depth_and_plain_sections_are_promoted():
    md = ("# A Survey of Things\n\n### 1. Introduction\n\nBody.\n\n"
          "1.1. Scope\n\nBody.\n\n**1.2. Approach**\n\nBody.\n\n"
          "# 2. Methods\n\nBody.\n\n#### 2.1. Data\n\nBody.\n\n2.1.1 Sources\n\nBody.")
    assert _heads(normalize_headings(md)) == [
        "# A Survey of Things", "## 1. Introduction", "### 1.1. Scope", "### 1.2. Approach",
        "## 2. Methods", "### 2.1. Data", "#### 2.1.1 Sources"]


def test_ieee_roman_and_letter_sections():
    md = ("# Title\n\n## I. INTRODUCTION\n\nText.\n\nII. METHODOLOGY\n\nText.\n\n"
          "A. The Model\n\nText.\n\nB. Criteria\n\nText.\n\nIII. RESULTS\n\nText.\n\n"
          "A. Enumerations\n\nText.\n\nREFERENCES\n\n[1] A. Smith, ...")
    assert _heads(normalize_headings(md)) == [
        "# Title", "## I. INTRODUCTION", "## II. METHODOLOGY", "### A. The Model",
        "### B. Criteria", "## III. RESULTS", "### A. Enumerations", "## REFERENCES"]


def test_unnumbered_headings_nest_under_their_section():
    md = ("## 4 Server API\n\n# 4.1 Discovery\n\n# Requests\n\nx\n\n"
          "### 4.2 Status\n\n## Examples\n\ny\n\n5 Collections\n\nz")
    assert _heads(normalize_headings(md)) == [
        "## 4 Server API", "### 4.1 Discovery", "#### Requests", "### 4.2 Status",
        "#### Examples", "## 5 Collections"]


def test_numbered_lists_and_body_lines_stay_put():
    md = ("## 1. Introduction\n\nText.\n\n## 2. Method\n\nSteps:\n\n"
          "1. Install the tool and run it once.\n\n2. Check the output.\n\n"
          "7. Unrelated numbered line\n\n## 3. Results\n\n2006 was a year.\n\n"
          "A. Smith wrote this.")
    out = normalize_headings(md)
    assert _heads(out) == ["## 1. Introduction", "## 2. Method", "## 3. Results"]
    assert "1. Install the tool" in out and "7. Unrelated numbered line" in out


def test_unnumbered_documents_are_untouched():
    md = "# Report\n\n### Findings\n\nText.\n\n# Annex\n\nMore."
    assert normalize_headings(md) == md


def test_page_markers_count_as_blank_and_cleanup_runs_it():
    md = ("## 1. Intro\n\nx\n\n## 2. Next\n\ny\n<!-- pdfcancel-page: 2 -->\n"
          "2.1. Detail\n\nz\n\n# 3. End\n\nw")
    assert _heads(clean_markdown(md)) == [
        "## 1. Intro", "## 2. Next", "### 2.1. Detail", "## 3. End"]
