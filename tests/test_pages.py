"""Tests for page marker injection (pages.py) and preservation (clean.py)."""

from __future__ import annotations

from pdfcancel.clean import clean_markdown
from pdfcancel.pages import (
    PAGE_MARKER_RE,
    build_page_index,
    is_page_marker_line,
    join_pages,
    page_marker,
    page_range_for_span,
    strip_page_markers,
)


def test_join_pages_without_preserve_is_plain_join():
    pages = ["page one", "page two"]
    assert join_pages(pages) == "page one\n\npage two"
    assert join_pages(pages, preserve_pages=False) == "page one\n\npage two"


def test_join_pages_injects_one_indexed_markers():
    pages = ["alpha", "beta", "gamma"]
    joined = join_pages(pages, preserve_pages=True)
    assert joined.startswith("<!-- pdfcancel-page: 1 -->\n\nalpha")
    assert "<!-- pdfcancel-page: 2 -->\n\nbeta" in joined
    assert "<!-- pdfcancel-page: 3 -->\n\ngamma" in joined
    assert len(PAGE_MARKER_RE.findall(joined)) == 3


def test_is_page_marker_line():
    assert is_page_marker_line("<!-- pdfcancel-page: 7 -->")
    assert is_page_marker_line("  <!-- pdfcancel-page: 7 -->  ")
    assert not is_page_marker_line("<!-- pdfcancel-chart-data: {} -->")
    assert not is_page_marker_line("regular text")


def test_build_page_index_and_span_lookup():
    text = join_pages(["one two three", "four five", "six"], preserve_pages=True)
    index = build_page_index(text)
    assert [p for _, p in index] == [1, 2, 3]

    # Span entirely inside page 1
    start = text.find("one")
    assert page_range_for_span(index, start, start + 3) == (1, 1)

    # Span crossing from page 1 into page 2
    end = text.find("five") + 4
    assert page_range_for_span(index, start, end) == (1, 2)

    # Span entirely in page 3
    start3 = text.find("six")
    assert page_range_for_span(index, start3, start3 + 3) == (3, 3)

    # Content before the first marker attributes to page 1
    assert page_range_for_span(index, 0, 1) == (1, 1)

    # No markers at all
    assert page_range_for_span([], 0, 100) == (None, None)


def test_strip_page_markers():
    text = join_pages(["alpha", "beta"], preserve_pages=True)
    stripped = strip_page_markers(text)
    assert "pdfcancel-page" not in stripped
    assert "alpha" in stripped and "beta" in stripped
    assert "\n\n\n" not in stripped


def test_clean_markdown_preserves_markers_across_all_passes():
    # Many pages so the markers' shared 20-char prefix would trip the
    # repeating-header heuristic without explicit protection.
    pages = []
    for n in range(1, 8):
        pages.append(
            f"Journal of Testing\n\nSome content for page {n}. "
            f"More sentences here.\n\n{n}"
        )
    text = join_pages(pages, preserve_pages=True)
    cleaned = clean_markdown(text)

    markers = PAGE_MARKER_RE.findall(cleaned)
    assert [int(m) for m in markers] == list(range(1, 8))

    # Each marker still sits alone on its own line (never merged into prose)
    for line in cleaned.split("\n"):
        if "pdfcancel-page" in line:
            assert is_page_marker_line(line), f"marker corrupted: {line!r}"

    # Cleanup still works around the markers: repeating header removed,
    # bare page numbers removed
    assert "Journal of Testing" not in cleaned
    assert "\n3\n" not in cleaned


def test_clean_markdown_does_not_merge_sentences_across_markers():
    text = (
        "<!-- pdfcancel-page: 1 -->\n\n"
        "This sentence was cut at the page boundary and\n\n"
        "<!-- pdfcancel-page: 2 -->\n\n"
        "continues on the next page with lowercase text.\n"
    )
    cleaned = clean_markdown(text)
    assert "<!-- pdfcancel-page: 1 -->" in cleaned
    assert "<!-- pdfcancel-page: 2 -->" in cleaned
    # The marker must not be glued into the broken sentence
    for line in cleaned.split("\n"):
        if "pdfcancel-page" in line:
            assert is_page_marker_line(line)


def test_clean_markdown_default_behavior_unchanged_without_markers():
    text = (
        "Some heading content\n\n"
        "A paragraph of real text that ends properly.\n\n"
        "42\n\n"
        "Another paragraph follows here.\n"
    )
    cleaned = clean_markdown(text)
    # Bare page number still stripped by default
    assert "\n42\n" not in cleaned
    assert "A paragraph of real text" in cleaned


def test_page_marker_template_roundtrip():
    m = page_marker(12)
    assert m == "<!-- pdfcancel-page: 12 -->"
    match = PAGE_MARKER_RE.fullmatch(m)
    assert match and match.group(1) == "12"
