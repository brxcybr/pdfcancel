"""Tests for local figure extraction (pdfcancel.figures) and vlm-engine --full."""

from __future__ import annotations

from pathlib import Path

import pytest

pymupdf = pytest.importorskip("pymupdf")

from pdfcancel import figures  # noqa: E402
from pdfcancel.config import Settings  # noqa: E402
from pdfcancel.images import OcrResult  # noqa: E402
from pdfcancel.pages import join_pages  # noqa: E402


def _png(w: int = 240, h: int = 160) -> bytes:
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, w, h), False)
    pix.set_rect(pix.irect, (40, 120, 200))
    return pix.tobytes("png")


def _make_pdf(path: Path) -> Path:
    """Three pages: a vector bar chart, a shaded code box, a raster figure."""
    doc = pymupdf.open()

    # Page 1: running header, vector bar chart with axis labels, body prose.
    p = doc.new_page(width=612, height=792)
    p.insert_text((72, 30), "J. Example et al. / Journal of Tests 1 (2026)", fontsize=8)
    for i, h in enumerate((80, 140, 110, 60)):
        x = 120 + i * 70
        p.draw_rect(pymupdf.Rect(x, 360 - h, x + 40, 360), color=(0, 0, 0), fill=(0.2, 0.2, 0.2))
    p.draw_line((110, 360), (420, 360))
    p.draw_line((110, 200), (110, 360))
    for i, label in enumerate(("Q1", "Q2", "Q3", "Q4")):
        p.insert_text((128 + i * 70, 374), label, fontsize=9)
    p.insert_text((70, 280), "Barrels", fontsize=9)
    p.insert_textbox(pymupdf.Rect(72, 420, 540, 700),
                     "Body prose that is long enough to be a paragraph. " * 12, fontsize=10)

    # Page 2: a shaded code box (vector fill + long text lines) — not a figure.
    p = doc.new_page(width=612, height=792)
    box = pymupdf.Rect(72, 200, 540, 420)
    p.draw_rect(box, color=None, fill=(0.92, 0.94, 0.98))
    for i in range(8):
        p.insert_text((80, 220 + i * 22),
                      f'"field_{i}": "a long example value for the code sample line {i}",',
                      fontsize=9)

    # Page 3: an embedded raster figure.
    p = doc.new_page(width=612, height=792)
    p.insert_image(pymupdf.Rect(150, 150, 450, 350), stream=_png())

    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture
def pdf(tmp_path):
    return _make_pdf(tmp_path / "doc.pdf")


def _ocr_markdown() -> str:
    return join_pages([
        "# Results\n\n![figure](figure)\n\nFigure 1: Output by quarter.",
        "## Example\n\n```json\n{}\n```",
        "![figure](figure)\n\nFigure 2: A photo.",
    ], preserve_pages=True)


def test_detects_vector_chart_with_labels_not_header(pdf):
    doc = pymupdf.open(str(pdf))
    rects, scanned = figures.detect_figures(doc[0])
    assert not scanned
    assert len(rects) == 1
    r = rects[0]
    assert r.contains(pymupdf.Rect(120, 220, 370, 360))  # the bars
    assert r.y1 >= 374 - 9  # grew to take in the Q1..Q4 tick labels
    assert r.x0 <= 70  # and the y-axis title
    assert r.y0 > 40  # but not the running header


def test_skips_shaded_code_box(pdf):
    doc = pymupdf.open(str(pdf))
    assert figures.detect_figures(doc[1])[0] == []


def test_detects_raster_image(pdf):
    doc = pymupdf.open(str(pdf))
    rects, _ = figures.detect_figures(doc[2])
    assert len(rects) == 1 and rects[0].contains(pymupdf.Rect(150, 150, 450, 350))


def test_extract_figures_links_placeholders_in_page_order(pdf):
    md, images = figures.extract_figures(pdf, _ocr_markdown())
    assert sorted(images) == ["p001-f01.png", "p003-f01.png"]
    assert all(b.startswith(b"\x89PNG") for b in images.values())
    assert "![figure](figure)" not in md
    page1, page3 = md.index("pdfcancel-page: 1"), md.index("pdfcancel-page: 3")
    assert page1 < md.index("![p001-f01.png](p001-f01.png)") < md.index("pdfcancel-page: 2")
    assert md.index("![p003-f01.png](p003-f01.png)") > page3
    assert "Figure 1: Output by quarter." in md


def test_extra_figures_append_to_their_page(pdf):
    md = join_pages(["# Results", "code", "no placeholder here"], preserve_pages=True)
    md, images = figures.extract_figures(pdf, md)
    assert md.rstrip().endswith("![p003-f01.png](p003-f01.png)")
    assert len(images) == 2


def test_scanned_placeholder_page_gets_full_page_render(tmp_path):
    doc = pymupdf.open()
    p = doc.new_page(width=612, height=792)
    p.insert_image(p.rect, stream=_png(612, 792))  # a "scan"
    path = tmp_path / "scan.pdf"
    doc.save(str(path))
    md, images = figures.extract_figures(path, join_pages(["![figure](figure)"], preserve_pages=True))
    assert list(images) == ["p001-page.png"]
    assert "![p001-page.png](p001-page.png)" in md


def test_repeated_logo_is_dropped(tmp_path):
    doc = pymupdf.open()
    for _ in range(3):
        doc.new_page(width=612, height=792).insert_image(
            pymupdf.Rect(150, 300, 450, 420), stream=_png())
    path = tmp_path / "logo.pdf"
    doc.save(str(path))
    _, images = figures.extract_figures(path, join_pages(["a", "b", "c"], preserve_pages=True))
    assert images == {}


def test_requires_page_markers(pdf):
    with pytest.raises(ValueError):
        figures.extract_figures(pdf, "no markers")


def test_split_pages_round_trips():
    md = "preamble\n\n" + _ocr_markdown()
    assert "".join(t for _, t in figures._split_pages(md)) == md
    assert [n for n, _ in figures._split_pages(md)] == [None, 1, 2, 3]


def test_strip_page_markers():
    out = figures.strip_page_markers(join_pages(["a", "b"], preserve_pages=True))
    assert "pdfcancel-page" not in out and out.split() == ["a", "b"]


# ── vlm-engine --full through convert_single ─────────────────────────────────

def _fake_vlm_ocr(markdown: str):
    def fake(pdf_path, settings, *, preserve_pages=False, cache_dir=None, stop=None):
        assert preserve_pages, "figure extraction needs page markers"
        return OcrResult(markdown=markdown, page_markdowns=["", "", ""])
    return fake


@pytest.mark.parametrize("preserve", [True, False])
def test_convert_single_vlm_full_crops_and_describes(pdf, tmp_path, monkeypatch, preserve):
    from pdfcancel import convert, llm

    monkeypatch.setattr(llm, "vlm_ocr_pdf", _fake_vlm_ocr(_ocr_markdown()))
    prompts: list[tuple[str, str]] = []

    def fake_chat(settings, *, image_bytes, prompt, model, max_tokens, mime, http_client):
        prompts.append((model, mime))
        return "A bar chart of output by quarter."
    monkeypatch.setattr(llm, "chat_vision", fake_chat)

    settings = Settings(api_base="https://h/v1", ocr_engine="vlm",
                        vlm_model="ocr-m", vlm_full_model="vision-m")
    out_dir = tmp_path / "out"
    out = convert.convert_single(pdf, settings, output_dir=out_dir, full=True, extract_images=True,
                                 preserve_pages=preserve)

    md = out.read_text()
    assert (out_dir / "doc_images" / "p001-f01.png").is_file()
    assert "](doc_images/p001-f01.png)" in md
    assert md.count("**Figure description:** A bar chart") == 2
    assert prompts == [("vision-m", "image/png")] * 2
    assert ("pdfcancel-page:" in md) is preserve


def test_cli_allows_full_with_vlm(monkeypatch, tmp_path, pdf):
    from click.testing import CliRunner

    from pdfcancel import cli

    seen = {}

    def fake_convert_single(path, settings, **kw):
        seen.update(engine=settings.ocr_engine, full=kw["full"], model=settings.vlm_full_model)
        out = tmp_path / "doc.md"
        out.write_text("ok")
        return out
    monkeypatch.setattr(cli, "convert_single", fake_convert_single)

    result = CliRunner().invoke(cli.main, [
        str(pdf), "--api-base", "https://h/v1", "--full", "--full-model", "vision-m",
    ])
    assert result.exit_code == 0, result.output
    assert seen == {"engine": "vlm", "full": True, "model": "vision-m"}
