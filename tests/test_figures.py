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


def test_born_digital_placeholder_without_figure_stays_placeholder(tmp_path):
    doc = pymupdf.open()
    doc.new_page(width=612, height=792).insert_text((72, 100), "Only text here.")
    path = tmp_path / "text.pdf"
    doc.save(str(path))
    md, images = figures.extract_figures(path, join_pages(["![figure](figure)"], preserve_pages=True))
    assert images == {} and "![figure](figure)" in md


# ── vlm-engine --full through convert_single ─────────────────────────────────

def _fake_vlm_ocr(markdown: str, events: list | None = None):
    def fake(pdf_path, settings, *, preserve_pages=False, cache_dir=None, stop=None,
             keep_cache=False):
        assert preserve_pages, "figure extraction needs page markers"
        assert keep_cache, "the page cache must outlive figure work"
        return OcrResult(markdown=markdown, page_markdowns=["", "", ""],
                         cleanup=lambda: events.append("cleanup") if events is not None else None)
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


def _vlm_settings(**kw) -> Settings:
    return Settings(api_base="https://h/v1", ocr_engine="vlm", vlm_model="ocr-m",
                    vlm_full_model="vision-m", **kw)


def test_page_cache_removed_only_after_output_written(pdf, tmp_path, monkeypatch):
    from pdfcancel import convert, llm

    events: list[str] = []
    monkeypatch.setattr(llm, "vlm_ocr_pdf", _fake_vlm_ocr(_ocr_markdown(), events))
    monkeypatch.setattr(llm, "chat_vision", lambda settings, **kw: "desc")
    real_write = convert.write_text_atomic
    monkeypatch.setattr(convert, "write_text_atomic",
                        lambda path, text: (events.append("write"), real_write(path, text)))
    convert.convert_single(pdf, _vlm_settings(), output_dir=tmp_path / "o", full=True,
                           extract_images=True)
    assert events[-1] == "cleanup" and events.count("cleanup") == 1 and "write" in events  # .md and manifest first


def test_page_cache_kept_when_describe_phase_dies(pdf, tmp_path, monkeypatch):
    from pdfcancel import convert, llm, multimodal

    events: list[str] = []
    monkeypatch.setattr(llm, "vlm_ocr_pdf", _fake_vlm_ocr(_ocr_markdown(), events))

    def boom(*a, **kw):
        raise RuntimeError("describer down")
    monkeypatch.setattr(multimodal, "describe_images", boom)
    with pytest.raises(RuntimeError):
        convert.convert_single(pdf, _vlm_settings(), output_dir=tmp_path / "o", full=True,
                               extract_images=True)
    assert "cleanup" not in events


def test_malformed_reply_is_a_per_image_failure(pdf, tmp_path, monkeypatch):
    """chat_vision's SystemExit on a bad reply must not abort the conversion."""
    from pdfcancel import convert, llm

    monkeypatch.setattr(llm, "vlm_ocr_pdf", _fake_vlm_ocr(_ocr_markdown()))
    calls = []

    def flaky(settings, *, image_bytes, prompt, **kw):
        calls.append(1)
        if len(calls) == 1:
            raise SystemExit("Error: unexpected chat/completions response shape")
        return "ok description"
    monkeypatch.setattr(llm, "chat_vision", flaky)
    out = convert.convert_single(pdf, _vlm_settings(ocr_concurrency=1), output_dir=tmp_path / "o",
                                 full=True, extract_images=True)
    assert out.read_text().count("**Figure description:** ok description") == 1


def test_failed_descriptions_are_never_cached(pdf, tmp_path, monkeypatch):
    from pdfcancel import convert, llm
    from pdfcancel.convert import load_manifest

    monkeypatch.setattr(llm, "vlm_ocr_pdf", _fake_vlm_ocr(_ocr_markdown()))

    def down(settings, **kw):
        raise RuntimeError("timeout")
    monkeypatch.setattr(llm, "chat_vision", down)
    out_dir = tmp_path / "o"
    convert.convert_single(pdf, _vlm_settings(), output_dir=out_dir, full=True, extract_images=True)
    assert "image_descriptions" not in load_manifest(out_dir)["doc"]

    # A failure cached by an older version is ignored, and the figure retried.
    manifest = load_manifest(out_dir)
    from pdfcancel.multimodal import cacheable_descriptions  # noqa: F401
    import hashlib
    _, images = figures.extract_figures(pdf, _ocr_markdown())
    manifest["doc"]["image_descriptions"] = {
        hashlib.sha256(b).hexdigest()[:16]: "[Image description unavailable: old]"
        for b in images.values()
    }
    from pdfcancel.convert import save_manifest
    save_manifest(out_dir, manifest)
    monkeypatch.setattr(llm, "chat_vision", lambda settings, **kw: "fresh")
    out = convert.convert_single(pdf, _vlm_settings(), output_dir=out_dir, full=True,
                                 extract_images=True, force=True)
    assert out.read_text().count("**Figure description:** fresh") == 2


def test_stop_during_descriptions_saves_partial_and_keeps_cache(pdf, tmp_path, monkeypatch):
    import threading

    from pdfcancel import convert, llm
    from pdfcancel.convert import load_manifest
    from pdfcancel.errors import ConversionIncomplete

    events: list[str] = []
    monkeypatch.setattr(llm, "vlm_ocr_pdf", _fake_vlm_ocr(_ocr_markdown(), events))
    stop = threading.Event()

    def first_then_stop(settings, **kw):
        stop.set()  # Ctrl-C arrives while the first figure is in flight
        return "first"
    monkeypatch.setattr(llm, "chat_vision", first_then_stop)
    out_dir = tmp_path / "o"
    with pytest.raises(ConversionIncomplete) as ei:
        convert.convert_single(pdf, _vlm_settings(ocr_concurrency=1), output_dir=out_dir,
                               full=True, extract_images=True, stop=stop)
    assert ei.value.interrupted and "1/2 figure(s)" in str(ei.value)
    assert list(load_manifest(out_dir)["doc"]["image_descriptions"].values()) == ["first"]
    assert "cleanup" not in events and not (out_dir / "doc.md").exists()


def test_vlm_descriptions_run_concurrently(monkeypatch):
    import threading
    import time

    from pdfcancel import llm, multimodal

    active = peak = 0
    lock = threading.Lock()

    def slow(settings, **kw):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        return "d"
    monkeypatch.setattr(llm, "chat_vision", slow)
    images = {f"p001-f{i:02d}.png": bytes([i]) * 8 for i in range(1, 9)}
    out = multimodal.describe_images(images, _vlm_settings(ocr_concurrency=4))
    assert set(out.values()) == {"d"} and len(out) == 8
    assert 1 < peak <= 4
