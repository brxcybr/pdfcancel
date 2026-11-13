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


def test_cleanup_drops_placeholders_no_crop_took():
    from pdfcancel.clean import clean_markdown

    md = ("Intro.\n\n![figure](figure)\n\n![MANDIANT](figure) Analyst Program "
          "![chart](figure) Framework\n\nEnd.")
    assert clean_markdown(md) == "Intro.\n\nAnalyst Program Framework\n\nEnd."


# ── vlm-engine --full through convert_single ─────────────────────────────────

def _fake_vlm_ocr(markdown: str, events: list | None = None):
    def fake(pdf_path, settings, *, preserve_pages=False, cache_dir=None, stop=None,
             keep_cache=False, stem=None):
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
    # Each figure: a data pass on the OCR model, then the description.
    assert sorted(prompts) == [("ocr-m", "image/png")] * 2 + [("vision-m", "image/png")] * 2
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
    out = convert.convert_single(pdf, _vlm_settings(ocr_concurrency=1, vlm_chart_pass=False),
                                 output_dir=tmp_path / "o", full=True, extract_images=True)
    assert out.read_text().count("**Figure description:** ok description") == 1


CHART_REPLY = ('Legend: A dark, B light. CHART_JSON: {"figure_type": "pie_chart", '
               '"series": [{"name": "share", "points": [{"x": "A", "y": 60}, '
               '{"x": "B", "y": 40}]}], "confidence": "exact"}')


def test_chart_pass_values_ground_the_description(pdf, tmp_path, monkeypatch):
    from pdfcancel import convert, llm

    monkeypatch.setattr(llm, "vlm_ocr_pdf", _fake_vlm_ocr(_ocr_markdown()))
    seen = []

    def fake_chat(settings, *, image_bytes, prompt, model, **kw):
        seen.append((model, len(image_bytes), prompt))
        if model == "ocr-m":
            return CHART_REPLY
        return ("A pie chart: A 60%, B 40%.\n\nCHART_JSON:\n"
                '{"figure_type": "pie_chart", "series": [{"name": "s", "points": '
                '[{"x": "A", "y": 40}]}]}')
    monkeypatch.setattr(llm, "chat_vision", fake_chat)
    out = convert.convert_single(pdf, _vlm_settings(ocr_concurrency=1), output_dir=tmp_path / "o",
                                 full=True, extract_images=True)
    md = out.read_text()
    data = [s for s in seen if s[0] == "ocr-m"]
    described = [s for s in seen if s[0] == "vision-m"]
    assert data and all(d[1] > v[1] for d, v in zip(data, described))  # sharper render
    assert all('"y": 60' in p for _, _, p in described)  # the describer sees the values
    assert "| A | 60 |" in md.replace("  ", " ") or "A | 60" in md
    assert '"y": 40}]}' not in md  # the describer's own guess is replaced


def test_non_chart_figures_keep_the_describers_data(pdf, tmp_path, monkeypatch):
    from pdfcancel import convert, llm

    monkeypatch.setattr(llm, "vlm_ocr_pdf", _fake_vlm_ocr(_ocr_markdown()))

    def fake_chat(settings, *, model, prompt, **kw):
        if model == "ocr-m":
            return 'CHART_JSON: {"figure_type": "other"}'
        assert "ALREADY READ" not in prompt
        return "A flowchart."
    monkeypatch.setattr(llm, "chat_vision", fake_chat)
    out = convert.convert_single(pdf, _vlm_settings(), output_dir=tmp_path / "o",
                                 full=True, extract_images=True)
    assert "**Figure description:** A flowchart." in out.read_text()


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


def test_backlog_passes_full_through(monkeypatch, tmp_path, pdf):
    from click.testing import CliRunner

    from pdfcancel import cli

    calls = []

    def fake_convert_single(path, settings, **kw):
        calls.append((kw["full"], kw["extract_images"], settings.vlm_full_model))
        out = kw["output_dir"] / f"{path.stem}.md"
        out.write_text("ok")
        return out
    monkeypatch.setattr(cli, "convert_single", fake_convert_single)

    result = CliRunner().invoke(cli.main, [
        "backlog", str(pdf.parent), "--api-base", "https://h/v1",
        "--full", "--full-model", "vision-m",
    ])
    assert result.exit_code == 0, result.output
    assert calls == [(True, True, "vision-m")]


def test_skips_api_callout_with_monospace_text(tmp_path):
    """An HTTP/JSON callout: method badge + decorations + short monospace lines."""
    doc = pymupdf.open()
    p = doc.new_page(width=612, height=792)
    box = pymupdf.Rect(72, 200, 540, 460)
    p.draw_rect(box, color=(0.4, 0.5, 0.8), fill=(0.93, 0.95, 1.0))
    p.draw_rect(pymupdf.Rect(80, 208, 120, 224), fill=(0.2, 0.4, 0.8), color=None)
    p.insert_text((86, 220), "GET", fontsize=9, color=(1, 1, 1))
    for i in range(30):  # small decorative marks, away from the text
        x = 300 + (i % 10) * 20
        y = 430 + (i // 10) * 8
        p.draw_rect(pymupdf.Rect(x, y, x + 6, y + 3), fill=(0.6, 0.6, 0.8), color=None)
    for i, line in enumerate(["/{api-root}/collections/", "Accept: application/taxii+json",
                              "200 - OK", '"id": "91a7b528"', '"title": "Indicators"']):
        p.insert_text((90, 250 + i * 24), line, fontsize=9, fontname="cour")
    path = tmp_path / "spec.pdf"
    doc.save(str(path))
    assert figures.detect_figures(pymupdf.open(str(path))[0])[0] == []


def test_skips_callout_with_one_long_monospace_line(tmp_path):
    """A request callout: badge, path, and one sentence of monospace prose."""
    doc = pymupdf.open()
    p = doc.new_page(width=612, height=792)
    p.draw_rect(pymupdf.Rect(72, 200, 540, 330), color=(0.4, 0.5, 0.8), fill=(0.93, 0.95, 1.0))
    p.draw_rect(pymupdf.Rect(80, 208, 120, 224), fill=(0.2, 0.4, 0.8), color=None)
    p.insert_text((86, 220), "GET", fontsize=9, color=(1, 1, 1))
    p.insert_text((130, 220), "/{api-root}/collections/", fontsize=11, fontname="cour")
    p.insert_text((90, 260), "Implementation Notes", fontsize=11)
    p.insert_text((100, 285), "Get information about all collections", fontsize=9, fontname="cour")
    path = tmp_path / "callout.pdf"
    doc.save(str(path))
    assert figures.detect_figures(pymupdf.open(str(path))[0])[0] == []


def test_white_boxes_behind_text_lines_are_not_a_figure(tmp_path):
    """Some exporters paint a white box behind every line; clustered, they look like a figure."""
    doc = pymupdf.open()
    p = doc.new_page(width=612, height=792)
    for i in range(6):  # boxes behind parts of lines; the text runs past them
        y = 200 + i * 12.2
        p.draw_rect(pymupdf.Rect(72 if i % 2 else 260, y - 10, 430, y + 2.2), color=None,
                    fill=(1, 1, 1))
        p.insert_text((72, y), "Body text over a white highlight, as a word processor "
                      "exported it, one line after another line.", fontsize=10)
    path = tmp_path / "white.pdf"
    doc.save(str(path))
    assert figures.detect_figures(pymupdf.open(str(path))[0])[0] == []


def test_crop_takes_in_tick_labels_drawn_as_outlines(tmp_path):
    """Text exported as glyph paths is too small to cluster but is still a label."""
    doc = pymupdf.open()
    p = doc.new_page(width=612, height=792)
    p.draw_polyline([(200, 400), (260, 330), (320, 360), (380, 280)], color=(0, 0, 0))
    for y in range(280, 401, 30):  # gridlines
        p.draw_line((200, y), (400, y), color=(0.8, 0.8, 0.8))
    for i, y in enumerate(range(400, 279, -30)):  # "0".."4" as 3 pt glyph outlines
        p.draw_rect(pymupdf.Rect(186, y - 2, 189, y + 2), fill=(0.4, 0.4, 0.4), color=None)
    path = tmp_path / "outlined.pdf"
    doc.save(str(path))
    (crop,) = figures.detect_figures(pymupdf.open(str(path))[0])[0]
    assert crop.x0 < 186


def test_keeps_vector_chart_with_proportional_labels(pdf):
    # The page-1 chart's labels use a proportional font; it must survive.
    assert len(figures.detect_figures(pymupdf.open(str(pdf))[0])[0]) == 1


def test_keeps_diagram_with_monospace_node_labels(tmp_path):
    """Two connected nodes labelled in monospace are a diagram, not a code listing."""
    doc = pymupdf.open()
    p = doc.new_page(width=612, height=792)
    for x, label in ((100, "client_process"), (360, "server_process")):
        p.draw_rect(pymupdf.Rect(x, 250, x + 150, 320), color=(0, 0, 0), width=1)
        p.insert_text((x + 22, 290), label, fontsize=10, fontname="cour")
    p.draw_line((250, 285), (360, 285))
    p.draw_polyline([(350, 280), (360, 285), (350, 290)])
    path = tmp_path / "diagram.pdf"
    doc.save(str(path))
    assert len(figures.detect_figures(pymupdf.open(str(path))[0])[0]) == 1


def _chart_then_photo_pdf(path: Path, *, captions: bool) -> Path:
    doc = pymupdf.open()
    p = doc.new_page(width=612, height=792)
    for i, h in enumerate((60, 110, 80)):  # vector bar chart with labels
        x = 150 + i * 80
        p.draw_rect(pymupdf.Rect(x, 300 - h, x + 50, 300), color=(0, 0, 0), fill=(0.3, 0.3, 0.3))
    p.draw_line((140, 300), (420, 300))
    for i, label in enumerate(("North", "South", "West")):
        p.insert_text((155 + i * 80, 314), label, fontsize=9)
    p.insert_image(pymupdf.Rect(170, 420, 440, 600), stream=_png())  # a photograph
    if captions:
        p.insert_text((150, 345), "Figure 1: Sales by region.", fontsize=9)
        p.insert_text((170, 625), "Figure 2: The warehouse.", fontsize=9)
    doc.save(str(path))
    return path


def test_text_chart_does_not_take_the_photos_placeholder(tmp_path):
    path = _chart_then_photo_pdf(tmp_path / "cp.pdf", captions=True)
    md = join_pages(["North South West\n\nFigure 1: Sales by region.\n\n"
                     "![figure](figure)\n\nFigure 2: The warehouse."], preserve_pages=True)
    md, images = figures.extract_figures(path, md)
    assert len(images) == 2
    chart, photo = "![p001-f01.png](p001-f01.png)", "![p001-f02.png](p001-f02.png)"
    assert md.index(chart) < md.index("Figure 1:") < md.index(photo) < md.index("Figure 2:")
    assert "![figure](figure)" not in md


def test_uncaptioned_placeholder_goes_to_the_pictorial_crop(tmp_path):
    path = _chart_then_photo_pdf(tmp_path / "cp.pdf", captions=False)
    md = join_pages(["North South West\n\n![figure](figure)\n\nThe end."], preserve_pages=True)
    md, _ = figures.extract_figures(path, md)
    photo, chart = "![p001-f02.png](p001-f02.png)", "![p001-f01.png](p001-f01.png)"
    assert md.index(photo) < md.index("The end.") < md.index(chart)  # chart appended


def test_placeholder_with_alt_text_takes_the_photo(tmp_path):
    """Models sometimes fill in the alt text: ``![warehouse](figure)``."""
    path = _chart_then_photo_pdf(tmp_path / "cp.pdf", captions=False)
    md = join_pages(["North South West\n\n![warehouse](figure)\n\nThe end."],
                    preserve_pages=True)
    md, _ = figures.extract_figures(path, md)
    assert md.index("![p001-f02.png](p001-f02.png)") < md.index("The end.")
    assert "(figure)" not in md


# ── Scans, panels, and empty boxes (figure noise, round 2) ───────────────────

def _scan_page(doc, *, ocr: bool, box=(60, 60, 550, 640)):
    p = doc.new_page(width=612, height=792)
    p.insert_image(pymupdf.Rect(*box), stream=_png(400, 480))
    if ocr:  # an OCR layer: invisible text over the scan
        p.insert_textbox(pymupdf.Rect(*box), "Scanned words of a book chapter. " * 40,
                         fontsize=10, render_mode=3)
    return p


def test_scan_with_margins_and_an_ocr_layer_is_not_a_figure(tmp_path):
    doc = pymupdf.open()
    _scan_page(doc, ocr=True)  # 70% of the page: below the full-page test
    path = tmp_path / "scan.pdf"
    doc.save(str(path))
    rects, scanned = figures.detect_figures(pymupdf.open(str(path))[0])
    assert rects == [] and scanned


def test_scanned_document_without_text_has_no_figures(tmp_path):
    doc = pymupdf.open()
    for _ in range(3):
        _scan_page(doc, ocr=False)
    path = tmp_path / "scans.pdf"
    doc.save(str(path))
    doc = pymupdf.open(str(path))
    assert figures.scanned_document(doc)
    assert all(figures.detect_figures(p, scan_doc=True)[0] == [] for p in doc)


def test_a_full_page_picture_in_a_born_digital_document_is_kept(tmp_path):
    doc = pymupdf.open()
    for _ in range(3):
        doc.new_page(width=612, height=792).insert_textbox(
            pymupdf.Rect(72, 72, 540, 700), "Body prose of a born-digital report. " * 30)
    _scan_page(doc, ocr=False)  # one text-less page holding a large picture
    path = tmp_path / "report.pdf"
    doc.save(str(path))
    doc = pymupdf.open(str(path))
    assert not figures.scanned_document(doc)
    assert len(figures.detect_figures(doc[3], scan_doc=False)[0]) == 1


def test_tail_of_a_code_box_is_not_a_figure(tmp_path):
    """A code box split across pages leaves a frame and a closing bracket."""
    doc = pymupdf.open()
    p = doc.new_page(width=612, height=792)
    p.draw_rect(pymupdf.Rect(72, 72, 540, 114), color=None, fill=(0.91, 0.94, 0.96))
    for r in ((72, 72, 73, 114), (540, 72, 541, 114), (72, 114, 541, 115)):  # frame rules
        p.draw_rect(pymupdf.Rect(*r), color=None, fill=(0.3, 0.47, 0.72))
    p.insert_text((90, 90), "]", fontsize=8, fontname="cour")
    p.insert_text((80, 104), "}", fontsize=8, fontname="cour")
    path = tmp_path / "tail.pdf"
    doc.save(str(path))
    assert figures.detect_figures(pymupdf.open(str(path))[0])[0] == []


def _cards_page(doc):
    p = doc.new_page(width=612, height=792)
    for i in range(8):  # bordered cards of running text, close enough to cluster
        x, y = 72 + (i % 2) * 226, 120 + (i // 2) * 136
        p.draw_rect(pymupdf.Rect(x, y, x + 220, y + 130), color=(0.2, 0.6, 0.3), width=1)
        p.insert_textbox(pymupdf.Rect(x + 8, y + 8, x + 212, y + 122),
                         "Each card holds a few sentences of advice on the practice. " * 3,
                         fontsize=9)


def test_cards_of_running_text_are_not_a_figure(tmp_path):
    doc = pymupdf.open()
    _cards_page(doc)
    path = tmp_path / "cards.pdf"
    doc.save(str(path))
    assert figures.detect_figures(pymupdf.open(str(path))[0])[0] == []


def test_bar_chart_with_sentence_labels_is_kept(tmp_path):
    """A Likert chart labels its rows with statements; its bars hold no text."""
    doc = pymupdf.open()
    p = doc.new_page(width=612, height=792)
    for i in range(6):
        y = 150 + i * 22
        p.insert_text((72, y + 12), f"Statement {i} that respondents rated on the scale",
                      fontsize=9)
        x = 330
        for k, w in enumerate((10, 25, 30, 40, 60)):
            p.draw_rect(pymupdf.Rect(x, y, x + w, y + 16), color=None,
                        fill=(0.9 - 0.15 * k, 0.6, 0.3 + 0.1 * k))
            x += w
    p.draw_line((395, 140), (395, 290), color=(0.5, 0.5, 0.5))  # the neutral axis
    path = tmp_path / "likert.pdf"
    doc.save(str(path))
    assert len(figures.detect_figures(pymupdf.open(str(path))[0])[0]) == 1


def test_an_empty_box_is_not_a_figure(tmp_path):
    doc = pymupdf.open()
    p = doc.new_page(width=612, height=792)
    p.draw_rect(pymupdf.Rect(100, 200, 400, 400), color=None, fill=(0.93, 0.93, 0.93))
    path = tmp_path / "empty.pdf"
    doc.save(str(path))
    assert figures.detect_figures(pymupdf.open(str(path))[0])[0] == []
