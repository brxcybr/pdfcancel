"""Transcription repair (loops, LaTeX), text-layer hints, and grounded charts."""

from __future__ import annotations

from pathlib import Path

import pytest

from pdfcancel import llm
from pdfcancel.charts import ground_chart
from pdfcancel.config import Settings
from pdfcancel.multimodal import _build_prompt, inject_descriptions
from pdfcancel.vlmtext import LOOP_NOTE, clean_latex, collapse_loops, find_loop

LOOP = "[CTI Networking] is an " + "untapped area for a lot of " * 300 + "untapped"


# ── repetition loops ─────────────────────────────────────────────────────────

def test_finds_and_collapses_a_phrase_loop():
    text = "Intro words here.\n\n> " + LOOP
    assert find_loop(text)
    out = collapse_loops(text)
    assert out.count("untapped area for a lot of") == 1
    assert LOOP_NOTE in out and out.startswith("Intro words here.")
    assert len(out) < 200


def test_collapses_a_single_letter_loop():
    # A scanned reference list looped on an initial: "[5] N. T. S. S. S. ..."
    out = collapse_loops("[5] N. T. " + "S. " * 400 + "\n[6] Next ref.")
    assert out.count("S.") == 1 and out.endswith("[6] Next ref.")


def test_collapses_a_latex_spacing_loop():
    text = "Provider " + "$\\quad$ " * 200 + "Consumer"
    out = collapse_loops(text)
    assert out.count("\\quad") == 1 and out.endswith("Consumer")


@pytest.mark.parametrize("text", [
    "| $\\square$ " * 60 + "|",                           # a column of checkboxes
    " ".join(["0.0"] * 100),                              # a row of zeros
    "The results of the study show that " * 3 + "the end.",  # a few repeats
])
def test_legitimate_repetition_is_not_a_loop(text):
    assert not find_loop(text)


def _ocr_settings(**kw) -> Settings:
    return Settings(api_base="https://h/v1", ocr_engine="vlm", vlm_model="m",
                    ocr_concurrency=1, **kw)


def _run_ocr(monkeypatch, replies, *, settings=None, pdf=Path("x.pdf"), pages=1):
    settings = settings or _ocr_settings()
    calls = []

    fake_chat = _fake_chat(replies, calls)
    if pdf == Path("x.pdf"):
        monkeypatch.setattr(llm, "rasterize_pdf", lambda p, dpi=200: [b"img"] * pages)
    monkeypatch.setattr(llm, "chat_vision", fake_chat)
    monkeypatch.setattr(settings, "build_vision_http_client", lambda: _Closeable())
    return llm.vlm_ocr_pdf(pdf, settings), calls


class _Closeable:
    def close(self):
        pass


SOUP = "[27] A. Serrano, " + "lI " * 20 + "I " * 25 + "ee CC C ii d TT " * 4 + "WETICE 2014."


def _fake_chat(replies, calls):
    def fake(s, *, image_bytes, prompt, model, max_tokens, mime, http_client, history=None):
        calls.append({"prompt": prompt, "history": history})
        return replies[min(len(calls), len(replies)) - 1]
    return fake


def test_looping_page_is_retried(monkeypatch):
    result, calls = _run_ocr(monkeypatch, [LOOP, "Clean page."])
    assert result.markdown == "Clean page."
    assert [c["history"] for c in calls] == [None, None]


def test_persistent_loop_is_continued_from_its_clean_text(monkeypatch):
    first = "Refs [1] Alpha. [2] Beta, Gamma. " + "S. " * 300
    result, calls = _run_ocr(monkeypatch, [first, LOOP, "[3] C. [4] D. The end."])
    assert len(calls) == 3
    history = calls[2]["history"]
    assert history[0] == {"role": "assistant", "content": "Refs [1] Alpha. [2] Beta, Gamma."}
    assert history[1] == {"role": "user", "content": llm.VLM_CONTINUE_PROMPT}
    assert result.markdown == "Refs [1] Alpha. [2] Beta, Gamma.\n[3] C. [4] D. The end."


def test_continuations_are_bounded_and_the_rest_is_cut(monkeypatch):
    result, calls = _run_ocr(monkeypatch, [LOOP])  # every call loops
    assert len(calls) == 2 + 1  # stalled: a continuation that adds nothing ends it
    assert result.markdown.count("untapped area for a lot of") <= 2
    assert LOOP_NOTE in result.markdown


def test_letter_soup_counts_as_a_loop_and_is_dropped():
    assert find_loop(SOUP)
    out = collapse_loops(SOUP)
    assert out == "[27] A. Serrano, " + LOOP_NOTE + " WETICE 2014."


# ── LaTeX used for layout ────────────────────────────────────────────────────

@pytest.mark.parametrize("src, want", [
    ("$\\text{Accuracy}\\quad 0.92$", "Accuracy 0.92"),
    ("see $\\text{e.g.}$, this", "see e.g., this"),
    ("$\\text{S}\\text{O}\\text{F}$", "SOF"),
    ("losses of $\\text{\\$5}$ million", "losses of \\$5 million"),
    ("$\\textbf{Bold}$", "**Bold**"),
    ("$\\checkmark$ and $\\times$", "✓ and ×"),
    ("$$\\text{Total}\\qquad 42$$", "Total 42"),
    ("a $\\quad$ b", "a   b"),
])
def test_layout_latex_is_unwrapped(src, want):
    assert clean_latex(src) == want


@pytest.mark.parametrize("src", [
    "$x^2 + y^2 = z^2$",
    "$\\alpha$",
    "$N_d$",
    "costs $5 (~10%) or $6",
    "`$\\quad$` and\n```\necho $\\text{a}$\n```",
])
def test_real_math_prices_and_code_are_kept(src):
    assert clean_latex(src) == src


def test_spacing_inside_real_math_is_dropped():
    assert clean_latex("$\\alpha\\quad\\beta$") == "$\\alpha \\beta$"


def test_cached_pages_are_repaired_at_assembly(monkeypatch):
    result, _ = _run_ocr(monkeypatch, ["Score $\\text{high}$ here."])
    assert result.markdown == "Score high here."


# ── text-layer hints ─────────────────────────────────────────────────────────

pymupdf = pytest.importorskip("pymupdf")

BODY = ("Vulnerabilities are tracked in the NVD and classified by CWE, not CVE. " * 6).strip()


def _text_pdf(tmp_path: Path) -> Path:
    doc = pymupdf.open()
    doc.new_page().insert_textbox(pymupdf.Rect(72, 72, 540, 700), BODY, fontsize=10)
    doc.new_page()  # a blank (or scanned) page has no usable text layer
    path = tmp_path / "t.pdf"
    doc.save(str(path))
    return path


def test_born_digital_page_prompt_carries_its_text_layer(tmp_path, monkeypatch):
    _, calls = _run_ocr(monkeypatch, ["ok"], pdf=_text_pdf(tmp_path))
    first, second = (c["prompt"] for c in calls)
    assert "TEXT LAYER" in first and "classified by CWE, not CVE" in first
    assert second == llm.VLM_OCR_PROMPT


def test_text_layer_hint_can_be_disabled(tmp_path, monkeypatch):
    settings = _ocr_settings(vlm_text_layer=False)
    _, calls = _run_ocr(monkeypatch, ["ok"], settings=settings, pdf=_text_pdf(tmp_path))
    assert all(c["prompt"] == llm.VLM_OCR_PROMPT for c in calls)


def test_text_layer_env_switch(monkeypatch):
    monkeypatch.setenv("PDFCANCEL_VLM_TEXT_LAYER", "0")
    assert Settings().vlm_text_layer is False
    monkeypatch.setenv("PDFCANCEL_VLM_TEXT_LAYER", "yes")
    assert Settings().vlm_text_layer is True


@pytest.mark.parametrize("text", ["", "short", "�" * 300, "@#% $^& *() " * 40])
def test_unusable_text_layers_are_skipped(text):
    assert llm._usable_text_layer(text) is None


# ── descriptions ─────────────────────────────────────────────────────────────

def test_every_description_line_stays_in_the_blockquote():
    md = "Before.\n\n![p001-f01.png](p001-f01.png)\n\nAfter."
    desc = "A flowchart.\n\nIt has three stages:\n- Ingest\n- Enrich\n\nDATA:\n| a | b |\n|---|---|\n| 1 | 2 |"
    out = inject_descriptions(md, {"p001-f01.png": desc})
    block = out[out.index("> **Figure description:**"):out.index("After.")].strip()
    assert all(line.startswith(">") for line in block.splitlines())
    assert "> - Enrich" in block and "> | 1 | 2 |" in block


def test_printed_text_goes_into_the_description_prompt():
    prompt = _build_prompt("p001-f01.png", None, "North 42\nSouth 17")
    assert "TEXT PRINTED INSIDE THIS FIGURE" in prompt and "North 42\nSouth 17" in prompt
    assert "TEXT PRINTED" not in _build_prompt("p001-f01.png", None, None)


def _chart(*ys, confidence="exact"):
    return {"figure_type": "bar_chart", "confidence": confidence, "notes": "",
            "series": [{"name": "s", "points": [{"x": str(i), "y": y} for i, y in enumerate(ys)]}],
            "table": {"columns": [], "rows": []}}


def test_exact_chart_with_printed_values_stays_exact():
    assert ground_chart(_chart(7, 15, 10), "Venues 7 15 10")["confidence"] == "exact"
    assert ground_chart(_chart(0.33, 20), "33% 20%")["confidence"] == "exact"  # percentages


@pytest.mark.parametrize("chart, printed, reason", [
    (_chart(7, 15, 9), "7 15 10", "1 of 3 values are not printed"),
    (_chart(7, 15, 10), None, "could not be checked"),
    (_chart(7, "~15"), "7 15", "approximate"),
])
def test_unverified_exact_chart_is_downgraded(chart, printed, reason):
    out = ground_chart(chart, printed)
    assert out["confidence"] == "estimated" and reason in out["notes"]


def test_estimated_charts_are_untouched():
    chart = _chart(1, 2, confidence="estimated")
    assert ground_chart(chart, None) is chart


def test_body_text_after_a_description_stays_out_of_the_quote():
    md = "![p001-f01.png](p001-f01.png)\nNext paragraph."
    out = inject_descriptions(md, {"p001-f01.png": "A chart."})
    assert "> **Figure description:** A chart.\n\nNext paragraph." in out


def test_chunker_keeps_caption_with_its_described_figure():
    from pdfcancel.chunks import _protect_figure_blocks

    md = "![p001-f01.png](p001-f01.png)\nFigure 1: Sales.\n\nNext."
    out = inject_descriptions(md, {"p001-f01.png": "A chart."})
    (block,) = _protect_figure_blocks(out)[1].values()
    assert block.endswith("Figure 1: Sales.") and "Next." not in block
