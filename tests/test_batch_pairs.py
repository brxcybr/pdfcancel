"""Tests for convert_batch (pdf, result) pairing (convert.py)."""

from __future__ import annotations

from pathlib import Path

from pdfcancel import convert as convert_mod
from pdfcancel.config import Settings


def test_convert_batch_returns_aligned_pairs_on_failure(monkeypatch, tmp_path):
    pdfs = [Path(f"/fake/{name}.pdf") for name in ("alpha", "beta", "gamma")]

    def fake_convert_single(pdf_path, settings, *, output_dir, **kwargs):
        if pdf_path.stem == "beta":
            raise RuntimeError("OCR exploded")
        return output_dir / f"{pdf_path.stem}.md"

    monkeypatch.setattr(convert_mod, "convert_single", fake_convert_single)

    pairs = convert_mod.convert_batch(
        pdfs, Settings(), output_dir=tmp_path,
    )

    assert len(pairs) == 3
    assert pairs[0] == (pdfs[0], tmp_path / "alpha.md")
    assert pairs[1] == (pdfs[1], None)  # failed file pairs with None
    assert pairs[2] == (pdfs[2], tmp_path / "gamma.md")

    # Successes never get misaligned with the wrong PDF name
    successes = [(pdf, out) for pdf, out in pairs if out is not None]
    for pdf, out in successes:
        assert pdf.stem == out.stem


def test_convert_batch_all_success(monkeypatch, tmp_path):
    pdfs = [Path("/fake/one.pdf"), Path("/fake/two.pdf")]

    monkeypatch.setattr(
        convert_mod,
        "convert_single",
        lambda pdf_path, settings, *, output_dir, **kw: output_dir / f"{pdf_path.stem}.md",
    )

    pairs = convert_mod.convert_batch(pdfs, Settings(), output_dir=tmp_path)
    assert [out for _, out in pairs] == [tmp_path / "one.md", tmp_path / "two.md"]
