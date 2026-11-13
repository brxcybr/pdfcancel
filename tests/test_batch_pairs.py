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


# ── output names: same-named PDFs in different folders ───────────────────────

def _pdf(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def test_output_stems_keep_unique_names(tmp_path):
    pdfs = [_pdf(tmp_path / "a" / "one.pdf", b"1"), _pdf(tmp_path / "b" / "two.pdf", b"2")]
    assert convert_mod.output_stems(pdfs) == {pdfs[0]: "one", pdfs[1]: "two"}


def test_output_stems_separate_different_pdfs_with_one_name(tmp_path):
    first = _pdf(tmp_path / "academic" / "Moller 2020.pdf", b"paper")
    other = _pdf(tmp_path / "doctrine" / "Moller 2020.pdf", b"another document")
    copy = _pdf(tmp_path / "archive" / "Moller 2020.pdf", b"paper")  # identical bytes
    stems = convert_mod.output_stems([first, copy, other])
    assert stems[first] == "Moller 2020"
    assert stems[copy] == "Moller 2020"  # same document: skipped as unchanged
    assert stems[other] == "Moller 2020 (doctrine)"


def test_output_stems_fall_back_to_a_hash_when_the_folder_name_is_taken(tmp_path):
    a = _pdf(tmp_path / "x" / "r.pdf", b"a")
    b = _pdf(tmp_path / "y" / "lib" / "r.pdf", b"b")
    c = _pdf(tmp_path / "z" / "lib" / "r.pdf", b"c")
    stems = convert_mod.output_stems([a, b, c])
    assert stems[b] == "r (lib)"
    assert stems[c].startswith("r (") and stems[c] not in ("r", "r (lib)")


def test_convert_batch_writes_same_named_pdfs_separately(monkeypatch, tmp_path):
    pdfs = [_pdf(tmp_path / "a" / "same.pdf", b"1"), _pdf(tmp_path / "b" / "same.pdf", b"2")]
    monkeypatch.setattr(
        convert_mod, "convert_single",
        lambda pdf_path, settings, *, output_dir, output_stem, **kw: output_dir / f"{output_stem}.md",
    )
    pairs = convert_mod.convert_batch(pdfs, Settings(), output_dir=tmp_path / "out")
    assert [out.name for _, out in pairs] == ["same.md", "same (b).md"]
