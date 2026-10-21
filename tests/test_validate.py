"""Tests for OCR output validation (validate.py)."""

from __future__ import annotations

from pathlib import Path

import pytest

from pdfcancel import validate as validate_mod
from pdfcancel.validate import OcrValidationError, validate_ocr_result

PDF = Path("/fake/paper.pdf")


def test_zero_pages_raises(monkeypatch):
    monkeypatch.setattr(validate_mod, "pdf_page_count", lambda _p: 10)
    with pytest.raises(OcrValidationError):
        validate_ocr_result(PDF, "", ocr_page_count=0)


def test_too_little_text_raises(monkeypatch):
    monkeypatch.setattr(validate_mod, "pdf_page_count", lambda _p: None)
    # 5 pages * 20 chars/page = 100 char minimum
    with pytest.raises(OcrValidationError):
        validate_ocr_result(PDF, "short", ocr_page_count=5)


def test_empty_text_raises_even_without_page_count(monkeypatch):
    monkeypatch.setattr(validate_mod, "pdf_page_count", lambda _p: None)
    with pytest.raises(OcrValidationError):
        validate_ocr_result(PDF, "   \n  ", ocr_page_count=None)


def test_page_count_mismatch_warns(monkeypatch):
    monkeypatch.setattr(validate_mod, "pdf_page_count", lambda _p: 12)
    warnings = validate_ocr_result(PDF, "x" * 500, ocr_page_count=10)
    assert len(warnings) == 1
    assert "10 page(s)" in warnings[0]
    assert "12 page(s)" in warnings[0]


def test_valid_output_passes_cleanly(monkeypatch):
    monkeypatch.setattr(validate_mod, "pdf_page_count", lambda _p: 3)
    warnings = validate_ocr_result(PDF, "x" * 500, ocr_page_count=3)
    assert warnings == []


def test_unreadable_pdf_skips_page_comparison(monkeypatch):
    monkeypatch.setattr(validate_mod, "pdf_page_count", lambda _p: None)
    warnings = validate_ocr_result(PDF, "x" * 500, ocr_page_count=3)
    assert warnings == []


def test_pdf_page_count_handles_missing_file():
    assert validate_mod.pdf_page_count(Path("/nonexistent/file.pdf")) is None
