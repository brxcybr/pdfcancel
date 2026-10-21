"""Post-OCR output validation for pdfcancel.

Catches silently-bad OCR responses (empty page lists, near-empty text,
page-count mismatches against the source PDF) before output is written.
"""

from __future__ import annotations

from pathlib import Path

# Average characters per page below which OCR output is considered broken.
MIN_AVG_CHARS_PER_PAGE = 20


class OcrValidationError(Exception):
    """Raised when OCR output fails validation; output must not be written."""


def pdf_page_count(pdf_path: Path) -> int | None:
    """Return the page count of the source PDF via pypdf, or None if unreadable."""
    try:
        from pypdf import PdfReader
    except ImportError:
        return None
    try:
        return len(PdfReader(str(pdf_path)).pages)
    except Exception:
        return None


def validate_ocr_result(
    pdf_path: Path,
    markdown_content: str,
    *,
    ocr_page_count: int | None = None,
) -> list[str]:
    """Validate OCR output for a single PDF.

    Args:
        pdf_path: The source PDF (used for page-count comparison via pypdf).
        markdown_content: The joined OCR markdown.
        ocr_page_count: Number of pages in the OCR response, when known.

    Returns:
        A list of non-fatal warning strings (possibly empty).

    Raises:
        OcrValidationError: When the output is unusable (zero pages or
            too little extracted text). Callers must not write output.
    """
    warnings: list[str] = []

    if ocr_page_count is not None and ocr_page_count == 0:
        raise OcrValidationError(
            f"OCR returned 0 pages for {pdf_path.name}."
        )

    expected_pages = pdf_page_count(pdf_path)

    text_len = len(markdown_content.strip())
    pages_for_threshold = ocr_page_count or expected_pages or 1
    min_len = MIN_AVG_CHARS_PER_PAGE * pages_for_threshold
    if text_len < min_len:
        raise OcrValidationError(
            f"OCR extracted only {text_len} characters for {pdf_path.name} "
            f"({pages_for_threshold} page(s); expected at least {min_len}). "
            "The response looks empty or truncated."
        )

    if (
        ocr_page_count is not None
        and expected_pages is not None
        and ocr_page_count != expected_pages
    ):
        warnings.append(
            f"OCR returned {ocr_page_count} page(s) but {pdf_path.name} "
            f"has {expected_pages} page(s)."
        )

    return warnings
