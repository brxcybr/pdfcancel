"""Image extraction and embedding for pdfcancel output."""

from __future__ import annotations

import base64
import re
from pathlib import Path
from typing import Any


def process_images(
    markdown_content: str,
    *,
    doc: Any,
    stem: str,
    output_dir: Path,
    extract: bool = False,
    embed: bool = False,
) -> str:
    """Process image references in OCR markdown output.

    Args:
        markdown_content: The raw markdown from OCR.
        doc: The MarkdownDocument from Chonkie's MistralOCR.
        stem: Base filename (without extension) for naming.
        output_dir: Where to write image files.
        extract: If True, save images to disk and rewrite refs to relative paths.
        embed: If True, embed images as base64 data URIs inline.

    Returns:
        Updated markdown content with image references handled.
    """
    if not extract and not embed:
        # Leave image references as-is from OCR output
        return markdown_content

    # Collect image data from the OCR document metadata
    # Chonkie's MistralOCR stores image info in the document metadata
    images = _collect_images(doc)

    if not images:
        return markdown_content

    if extract:
        return _extract_images(markdown_content, images, stem, output_dir)
    if embed:
        return _embed_images(markdown_content, images)

    return markdown_content


def _collect_images(doc: Any) -> dict[str, bytes]:
    """Extract image ID → raw bytes mapping from the OCR document.

    Chonkie's MistralOCR wraps the Mistral OCR response. We look for
    base64-encoded image data in the metadata.
    """
    images: dict[str, bytes] = {}

    metadata = getattr(doc, "metadata", {}) or {}

    # The Mistral OCR response includes pages with image data
    # when include_image_base64=True. Chonkie may store this in metadata
    # or we can access it from the raw response if available.
    raw_response = metadata.get("raw_response") or metadata.get("response")
    if raw_response and hasattr(raw_response, "pages"):
        for page in raw_response.pages:
            for img in getattr(page, "images", []):
                img_id = getattr(img, "id", None)
                img_b64 = getattr(img, "image_base64", None)
                if img_id and img_b64:
                    # Strip data URI prefix if present
                    if "," in img_b64:
                        img_b64 = img_b64.split(",", 1)[1]
                    try:
                        images[img_id] = base64.b64decode(img_b64)
                    except Exception:
                        pass

    return images


def _extract_images(
    markdown_content: str,
    images: dict[str, bytes],
    stem: str,
    output_dir: Path,
) -> str:
    """Save images to disk and rewrite markdown references."""
    if not images:
        return markdown_content

    img_dir = output_dir / f"{stem}_images"
    img_dir.mkdir(parents=True, exist_ok=True)

    for img_id, img_bytes in images.items():
        # Determine extension from the image ID or default to png
        ext = Path(img_id).suffix or ".png"
        img_filename = f"{Path(img_id).stem}{ext}"
        img_path = img_dir / img_filename
        img_path.write_bytes(img_bytes)

        # Rewrite references in markdown: ![...](img_id) → ![...](relative_path)
        relative = f"{stem}_images/{img_filename}"
        markdown_content = markdown_content.replace(f"]({img_id})", f"]({relative})")

    return markdown_content


def _embed_images(
    markdown_content: str,
    images: dict[str, bytes],
) -> str:
    """Replace image references with base64 data URIs."""
    for img_id, img_bytes in images.items():
        ext = Path(img_id).suffix.lstrip(".") or "png"
        mime = f"image/{ext}" if ext != "jpg" else "image/jpeg"
        b64 = base64.b64encode(img_bytes).decode("ascii")
        data_uri = f"data:{mime};base64,{b64}"

        # Replace the image reference
        markdown_content = markdown_content.replace(f"]({img_id})", f"]({data_uri})")

    return markdown_content
