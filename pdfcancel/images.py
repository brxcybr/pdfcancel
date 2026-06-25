"""Image extraction and embedding for pdfcancel output."""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console

from pdfcancel.config import Settings
from pdfcancel.pages import join_pages
from pdfcancel.retry import with_retry

console = Console()


@dataclass
class OcrResult:
    """Result of a direct-SDK OCR call."""

    markdown: str
    images: dict[str, bytes] = field(default_factory=dict)
    page_markdowns: list[str] = field(default_factory=list)
    dropped_images: int = 0

    @property
    def page_count(self) -> int:
        return len(self.page_markdowns)


def decode_page_images(
    images: dict[str, bytes],
    image_items: list[tuple[str, str]],
) -> int:
    """Decode (image_id, base64) pairs into `images`, returning the drop count.

    Strips data-URI prefixes; entries that fail base64 decoding are counted
    rather than silently discarded.
    """
    dropped = 0
    for img_id, img_b64 in image_items:
        if not img_id or not img_b64:
            continue
        if "," in img_b64:
            img_b64 = img_b64.split(",", 1)[1]
        try:
            images[img_id] = base64.b64decode(img_b64)
        except Exception:
            dropped += 1
    return dropped


def ocr_with_images(
    pdf_path: Path,
    settings: Settings,
    *,
    preserve_pages: bool = False,
) -> OcrResult:
    """Run Mistral OCR directly via the SDK with include_image_base64=True.

    Returns an OcrResult with markdown, image bytes, per-page markdown,
    and the count of images dropped due to decode failures.
    When preserve_pages is True, page markers are injected at join time.
    """
    client = settings.build_client()

    # Upload PDF to Mistral for OCR
    with open(pdf_path, "rb") as f:
        content = f.read()
    uploaded = with_retry(
        lambda: client.files.upload(
            file={"file_name": pdf_path.name, "content": content},
            purpose="ocr",
        ),
        description=f"upload of {pdf_path.name}",
    )
    signed_url = with_retry(
        lambda: client.files.get_signed_url(file_id=uploaded.id, expiry=1),
        description="signed URL request",
    )

    # Run OCR with image extraction
    from mistralai.client.models import DocumentURLChunk
    response = with_retry(
        lambda: client.ocr.process(
            document=DocumentURLChunk(document_url=signed_url.url),
            model=settings.ocr_model,
            include_image_base64=True,
        ),
        description=f"OCR of {pdf_path.name}",
    )

    # Assemble markdown from all pages
    page_markdowns: list[str] = []
    images: dict[str, bytes] = {}
    dropped = 0

    for page in response.pages:
        page_markdowns.append(page.markdown)
        dropped += decode_page_images(
            images,
            [(img.id, img.image_base64) for img in page.images],
        )

    if dropped:
        console.print(
            f"  [yellow]Warning: {dropped} image(s) from {pdf_path.name} "
            "could not be decoded and were dropped.[/yellow]"
        )

    markdown_content = join_pages(page_markdowns, preserve_pages=preserve_pages)
    return OcrResult(
        markdown=markdown_content,
        images=images,
        page_markdowns=page_markdowns,
        dropped_images=dropped,
    )


def process_images_from_raw(
    markdown_content: str,
    *,
    images: dict[str, bytes],
    stem: str,
    output_dir: Path,
    extract: bool = False,
    embed: bool = False,
) -> str:
    """Process images from raw bytes (used when we have image data from direct SDK call)."""
    if not images or (not extract and not embed):
        return markdown_content
    if extract:
        return _extract_images(markdown_content, images, stem, output_dir)
    if embed:
        return _embed_images(markdown_content, images)
    return markdown_content


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
        dropped = 0
        for page in raw_response.pages:
            items = [
                (getattr(img, "id", ""), getattr(img, "image_base64", ""))
                for img in getattr(page, "images", [])
            ]
            dropped += decode_page_images(images, items)
        if dropped:
            console.print(
                f"  [yellow]Warning: {dropped} image(s) could not be decoded "
                "and were dropped.[/yellow]"
            )

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
