"""Enhance mode for pdfcancel --enhance.

Takes an existing markdown file and its source PDF, then:
1. Runs OCR on the PDF to extract images (if not already extracted)
2. Adds multimodal AI descriptions to images lacking them
3. Optionally applies post-OCR cleanup to fix remaining artifacts
4. Preserves all existing user content (annotations, edits, notes)

Usage:
    pdfcancel paper.pdf --enhance paper.md
    pdfcancel paper.pdf --enhance paper.md --full  # add image descriptions
"""

from __future__ import annotations

import re
from pathlib import Path

from rich.console import Console

from pdfcancel.config import Settings

console = Console()


def enhance_markdown(
    pdf_path: Path,
    md_path: Path,
    settings: Settings,
    *,
    output_dir: Path,
    full: bool = False,
    no_clean: bool = False,
) -> Path:
    """Enhance an existing markdown file using its source PDF.

    Args:
        pdf_path: Path to the source PDF.
        md_path: Path to the existing markdown file.
        settings: Runtime settings.
        output_dir: Where to write enhanced output.
        full: If True, add multimodal image descriptions.
        no_clean: If True, skip post-OCR cleanup pass.

    Returns:
        Path to the enhanced markdown file.
    """
    settings.require_api_key()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read the existing markdown
    markdown_content = md_path.read_text()
    stem = md_path.stem

    console.print(f"  [bold]Enhancing[/bold] {md_path.name} ...")

    # Count existing image references and descriptions
    img_refs = re.findall(r"!\[[^\]]*\]\([^)]+\)", markdown_content)
    existing_descs = len(re.findall(r"> \*\*Figure description:\*\*", markdown_content))
    console.print(f"    Found {len(img_refs)} image ref(s), {existing_descs} existing description(s)")

    # Apply cleanup to the existing markdown if requested
    if not no_clean:
        from pdfcancel.clean import clean_markdown
        original_lines = len(markdown_content.splitlines())
        markdown_content = clean_markdown(markdown_content)
        cleaned_lines = len(markdown_content.splitlines())
        removed = original_lines - cleaned_lines
        if removed > 0:
            console.print(f"    Cleaned {removed} artifact lines")

    # If --full, extract images from PDF and describe any that lack descriptions
    if full:
        from pdfcancel.images import ocr_with_images
        from pdfcancel.multimodal import describe_images, inject_descriptions
        from pdfcancel.convert import load_manifest, save_manifest

        # Extract images from the source PDF
        console.print(f"    Extracting images from {pdf_path.name} ...")
        _, raw_images = ocr_with_images(pdf_path, settings)

        if raw_images:
            # Filter to images that don't already have descriptions
            undescribed = _find_undescribed_images(markdown_content, raw_images)

            if undescribed:
                console.print(f"  [bold]Describing[/bold] {len(undescribed)} new image(s) ...")

                # Load cached descriptions from manifest
                manifest = load_manifest(output_dir)
                cached = manifest.get(stem, {}).get("image_descriptions", {})

                descriptions = describe_images(
                    undescribed,
                    settings,
                    cached_descriptions=cached,
                )
                markdown_content = inject_descriptions(markdown_content, descriptions)

                # Update manifest cache
                import hashlib
                desc_cache = cached.copy()
                for img_id, img_bytes in undescribed.items():
                    h = hashlib.sha256(img_bytes).hexdigest()[:16]
                    if img_id in descriptions:
                        desc_cache[h] = descriptions[img_id]

                manifest.setdefault(stem, {})["image_descriptions"] = desc_cache
                save_manifest(output_dir, manifest)
            else:
                console.print("    All images already have descriptions")
        else:
            console.print("    No images found in PDF")

    # Write enhanced output
    out_path = output_dir / f"{stem}.md"
    out_path.write_text(markdown_content)

    return out_path


def _find_undescribed_images(
    markdown_content: str,
    raw_images: dict[str, bytes],
) -> dict[str, bytes]:
    """Return only images that don't already have a description block.

    An image is considered "described" if the markdown contains:
        ![...](anything/img_stem.ext)
        > **Figure description:** ...
    """
    undescribed: dict[str, bytes] = {}

    for img_id, img_bytes in raw_images.items():
        img_stem = Path(img_id).stem
        # Check if there's already a description block after this image ref
        pattern = re.compile(
            re.escape(img_stem) + r"[^)]*\)"
            + r"\s*\n\s*>\s*\*\*Figure description:\*\*"
        )
        if not pattern.search(markdown_content):
            undescribed[img_id] = img_bytes

    return undescribed
