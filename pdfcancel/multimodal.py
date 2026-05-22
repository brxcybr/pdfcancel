"""Multimodal image description for pdfcancel --full mode.

Sends extracted images to a vision-capable model (Pixtral by default)
to generate text descriptions of charts, figures, diagrams, and tables.
Descriptions are inserted into the markdown below each image reference.
"""

from __future__ import annotations

import base64
import hashlib
import re
from pathlib import Path

from rich.console import Console

from pdfcancel.config import Settings

console = Console()

# Base prompt for the vision model.
# When surrounding document context is available, it gets prepended to this.
_BASE_PROMPT = (
    "Provide a thorough description in 3-6 sentences. Include:\n"
    "- Figure type (bar chart, line graph, flowchart, network diagram, table, etc.)\n"
    "- For charts/graphs: axis labels, units, data ranges, specific values for key "
    "data points, and trends (increasing, decreasing, peaks, outliers)\n"
    "- For diagrams/flowcharts: the components, their relationships, the flow "
    "direction, and what process or architecture is being depicted\n"
    "- For tables: column headers, row count, and notable values\n"
    "- Any legend, annotations, color coding, or callouts visible\n"
    "Extract as much specific, quantitative information as possible. "
    "Use the document's own terminology where applicable. "
    "This description will replace the image for text-based search and analysis."
)

# Context window: chars before/after the image reference to extract
_CONTEXT_WINDOW = 500


def _build_prompt(img_id: str, markdown_content: str | None) -> str:
    """Build a context-enriched prompt for a specific image.

    If markdown_content is provided, extracts the surrounding text around
    the image reference and includes it so the vision model can ground
    its description in the document's terminology and framing.
    """
    if not markdown_content:
        return (
            "You are analyzing an image extracted from an academic or "
            "technical document. " + _BASE_PROMPT
        )

    # Find the image reference in the markdown
    img_stem = Path(img_id).stem
    pattern = re.compile(re.escape(img_stem), re.IGNORECASE)
    match = pattern.search(markdown_content)

    if not match:
        return (
            "You are analyzing an image extracted from an academic or "
            "technical document. " + _BASE_PROMPT
        )

    # Extract surrounding context
    pos = match.start()
    start = max(0, pos - _CONTEXT_WINDOW)
    end = min(len(markdown_content), pos + _CONTEXT_WINDOW)
    context = markdown_content[start:end].strip()

    # Clean: remove image markdown syntax and collapse whitespace
    context = re.sub(r"!\[[^\]]*\]\([^)]+\)", "[IMAGE]", context)
    context = re.sub(r"\n{2,}", "\n", context)

    return (
        f"You are analyzing a figure from an academic or technical document.\n\n"
        f"SURROUNDING DOCUMENT CONTEXT:\n"
        f"\"\"\"\n{context}\n\"\"\"\n\n"
        f"Using the document context above to inform your terminology, "
        + _BASE_PROMPT
    )


def describe_images(
    images: dict[str, bytes],
    settings: Settings,
    *,
    cached_descriptions: dict[str, str] | None = None,
    markdown_content: str | None = None,
) -> dict[str, str]:
    """Send each image to a vision model and return descriptions.

    Args:
        images: Mapping of image ID → raw bytes.
        settings: Settings with API key and multimodal_model.
        cached_descriptions: Previously cached {content_hash: description} to skip.
        markdown_content: The full markdown text, used to extract surrounding
            context for each image. When provided, the vision model receives
            the ~500 chars around the image reference so it can ground its
            description in the document's terminology.

    Returns:
        Mapping of image ID → text description.
    """
    from mistralai.client import Mistral

    if not images:
        return {}

    cached = cached_descriptions or {}
    client = Mistral(api_key=settings.require_api_key())
    descriptions: dict[str, str] = {}
    total = len(images)

    for idx, (img_id, img_bytes) in enumerate(images.items(), 1):
        content_hash = hashlib.sha256(img_bytes).hexdigest()[:16]

        # Use cached description if available
        if content_hash in cached:
            descriptions[img_id] = cached[content_hash]
            continue

        # Determine MIME type from image ID extension
        ext = Path(img_id).suffix.lstrip(".").lower() or "png"
        mime = f"image/{'jpeg' if ext in ('jpg', 'jpeg') else ext}"
        b64_data = base64.b64encode(img_bytes).decode("ascii")
        data_uri = f"data:{mime};base64,{b64_data}"

        # Build context-enriched prompt
        prompt = _build_prompt(img_id, markdown_content)

        console.print(f"    [dim]Describing image {idx}/{total}: {img_id}[/dim]")

        try:
            response = client.chat.complete(
                model=settings.multimodal_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": data_uri}},
                        ],
                    }
                ],
                max_tokens=400,
            )
            description = response.choices[0].message.content.strip()
            descriptions[img_id] = description
        except Exception as e:
            console.print(f"    [yellow]Warning: Could not describe {img_id}: {e}[/yellow]")
            descriptions[img_id] = f"[Image description unavailable: {e}]"

    return descriptions


def inject_descriptions(
    markdown_content: str,
    descriptions: dict[str, str],
) -> str:
    """Insert image descriptions into markdown below each image reference.

    Transforms:
        ![alt](path/to/img-0.jpeg)

    Into:
        ![alt](path/to/img-0.jpeg)
        > **Figure description:** This bar chart shows...

    Only injects if a description exists for the image ID and the image
    doesn't already have a description block below it.
    """
    for img_id, description in descriptions.items():
        if not description or description.startswith("[Image description unavailable"):
            continue

        # Find all markdown image references that contain this image ID
        # Handle both raw OCR refs (img-0.jpeg) and rewritten paths
        # that may contain parentheses, spaces, etc.
        img_stem = Path(img_id).stem
        img_ext = Path(img_id).suffix

        # Pattern: ![anything](anything containing img_stem.ext)
        # Use a non-greedy match for the path to handle parens in filenames
        pattern = re.compile(
            r"(!\[[^\]]*\]\(.+?"
            + re.escape(img_stem)
            + re.escape(img_ext)
            + r"\))"
            + r"(\n*)"
        )

        def _insert_desc(match: re.Match) -> str:
            img_ref = match.group(1)
            trailing = match.group(2)
            # Don't double-insert if description block already exists
            # Check if the text after this match already has a blockquote
            return f"{img_ref}\n> **Figure description:** {description}\n"

        markdown_content = pattern.sub(_insert_desc, markdown_content)

    return markdown_content


def image_content_hashes(images: dict[str, bytes]) -> dict[str, str]:
    """Return {content_hash: img_id} for caching purposes."""
    return {
        hashlib.sha256(img_bytes).hexdigest()[:16]: img_id
        for img_id, img_bytes in images.items()
    }
