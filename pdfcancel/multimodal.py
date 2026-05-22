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

# Prompt for the vision model — tuned for academic/technical documents.
DESCRIBE_PROMPT = (
    "You are analyzing an image extracted from an academic or technical document. "
    "Describe what this image shows in 2-4 sentences. Focus on:\n"
    "- What type of figure it is (chart, diagram, table, photograph, etc.)\n"
    "- The key data, relationships, or concepts it conveys\n"
    "- Any specific values, labels, or trends visible\n"
    "Be factual and concise. Do not speculate beyond what is visible."
)


def describe_images(
    images: dict[str, bytes],
    settings: Settings,
    *,
    cached_descriptions: dict[str, str] | None = None,
) -> dict[str, str]:
    """Send each image to a vision model and return descriptions.

    Args:
        images: Mapping of image ID → raw bytes.
        settings: Settings with API key and multimodal_model.
        cached_descriptions: Previously cached {content_hash: description} to skip.

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

        console.print(f"    [dim]Describing image {idx}/{total}: {img_id}[/dim]")

        try:
            response = client.chat.complete(
                model=settings.multimodal_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": DESCRIBE_PROMPT},
                            {"type": "image_url", "image_url": {"url": data_uri}},
                        ],
                    }
                ],
                max_tokens=300,
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
        # Handle both raw OCR refs (img-0.jpeg) and rewritten paths (stem_images/img-0.jpeg)
        img_stem = Path(img_id).stem
        img_ext = Path(img_id).suffix

        # Pattern: ![anything](anything/img_stem.ext) possibly followed by newlines
        pattern = re.compile(
            r"(!\[[^\]]*\]\([^)]*"
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
