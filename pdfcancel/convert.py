"""Core PDF → markdown/plaintext conversion."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console

from pdfcancel.config import Settings

console = Console()

MANIFEST_NAME = ".pdfcancel_manifest.json"


def file_hash(path: Path) -> str:
    """Compute SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(output_dir: Path) -> dict:
    """Load the processing manifest from output_dir, or return empty dict."""
    manifest_path = output_dir / MANIFEST_NAME
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    return {}


def save_manifest(output_dir: Path, manifest: dict) -> None:
    """Persist the processing manifest."""
    manifest_path = output_dir / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2))


def strip_markdown(text: str) -> str:
    """Convert markdown to plaintext by stripping syntax."""
    # Remove images
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)
    # Remove links, keep text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Remove heading markers
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Remove bold/italic
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}([^_]+)_{1,3}", r"\1", text)
    # Remove inline code backticks
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # Remove horizontal rules
    text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def gather_pdfs(path: Path) -> list[Path]:
    """Resolve a path to a list of PDF files (single file or recursive dir)."""
    if path.is_file():
        if path.suffix.lower() != ".pdf":
            raise SystemExit(f"Error: {path} is not a PDF file.")
        return [path]
    if path.is_dir():
        pdfs = sorted(path.rglob("*.pdf"))
        if not pdfs:
            raise SystemExit(f"Error: No PDF files found in {path}")
        return pdfs
    raise SystemExit(f"Error: {path} does not exist.")


def convert_single(
    pdf_path: Path,
    settings: Settings,
    *,
    output_dir: Path,
    plaintext: bool = False,
    extract_images: bool = False,
    embed_images: bool = False,
    full: bool = False,
    force: bool = False,
    no_clean: bool = False,
    produce_chunks: bool = False,
    chunk_size: int = 1024,
    chunker_type: str = "recursive",
) -> Path:
    """Convert a single PDF to markdown (or plaintext).

    Returns the path to the output file.
    """
    from pdfcancel.clean import clean_markdown
    from pdfcancel.images import process_images

    settings.require_api_key()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check manifest for skip-processing
    manifest = load_manifest(output_dir)
    pdf_hash = file_hash(pdf_path)
    stem = pdf_path.stem

    if not force and stem in manifest and manifest[stem].get("hash") == pdf_hash:
        out_ext = ".txt" if plaintext else ".md"
        existing = output_dir / f"{stem}{out_ext}"
        if existing.exists():
            console.print(f"  [dim]Skipping {pdf_path.name} (unchanged)[/dim]")
            return existing

    console.print(f"  [bold]Cancelling[/bold] {pdf_path.name} ...")

    if full:
        # Use mistralai SDK directly to get both text and image base64 data
        from pdfcancel.images import ocr_with_images
        markdown_content, raw_images = ocr_with_images(pdf_path, settings)
    else:
        # Standard path: Chonkie MistralOCR (no image data)
        from chonkie import MistralOCR
        ocr = MistralOCR(model=settings.ocr_model, api_key=settings.mistral_api_key)
        doc = ocr.process(str(pdf_path))
        markdown_content = doc.content
        raw_images = {}

    # Handle images (extract to disk / embed as base64)
    if extract_images or embed_images:
        from pdfcancel.images import process_images_from_raw
        markdown_content = process_images_from_raw(
            markdown_content,
            images=raw_images,
            stem=stem,
            output_dir=output_dir,
            extract=extract_images,
            embed=embed_images,
        )

    # Post-OCR cleanup: strip page artifacts, rejoin broken sentences
    if not no_clean:
        markdown_content = clean_markdown(markdown_content)

    # --full: multimodal image descriptions
    if full and raw_images:
        from pdfcancel.multimodal import describe_images, inject_descriptions

        console.print(f"  [bold]Describing[/bold] {len(raw_images)} image(s) ...")

        # Load cached descriptions from manifest
        cached = manifest.get(stem, {}).get("image_descriptions", {})

        descriptions = describe_images(
            raw_images,
            settings,
            cached_descriptions=cached,
        )
        markdown_content = inject_descriptions(markdown_content, descriptions)

        # Build cache for manifest: {content_hash: description}
        import hashlib as _hl
        desc_cache = {}
        for img_id, img_bytes in raw_images.items():
            h = _hl.sha256(img_bytes).hexdigest()[:16]
            if img_id in descriptions:
                desc_cache[h] = descriptions[img_id]
    else:
        desc_cache = {}

    # --chunks: produce JSONL chunked output
    if produce_chunks:
        from pdfcancel.chunks import chunk_markdown, write_chunks_jsonl

        console.print(f"  [bold]Chunking[/bold] ({chunker_type}, size={chunk_size}) ...")
        chunks = chunk_markdown(
            markdown_content,
            source_file=pdf_path.name,
            chunk_size=chunk_size,
            chunker_type=chunker_type,
        )
        chunks_path = output_dir / f"{stem}_chunks.jsonl"
        write_chunks_jsonl(chunks, chunks_path)
        console.print(f"    {len(chunks)} chunks → {chunks_path.name}")

    # Write output
    if plaintext:
        out_path = output_dir / f"{stem}.txt"
        out_path.write_text(strip_markdown(markdown_content))
    else:
        out_path = output_dir / f"{stem}.md"
        out_path.write_text(markdown_content)

    # Update manifest
    manifest_entry = {
        "hash": pdf_hash,
        "source": str(pdf_path),
        "output": str(out_path),
        "converted_at": datetime.now(timezone.utc).isoformat(),
    }
    if desc_cache:
        manifest_entry["image_descriptions"] = desc_cache
    manifest[stem] = manifest_entry
    save_manifest(output_dir, manifest)

    return out_path


def convert_batch(
    pdf_paths: list[Path],
    settings: Settings,
    *,
    output_dir: Path,
    plaintext: bool = False,
    extract_images: bool = False,
    embed_images: bool = False,
    full: bool = False,
    force: bool = False,
    no_clean: bool = False,
    produce_chunks: bool = False,
    chunk_size: int = 1024,
    chunker_type: str = "recursive",
    verbose: bool = False,
) -> list[Path]:
    """Convert multiple PDFs, with progress display."""
    results = []
    total = len(pdf_paths)
    failed = 0

    for idx, pdf_path in enumerate(pdf_paths, 1):
        try:
            console.print(
                f"  [dim]({idx}/{total})[/dim] ", end="",
            )
            out = convert_single(
                pdf_path,
                settings,
                output_dir=output_dir,
                plaintext=plaintext,
                extract_images=extract_images,
                embed_images=embed_images,
                full=full,
                force=force,
                no_clean=no_clean,
                produce_chunks=produce_chunks,
                chunk_size=chunk_size,
                chunker_type=chunker_type,
            )
            results.append(out)
            if verbose:
                console.print(f"  [green]✓[/green] {pdf_path.name} → {out.name}")
        except Exception as e:
            failed += 1
            console.print(f"  [red]✗[/red] {pdf_path.name}: {e}")

    if failed:
        console.print(f"  [yellow]{failed} file(s) failed.[/yellow]")

    return results
