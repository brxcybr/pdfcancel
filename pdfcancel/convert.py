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
    force: bool = False,
    no_clean: bool = False,
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

    # Run Mistral OCR via Chonkie
    from chonkie import MistralOCR

    ocr = MistralOCR(model=settings.ocr_model, api_key=settings.mistral_api_key)

    console.print(f"  [bold]Cancelling[/bold] {pdf_path.name} ...")
    doc = ocr.process(str(pdf_path))
    markdown_content = doc.content

    # Handle images
    markdown_content = process_images(
        markdown_content,
        doc=doc,
        stem=stem,
        output_dir=output_dir,
        extract=extract_images,
        embed=embed_images,
    )

    # Post-OCR cleanup: strip page artifacts, rejoin broken sentences
    if not no_clean:
        markdown_content = clean_markdown(markdown_content)

    # Write output
    if plaintext:
        out_path = output_dir / f"{stem}.txt"
        out_path.write_text(strip_markdown(markdown_content))
    else:
        out_path = output_dir / f"{stem}.md"
        out_path.write_text(markdown_content)

    # Update manifest
    manifest[stem] = {
        "hash": pdf_hash,
        "source": str(pdf_path),
        "output": str(out_path),
        "converted_at": datetime.now(timezone.utc).isoformat(),
    }
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
    force: bool = False,
    no_clean: bool = False,
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
                force=force,
                no_clean=no_clean,
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
