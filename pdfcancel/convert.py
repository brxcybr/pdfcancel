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


def gather_pdfs(path: Path, *, exclude: list[str] | None = None) -> list[Path]:
    """Resolve a path to a list of PDF files (single file or recursive dir).

    Args:
        path: PDF file or directory to scan.
        exclude: List of patterns — PDFs whose filename contains any
                 pattern (case-insensitive) are skipped.
    """
    if path.is_file():
        if path.suffix.lower() != ".pdf":
            raise SystemExit(f"Error: {path} is not a PDF file.")
        return [path]
    if path.is_dir():
        pdfs = sorted(path.rglob("*.pdf"))
        if exclude:
            before = len(pdfs)
            patterns = [p.lower() for p in exclude]
            pdfs = [
                pdf for pdf in pdfs
                if not any(pat in pdf.name.lower() for pat in patterns)
            ]
            skipped = before - len(pdfs)
            if skipped:
                console.print(f"  [dim]Excluded {skipped} PDF(s) matching {exclude}[/dim]")
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
    preserve_pages: bool = False,
) -> Path:
    """Convert a single PDF to markdown (or plaintext).

    Returns the path to the output file.

    Raises:
        OcrValidationError: When OCR output fails validation; no output
            file is written.
    """
    from pdfcancel.clean import clean_markdown
    from pdfcancel.validate import validate_ocr_result
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

    # The direct SDK path is required whenever we need per-page content
    # (--preserve-pages) or image base64 data (--full / --images /
    # --embed-images). The Chonkie path only returns joined text.
    use_sdk = full or preserve_pages or extract_images or embed_images
    ocr_page_count: int | None = None

    if use_sdk:
        # Use mistralai SDK directly to get text, page list, and image data
        from pdfcancel.images import ocr_with_images
        ocr_result = ocr_with_images(
            pdf_path, settings, preserve_pages=preserve_pages,
        )
        markdown_content = ocr_result.markdown
        raw_images = ocr_result.images
        ocr_page_count = ocr_result.page_count
    else:
        # Standard path: Chonkie MistralOCR (no image data, joined text only)
        from chonkie import MistralOCR

        from pdfcancel.retry import with_retry
        ocr = MistralOCR(model=settings.ocr_model, api_key=settings.mistral_api_key)
        doc = with_retry(
            lambda: ocr.process(str(pdf_path)),
            description=f"OCR of {pdf_path.name}",
        )
        markdown_content = doc.content
        raw_images = {}

    # Validate OCR output before writing anything
    for warning in validate_ocr_result(
        pdf_path, markdown_content, ocr_page_count=ocr_page_count,
    ):
        console.print(f"  [yellow]Warning: {warning}[/yellow]")

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
            markdown_content=markdown_content,
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
    preserve_pages: bool = False,
) -> list[tuple[Path, Path | None]]:
    """Convert multiple PDFs, with progress display.

    Returns a list of (pdf_path, output_path_or_None) pairs in input order;
    None marks a failed conversion so callers never misalign results.
    """
    results: list[tuple[Path, Path | None]] = []
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
                preserve_pages=preserve_pages,
            )
            results.append((pdf_path, out))
            if verbose:
                console.print(f"  [green]✓[/green] {pdf_path.name} → {out.name}")
        except Exception as e:
            failed += 1
            results.append((pdf_path, None))
            console.print(f"  [red]✗[/red] {pdf_path.name}: {e}")

    if failed:
        console.print(f"  [yellow]{failed} file(s) failed.[/yellow]")

    return results


def convert_batch_pipeline(
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
    preserve_pages: bool = False,
) -> list[tuple[Path, Path | None]]:
    """Convert multiple PDFs using the Mistral Batch API for 50% cost savings.

    Uploads all PDFs, submits batch OCR, optionally batch vision for --full,
    then performs local cleanup, description injection, chunking, and output.

    Returns a list of (pdf_path, output_path_or_None) pairs; None marks a
    PDF that failed (upload, OCR, validation, or post-processing).
    """
    from pdfcancel.batch import batch_ocr, batch_vision
    from pdfcancel.clean import clean_markdown
    from pdfcancel.images import process_images_from_raw
    from pdfcancel.multimodal import (
        build_batch_vision_requests,
        inject_descriptions,
    )
    from pdfcancel.validate import validate_ocr_result

    settings.require_api_key()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter already-processed PDFs unless --force
    manifest = load_manifest(output_dir)
    to_process = []
    skipped: list[tuple[Path, Path | None]] = []
    for pdf_path in pdf_paths:
        pdf_hash = file_hash(pdf_path)
        stem = pdf_path.stem
        out_ext = ".txt" if plaintext else ".md"
        existing = output_dir / f"{stem}{out_ext}"
        if (
            not force
            and stem in manifest
            and manifest[stem].get("hash") == pdf_hash
            and existing.exists()
        ):
            skipped.append((pdf_path, existing))
            if verbose:
                console.print(f"  [dim]Skipping {pdf_path.name} (unchanged)[/dim]")
        else:
            to_process.append(pdf_path)

    if not to_process:
        console.print("[dim]All files up to date, nothing to process.[/dim]")
        return skipped

    total = len(to_process)
    console.print(
        f"\n[bold]Batch processing {total} PDF(s)[/bold]"
        f" ({len(skipped)} skipped)"
    )

    # Step 1: Batch OCR
    ocr_results = batch_ocr(
        to_process, settings, include_images=full, preserve_pages=preserve_pages,
    )

    if not ocr_results:
        console.print("[red]Batch OCR returned no results.[/red]")
        return skipped + [(p, None) for p in to_process]

    # Step 2: For --full, collect all vision requests across all PDFs
    all_vision_requests: list[dict] = []
    per_pdf_cached: dict[str, dict[str, str]] = {}  # stem → {img_id: desc}

    if full:
        for result in ocr_results:
            stem = result.pdf_path.stem
            cached = manifest.get(stem, {}).get("image_descriptions", {})
            requests, cached_hits = build_batch_vision_requests(
                result.images,
                settings,
                pdf_stem=stem,
                cached_descriptions=cached,
                markdown_content=result.markdown,
            )
            all_vision_requests.extend(requests)
            per_pdf_cached[stem] = cached_hits

    # Submit batch vision if we have requests
    vision_results: dict[str, str] = {}
    if all_vision_requests:
        vision_results = batch_vision(all_vision_requests, settings)

    # Step 3: Local post-processing per PDF
    results: list[tuple[Path, Path | None]] = list(skipped)
    failed = 0

    # PDFs that never came back from the batch job count as failures
    returned = {r.pdf_path for r in ocr_results}
    for pdf_path in to_process:
        if pdf_path not in returned:
            failed += 1
            results.append((pdf_path, None))

    for idx, result in enumerate(ocr_results, 1):
        pdf_path = result.pdf_path
        stem = pdf_path.stem
        markdown_content = result.markdown
        raw_images = result.images

        try:
            console.print(
                f"  [dim]({idx}/{len(ocr_results)})[/dim] "
                f"[bold]Post-processing[/bold] {pdf_path.name}"
            )

            # Validate OCR output before writing anything
            for warning in validate_ocr_result(
                pdf_path, markdown_content, ocr_page_count=result.page_count,
            ):
                console.print(f"  [yellow]Warning: {warning}[/yellow]")

            # Handle images (extract to disk / embed as base64)
            if extract_images or embed_images:
                markdown_content = process_images_from_raw(
                    markdown_content,
                    images=raw_images,
                    stem=stem,
                    output_dir=output_dir,
                    extract=extract_images,
                    embed=embed_images,
                )

            # Post-OCR cleanup
            if not no_clean:
                markdown_content = clean_markdown(markdown_content)

            # --full: inject image descriptions
            if full and raw_images:
                # Merge cached + batch results for this PDF
                descriptions: dict[str, str] = dict(per_pdf_cached.get(stem, {}))
                prefix = f"{stem}::"
                for cid, desc in vision_results.items():
                    if cid.startswith(prefix):
                        img_id = cid[len(prefix):]
                        descriptions[img_id] = desc

                if descriptions:
                    markdown_content = inject_descriptions(
                        markdown_content, descriptions,
                    )

                # Build desc_cache for manifest
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
                console.print(
                    f"    [bold]Chunking[/bold] ({chunker_type}, size={chunk_size})"
                )
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
            pdf_hash = file_hash(pdf_path)
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

            results.append((pdf_path, out_path))
            if verbose:
                console.print(
                    f"  [green]✓[/green] {pdf_path.name} → {out_path.name}"
                )

        except Exception as e:
            failed += 1
            results.append((pdf_path, None))
            console.print(f"  [red]✗[/red] {pdf_path.name}: {e}")

    if failed:
        console.print(f"  [yellow]{failed} file(s) failed.[/yellow]")

    return results
