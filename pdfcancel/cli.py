"""CLI entry point for pdfcancel.

Usage:
    pdfcancel paper.pdf                  # cancel a PDF → paper.md
    pdfcancel ./papers/                  # cancel a whole directory
    pdfcancel paper.pdf --images         # extract images alongside
    pdfcancel paper.pdf --full           # multimodal: describe charts/figures
    pdfcancel paper.pdf --enhance paper.md  # enhance existing markdown
"""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console

from pdfcancel import __version__
from pdfcancel.config import Settings
from pdfcancel.convert import convert_batch, convert_single, gather_pdfs

console = Console()


@click.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("-o", "--output", "output_dir", type=click.Path(), default=None,
              help="Output directory (default: same as input).")
@click.option("--images", is_flag=True, help="Extract images to companion directory.")
@click.option("--embed-images", is_flag=True, help="Embed images as base64 in markdown.")
@click.option("--plaintext", is_flag=True, help="Output plaintext instead of markdown.")
@click.option("--full", is_flag=True,
              help="Full multimodal extraction: AI-describe charts, figures, and diagrams.")
@click.option("--full-model", default=None,
              help="Vision model for --full (default: pixtral-large-latest).")
@click.option("--enhance", "enhance_md", type=click.Path(exists=True), default=None,
              help="Enhance an existing markdown file instead of generating from scratch.")
@click.option("--model", default=None, help="OCR model override (default: mistral-ocr-latest).")
@click.option("--chunks", is_flag=True, help="Also produce chunked JSONL output for RAG/LLM.")
@click.option("--chunk-size", type=int, default=1024,
              help="Max tokens per chunk (default: 1024). Used with --chunks.")
@click.option("--chunker", type=click.Choice(["recursive", "semantic", "sentence"]),
              default="recursive", help="Chunking strategy (default: recursive).")
@click.option("--no-clean", is_flag=True, help="Skip post-OCR cleanup (keep raw OCR output).")
@click.option("--force", is_flag=True, help="Re-process even if already converted.")
@click.option("--verbose", is_flag=True, help="Show detailed progress.")
@click.option("--version", is_flag=True, is_eager=True, expose_value=False,
              callback=lambda ctx, param, val: (
                  click.echo(f"pdfcancel {__version__}") or ctx.exit()
              ) if val else None,
              help="Show version and exit.")
def main(
    path,
    output_dir,
    images,
    embed_images,
    plaintext,
    full,
    full_model,
    enhance_md,
    model,
    chunks,
    chunk_size,
    chunker,
    no_clean,
    force,
    verbose,
):
    """Cancel your PDFs — convert them to clean markdown.

    \b
    Examples:
      pdfcancel paper.pdf                  # → paper.md
      pdfcancel ./papers/                  # cancel all PDFs in dir
      pdfcancel paper.pdf --images         # extract images too
      pdfcancel paper.pdf --full           # AI-describe all figures
      pdfcancel paper.pdf --enhance old.md # upgrade existing markdown
    """
    input_path = Path(path)
    settings = Settings()
    if model:
        settings.ocr_model = model
    if full_model:
        settings.multimodal_model = full_model

    # --enhance mode: enhance an existing markdown file
    if enhance_md:
        console.print("[yellow]Enhance mode is not yet implemented.[/yellow]")
        console.print(f"  PDF: {path}")
        console.print(f"  Markdown: {enhance_md}")
        console.print(f"  Full: {full}")
        return

    # --full implies --images (need the images to describe them)
    if full:
        images = True

    # Resolve output directory
    if output_dir:
        out = Path(output_dir)
    elif input_path.is_dir():
        out = input_path
    else:
        out = input_path.parent

    pdf_paths = gather_pdfs(input_path)

    if len(pdf_paths) == 1:
        result = convert_single(
            pdf_paths[0],
            settings,
            output_dir=out,
            plaintext=plaintext,
            extract_images=images,
            embed_images=embed_images,
            full=full,
            force=force,
            no_clean=no_clean,
            produce_chunks=chunks,
            chunk_size=chunk_size,
            chunker_type=chunker,
        )
        console.print(f"\n[green]Cancelled.[/green] → {result}")
    else:
        results = convert_batch(
            pdf_paths,
            settings,
            output_dir=out,
            plaintext=plaintext,
            extract_images=images,
            embed_images=embed_images,
            full=full,
            force=force,
            no_clean=no_clean,
            produce_chunks=chunks,
            chunk_size=chunk_size,
            chunker_type=chunker,
            verbose=verbose,
        )
        console.print(f"\n[green]Cancelled {len(results)} PDF(s).[/green]")
