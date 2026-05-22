"""CLI entry point for pdfcancel.

Usage:
    pdfcancel paper.pdf                       # cancel a PDF → paper.md
    pdfcancel paper.pdf --index cti           # cancel + index for search
    pdfcancel search cti "threat intelligence" # search indexed docs
    pdfcancel indexes                          # list all indexes
"""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from pdfcancel import __version__
from pdfcancel.config import Settings
from pdfcancel.convert import convert_batch, convert_single, gather_pdfs

console = Console()


class DefaultGroup(click.Group):
    """Click group that routes unknown first args to the default 'cancel' command."""

    def parse_args(self, ctx, args):
        if args and args[0] not in self.commands and not args[0].startswith("-"):
            args = ["cancel"] + args
        return super().parse_args(ctx, args)


@click.group(cls=DefaultGroup, invoke_without_command=True)
@click.option("--version", is_flag=True, is_eager=True, expose_value=False,
              callback=lambda ctx, param, val: (
                  click.echo(f"pdfcancel {__version__}") or ctx.exit()
              ) if val else None,
              help="Show version and exit.")
@click.pass_context
def main(ctx):
    """Cancel your PDFs — convert them to clean markdown."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


# ── Default command: cancel ───────────────────────────────────────────────────

@main.command(hidden=True)
@click.argument("path", type=click.Path(exists=True))
@click.option("-o", "--output", "output_dir", type=click.Path(), default=None,
              help="Output directory (default: same as input).")
@click.option("--images", is_flag=True, help="Extract images to companion directory.")
@click.option("--embed-images", is_flag=True, help="Embed images as base64 in markdown.")
@click.option("--plaintext", is_flag=True, help="Output plaintext instead of markdown.")
@click.option("--full", is_flag=True,
              help="Full multimodal: AI-describe charts, figures, and diagrams.")
@click.option("--full-model", default=None,
              help="Vision model for --full (default: pixtral-large-latest).")
@click.option("--enhance", "enhance_md", type=click.Path(exists=True), default=None,
              help="Enhance an existing markdown file.")
@click.option("--model", default=None, help="OCR model override (default: mistral-ocr-latest).")
@click.option("--chunks", is_flag=True, help="Also produce chunked JSONL output.")
@click.option("--chunk-size", type=int, default=1024,
              help="Max tokens per chunk (default: 1024).")
@click.option("--chunker", type=click.Choice(["recursive", "semantic", "sentence"]),
              default="recursive", help="Chunking strategy (default: recursive).")
@click.option("--index", "index_name", default=None,
              help="Add to a named search index (implies --chunks).")
@click.option("--no-clean", is_flag=True, help="Skip post-OCR cleanup.")
@click.option("--force", is_flag=True, help="Re-process even if already converted.")
@click.option("--verbose", is_flag=True, help="Show detailed progress.")
def cancel(
    path, output_dir, images, embed_images, plaintext, full, full_model,
    enhance_md, model, chunks, chunk_size, chunker, index_name,
    no_clean, force, verbose,
):
    """Cancel PDFs — convert to clean markdown.

    \b
    Examples:
      pdfcancel paper.pdf                       # → paper.md
      pdfcancel ./papers/                       # cancel all PDFs in dir
      pdfcancel paper.pdf --full                # AI-describe all figures
      pdfcancel paper.pdf --index cti           # cancel + index for search
      pdfcancel paper.pdf --enhance old.md      # upgrade existing markdown
    """
    input_path = Path(path)
    settings = Settings()
    if model:
        settings.ocr_model = model
    if full_model:
        settings.multimodal_model = full_model

    # --index implies --chunks
    if index_name:
        chunks = True

    # --enhance mode
    if enhance_md:
        from pdfcancel.enhance import enhance_markdown
        enh_out = Path(output_dir) if output_dir else Path(enhance_md).parent
        result = enhance_markdown(
            input_path, Path(enhance_md), settings,
            output_dir=enh_out, full=full, no_clean=no_clean,
        )
        console.print(f"\n[green]Enhanced.[/green] → {result}")
        return

    # --full implies --images
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
            pdf_paths[0], settings,
            output_dir=out, plaintext=plaintext, extract_images=images,
            embed_images=embed_images, full=full, force=force,
            no_clean=no_clean, produce_chunks=chunks,
            chunk_size=chunk_size, chunker_type=chunker,
        )
        if index_name:
            _index_file(result, pdf_paths[0].name, index_name, chunk_size, chunker)
        console.print(f"\n[green]Cancelled.[/green] → {result}")
    else:
        results = convert_batch(
            pdf_paths, settings,
            output_dir=out, plaintext=plaintext, extract_images=images,
            embed_images=embed_images, full=full, force=force,
            no_clean=no_clean, produce_chunks=chunks,
            chunk_size=chunk_size, chunker_type=chunker, verbose=verbose,
        )
        if index_name:
            for res_path, pdf_path in zip(results, pdf_paths):
                _index_file(res_path, pdf_path.name, index_name, chunk_size, chunker)
        console.print(f"\n[green]Cancelled {len(results)} PDF(s).[/green]")


def _index_file(
    md_path: Path, source_name: str, index_name: str,
    chunk_size: int, chunker_type: str,
) -> None:
    """Read a markdown file, chunk it, and ingest into the named index."""
    from pdfcancel.chunks import chunk_markdown
    from pdfcancel.index import ingest_chunks

    markdown_content = md_path.read_text()
    chunks_data = chunk_markdown(
        markdown_content, source_file=source_name,
        chunk_size=chunk_size, chunker_type=chunker_type,
    )
    console.print(f"  [bold]Indexing[/bold] {source_name} → '{index_name}' ...")
    count = ingest_chunks(chunks_data, index_name, source_name)
    console.print(f"    {count} chunks indexed")


# ── Search command ─────────────────────────────────────────────────────────

@main.command()
@click.argument("index_name")
@click.argument("query")
@click.option("-k", "--top-k", type=int, default=10, help="Number of results (default: 10).")
@click.option("--mode", type=click.Choice(["hybrid", "semantic", "text"]),
              default="hybrid", help="Search mode (default: hybrid).")
def search(index_name, query, top_k, mode):
    """Search across indexed documents.

    \b
    Examples:
      pdfcancel search cti "threat intelligence sharing"
      pdfcancel search cti "ATT&CK framework" --mode semantic
      pdfcancel search cti "risk management" -k 5
    """
    from pdfcancel.index import search as do_search

    results = do_search(query, index_name, top_k=top_k, mode=mode)

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    console.print(
        f"\n[bold]Results for[/bold] '{query}' "
        f"[dim](index: {index_name}, mode: {mode})[/dim]\n"
    )

    for i, r in enumerate(results, 1):
        score = f"{r['score']:.3f}"
        method = r.get("method", "")
        source = Path(r["source"]).stem
        section = r.get("section", "")

        console.print(f"[bold cyan]#{i}[/bold cyan] [green]{score}[/green] [{method}]  [dim]{source}[/dim]")
        if section:
            console.print(f"   [dim]§ {section}[/dim]")
        preview = r["text"][:200].replace("\n", " ").strip()
        if len(r["text"]) > 200:
            preview += "..."
        console.print(f"   {preview}\n")


# ── Indexes command ───────────────────────────────────────────────────────

@main.command()
def indexes():
    """List all search indexes and their contents."""
    from pdfcancel.index import list_indexes

    idx_list = list_indexes()
    if not idx_list:
        console.print("[yellow]No indexes found.[/yellow]")
        console.print("  Create one: pdfcancel paper.pdf --index myproject")
        return

    for idx in idx_list:
        table = Table(title=f"Index: {idx['name']}", show_lines=True)
        table.add_column("Source", style="cyan")
        table.add_column("Chunks", justify="right")
        for src in idx["sources"]:
            table.add_row(Path(src["source"]).stem, str(src["chunks"]))
        table.add_row("[bold]Total[/bold]", f"[bold]{idx['total_chunks']}[/bold]")
        console.print(table)
        console.print(f"  [dim]{idx['path']}[/dim]\n")
