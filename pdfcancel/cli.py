"""CLI entry point for pdfcancel.

Usage:
    pdfcancel paper.pdf                       # cancel a PDF → paper.md
    pdfcancel paper.pdf --index cti           # cancel + index for search
    pdfcancel search cti "threat intelligence" # search indexed docs
    pdfcancel indexes                          # list all indexes
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from pdfcancel import __version__
from pdfcancel.config import Settings
from pdfcancel.convert import (
    convert_batch,
    convert_batch_pipeline,
    convert_single,
    gather_pdfs,
    output_stems,
)
from pdfcancel.errors import ConversionIncomplete
from pdfcancel.runstate import RUN_STATE_NAME, RunState, stop_on_signals

console = Console()


def _maybe_reexec_caffeinate(enabled: bool) -> None:
    """On macOS, re-exec the current command under ``caffeinate -i`` so idle
    sleep cannot interrupt a long run. Guarded by an env var so it happens at
    most once; a no-op off macOS or when caffeinate is unavailable.
    """
    if not enabled or os.environ.get("PDFCANCEL_CAFFEINATED") == "1":
        return
    if sys.platform != "darwin":
        console.print("[dim]--caffeinate is macOS-only; continuing without it.[/dim]")
        return
    import shutil

    caffeinate = shutil.which("caffeinate")
    if not caffeinate:
        console.print("[yellow]caffeinate not found; continuing without it.[/yellow]")
        return
    env = dict(os.environ, PDFCANCEL_CAFFEINATED="1")
    console.print("[dim]Keeping the system awake via caffeinate for this run.[/dim]")
    os.execve(caffeinate, [caffeinate, "-i", sys.executable, *sys.argv], env)


def _apply_vlm_overrides(
    settings: Settings,
    *,
    ocr_engine: str | None,
    api_base: str | None,
    api_key: str | None,
    ca_bundle: str | None,
    model: str | None,
    ocr_dpi: int | None,
    ocr_concurrency: int | None,
    ocr_retries: int | None,
    ocr_timeout: float | None,
) -> None:
    """Apply engine/provider CLI overrides onto ``settings`` (shared by backlog)."""
    from pdfcancel.config import _normalize_base_url

    if ocr_engine:
        settings.ocr_engine = ocr_engine
    if api_base:
        settings.api_base = _normalize_base_url(api_base)
        if not ocr_engine:
            settings.ocr_engine = "vlm"
    if api_key:
        settings.api_key = api_key
    if ca_bundle:
        settings.mistral_ca_bundle = ca_bundle
    if model:
        if settings.ocr_engine == "vlm":
            settings.vlm_model = model
        else:
            settings.ocr_model = model
    if ocr_dpi:
        settings.ocr_dpi = ocr_dpi
    if ocr_concurrency:
        settings.ocr_concurrency = ocr_concurrency
    if ocr_retries is not None:
        settings.ocr_retries = ocr_retries
    if ocr_timeout is not None:
        settings.ocr_timeout = ocr_timeout


class DefaultGroup(click.Group):
    """Click group that routes unknown first args to the default 'cancel' command."""

    def parse_args(self, ctx, args):
        if args and args[0] not in self.commands and not args[0].startswith("-"):
            args = ["cancel"] + args
        return super().parse_args(ctx, args)


@click.group(
    cls=DefaultGroup,
    invoke_without_command=True,
    help="Convert PDFs to clean markdown using AI-powered OCR. "
         "Supports Mistral API and OpenAI-compatible vision models."
)
@click.option("--version", is_flag=True, is_eager=True, expose_value=False,
              callback=lambda ctx, param, val: (
                  click.echo(f"pdfcancel {__version__}") or ctx.exit()
              ) if val else None,
              help="Show version and exit.")
@click.pass_context
def main(ctx):
    """Cancel your PDFs — convert them to clean markdown.
    
    PDFCANCEL is a powerful PDF to markdown converter that uses AI-powered OCR
    to extract text, tables, and figures from PDF documents, producing clean,
    readable markdown output.
    
    The tool supports two OCR engines:
    
    \b
    MISTRAL ENGINE (default):
      Uses Mistral AI's native OCR API (mistral-ocr-latest). Best for general
      purpose PDF conversion. Requires MISTRAL_API_KEY environment variable
      or --mistral-url for custom endpoints.
    
    \b
    VLM ENGINE:
      Uses any OpenAI-compatible vision model via /v1/chat/completions.
      Allows local/self-hosted models (e.g., Ollama, vLLM, LiteLLM).
      Activate with --ocr-engine vlm or by setting --api-base.
    
    \b
    Common Workflows:
      Single PDF:          pdfcancel document.pdf
      Directory:           pdfcancel backlog ./papers/
      With indexing:       pdfcancel paper.pdf --index my_index
      Full multimodal:     pdfcancel paper.pdf --full
      Custom endpoint:     pdfcancel paper.pdf --api-base http://localhost:11434
    
    \b
    Environment Variables:
      MISTRAL_API_KEY      API key for Mistral's OCR service
      PDFCANCEL_API_BASE   Default API base URL for vlm engine
      PDFCANCEL_API_KEY    API key for vlm endpoint
      PDFCANCEL_OCR_ENGINE  Default OCR engine (mistral/vlm)
      MISTRAL_CA_BUNDLE    CA bundle PEM file for self-signed certificates
    
    See also: man:pdfcancel(1) for detailed documentation.
    """
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


# Register plugins
try:
    from pdfcancel.plugins.zotero import register as _register_zotero
    _register_zotero(main)
except ImportError:
    pass  # Zotero plugin deps not installed


# ── Default command: cancel ───────────────────────────────────────────────────

@main.command(hidden=True)
@click.argument("path", type=click.Path(exists=True))
@click.option("-o", "--output", "output_dir", type=click.Path(), default=None,
              help="Output directory for converted files. "
                   "Default: same directory as input PDF.")
@click.option("--images", is_flag=True,
              help="Extract figures and images from the PDF to a companion "
                   "directory named <input>_images/. Images are referenced in the "
                   "markdown output.")
@click.option("--embed-images", is_flag=True,
              help="Embed all images as base64-encoded data URIs directly in the "
                   "markdown output instead of using external file references.")
@click.option("--plaintext", is_flag=True,
              help="Output plain text instead of markdown. Disables markdown "
                   "formatting, useful for simple text extraction.")
@click.option("--full", is_flag=True,
              help="Full multimodal conversion. In addition to text extraction, "
                   "AI-describes all charts, figures, diagrams, and images using a "
                   "vision model. Requires a vision-capable model. Implies --images.")
@click.option("--full-model", default=None,
              help="Vision model to use for --full image descriptions. "
                   "Default: pixtral-large-latest. For vlm engine: any vision model "
                   "supported by your endpoint (e.g., llava, gemma-4-vision). "
                   "For mistral engine: multimodal model name.")
@click.option("--mistral-url", default=None,
              help="Custom Mistral-compatible server URL for the mistral OCR engine. "
                   "Overrides MISTRAL_BASE_URL environment variable. "
                   "Example: https://api.mistral.ai or http://localhost:11434")
@click.option("--mistral-ca-bundle", type=click.Path(exists=True), default=None,
              help="CA bundle (PEM file) to verify the TLS certificate of a "
                   "self-hosted Mistral server. Overrides MISTRAL_CA_BUNDLE "
                   "environment variable. TLS verification remains enabled.")
@click.option("--ocr-engine", type=click.Choice(["mistral", "vlm"]), default=None,
              help="OCR engine to use. 'mistral' uses Mistral's native OCR API "
                   "(default). 'vlm' uses any OpenAI-compatible vision model via "
                   "/v1/chat/completions endpoint. Auto-switches to 'vlm' when "
                   "--api-base is specified.")
@click.option("--api-base", default=None,
              help="OpenAI-compatible server URL for the vlm engine. "
                   "Examples: http://localhost:11434 (Ollama), "
                   "https://your-vllm-endpoint.com/v1. "
                   "Sets PDFCANCEL_API_BASE environment variable for the session. "
                   "Implicitly sets --ocr-engine to vlm.")
@click.option("--api-key", default=None,
              help="Bearer token for the vlm endpoint authentication. "
                   "Default: 'EMPTY' (no authentication). Overrides PDFCANCEL_API_KEY "
                   "environment variable. Use 'sk-...' for OpenAI, 'ollama' not needed "
                   "for local Ollama.")
@click.option("--ca-bundle", type=click.Path(exists=True), default=None,
              help="CA bundle (PEM file) to verify a self-hosted server's TLS "
                   "certificate for either engine. Commonly needed for internal/"
                   "self-signed certificates. Overrides MISTRAL_CA_BUNDLE.")
@click.option("--model", default=None,
              help="Model override for the active OCR engine. "
                   "For mistral engine: OCR model name (default: mistral-ocr-latest). "
                   "For vlm engine: vision model for text extraction "
                   "(default: google/gemma-4-31B-it).")
@click.option("--ocr-dpi", type=int, default=None,
              help="VLM engine only: Resolution in DPI for rasterizing page images. "
                   "Higher values produce better OCR quality but are slower. "
                   "Default: 200. Recommended: 150-300.")
@click.option("--ocr-concurrency", type=int, default=None,
              help="VLM engine only: Number of concurrent page requests to send. "
                   "Increases throughput but requires more API quota. "
                   "Default: 8. Adjust based on your endpoint rate limits.")
@click.option("--ocr-retries", type=int, default=None,
              help="VLM engine only: Number of retry attempts for failed requests. "
                   "Useful for unstable connections or rate-limited endpoints. "
                   "Default: 5.")
@click.option("--ocr-timeout", type=float, default=None,
              help="VLM engine only: Request timeout in seconds for read/write "
                   "operations. Default: 180 (3 minutes). Increase for slow endpoints.")
@click.option("--caffeinate", is_flag=True,
              help="macOS only: Prevents system sleep during conversion. "
                   "Useful for long-running batch conversions on laptops.")
@click.option("--enhance", "enhance_md", type=click.Path(exists=True), default=None,
              help="Enhance an existing markdown file instead of converting a PDF. "
                   "Useful for improving previously converted documents.")
@click.option("--chunks", is_flag=True,
              help="Also produce chunked JSONL output alongside the markdown. "
                   "Each line is a JSON object with text chunk and metadata. "
                   "Required for search indexing.")
@click.option("--chunk-size", type=int, default=1024,
              help="Maximum tokens per chunk when using --chunks. "
                   "Default: 1024. Larger values = fewer chunks, less granularity.")
@click.option("--chunker", type=click.Choice(["recursive", "semantic", "sentence", "hierarchical"]),
              default="recursive",
              help="Chunking strategy for --chunks. 'recursive': split on sentences "
                   "and paragraphs. 'semantic': use sentence embeddings for splitting. "
                   "'sentence': one chunk per sentence. 'hierarchical': preserve document "
                   "structure. Default: recursive.")
@click.option("--index", "index_name", default=None,
              help="Add the converted document to a named search index. "
                   "Creates or appends to the index database. Implies --chunks. "
                   "Use 'pdfcancel indexes' to list existing indexes and 'pdfcancel "
                   "search <index> <query>' to search.")
@click.option("--index-path", type=click.Path(), default=None,
              help="Custom filesystem path for the search index database. "
                   "Default: ~/.local/share/pdfcancel/indexes/<index_name>.db")
@click.option("--no-clean", is_flag=True,
              help="Skip post-OCR cleanup passes. Produces raw OCR output without "
                   "fixing common issues like: repeated headers/footers, "
                   "page numbers, hyphenated line breaks, etc. Faster but less clean.")
@click.option("--preserve-pages", is_flag=True,
              help="Inject page markers into chunks so each chunk carries page_start "
                   "and page_end metadata. Required for page-level citations when "
                   "using --index and --chunks. Essential for legal/academic documents.")
@click.option("--force", is_flag=True,
              help="Force re-processing even if the output file already exists. "
                   "Useful when you need to regenerate with different options.")
@click.option("--verbose", is_flag=True,
              help="Show detailed progress information including per-page status, "
                   "API requests, and timing. Useful for debugging.")
@click.option("--batch", is_flag=True,
              help="Use Mistral Batch API for OCR (50%% cost savings on directory runs). "
                   "Uploads all pages at once for batch processing. "
                   "Only available with mistral engine and Mistral Batch API support.")
@click.option("--exclude", multiple=True,
              help="Exclude PDFs whose filename contains this text. Repeatable. "
                   "Example: --exclude draft --exclude temp. Case-sensitive.")
def cancel(
    path, output_dir, images, embed_images, plaintext, full, full_model,
    mistral_url, mistral_ca_bundle, ocr_engine, api_base, api_key, ca_bundle,
    ocr_dpi, ocr_concurrency, ocr_retries, ocr_timeout, caffeinate,
    enhance_md, model, chunks, chunk_size,
    chunker, index_name, index_path, no_clean, preserve_pages, force, verbose,
    batch, exclude,
):
    """Cancel PDFs — convert to clean markdown.
    
    Converts PDF files to clean, readable markdown format using AI-powered OCR.
    Supports both Mistral's native OCR API and OpenAI-compatible vision models.
    
    \b
    BASIC USAGE:
      pdfcancel document.pdf                     Convert single PDF → document.md
      pdfcancel ./papers/                        Convert all PDFs in a directory
      pdfcancel paper.pdf -o output/             Specify output directory
    
    \b
    MISTRAL ENGINE (default):
      Uses Mistral AI's native OCR (mistral-ocr-latest). Requires MISTRAL_API_KEY.
      
      pdfcancel paper.pdf                        # Uses Mistral's API
      MISTRAL_API_KEY=your_key pdfcancel paper.pdf
    
    \b
    VLM ENGINE (local/self-hosted):
      Uses any OpenAI-compatible vision model. Perfect for local inference.
      
      pdfcancel paper.pdf --api-base http://localhost:11434  # Ollama
      pdfcancel paper.pdf --ocr-engine vlm --model llava:13b
    
    \b
    FULL MULTIMODAL:
      Extract text + AI-describe all figures, charts, and diagrams.
      
      pdfcancel paper.pdf --full                 # AI describes all images
      pdfcancel paper.pdf --full --full-model llava:13b
    
    \b
    SEARCH & INDEXING:
      Create searchable indexes for your document collection.
      
      pdfcancel paper.pdf --index research       # Add to 'research' index
      pdfcancel ./docs/ --index docs --chunks    # Index all docs with chunks
      pdfcancel search research "machine learning"  # Search the index
    
    \b
    BATCH PROCESSING:
      Process multiple PDFs with resumable progress and checkpointing.
      
      pdfcancel backlog ./papers/                # Resumable directory conversion
      pdfcancel status ./papers/                 # Check progress
      pdfcancel backlog ./papers/ --force        # Re-process everything
    
    \b
    ADVANCED OPTIONS:
      pdfcancel paper.pdf --images --embed-images    # Extract & embed images
      pdfcancel paper.pdf --plaintext              # Output as plain text
      pdfcancel paper.pdf --no-clean               # Skip post-processing cleanup
      pdfcancel paper.pdf --ocr-dpi 300            # Higher quality for complex PDFs
      pdfcancel paper.pdf --exclude draft         # Skip files matching pattern
    """
    input_path = Path(path)
    _maybe_reexec_caffeinate(caffeinate)
    settings = Settings()

    # OCR engine + generic OpenAI-compatible (vlm) provider overrides.
    if ocr_engine:
        settings.ocr_engine = ocr_engine
    if api_base:
        from pdfcancel.config import _normalize_base_url
        settings.api_base = _normalize_base_url(api_base)
        if not ocr_engine:
            settings.ocr_engine = "vlm"  # --api-base implies the vlm engine
    if api_key:
        settings.api_key = api_key
    if ocr_dpi:
        settings.ocr_dpi = ocr_dpi
    if ocr_concurrency:
        settings.ocr_concurrency = ocr_concurrency
    if ocr_retries is not None:
        settings.ocr_retries = ocr_retries
    if ocr_timeout is not None:
        settings.ocr_timeout = ocr_timeout

    # TLS CA bundle: --ca-bundle is the neutral alias; --mistral-ca-bundle is
    # kept for back-compat. Either sets the shared bundle used by both engines.
    if ca_bundle:
        settings.mistral_ca_bundle = ca_bundle
    if mistral_ca_bundle:
        settings.mistral_ca_bundle = mistral_ca_bundle
    if mistral_url:
        settings.mistral_base_url = mistral_url

    # --model targets the active engine's model.
    if model:
        if settings.ocr_engine == "vlm":
            settings.vlm_model = model
        else:
            settings.ocr_model = model
    if full_model:
        if settings.ocr_engine == "vlm":
            settings.vlm_full_model = full_model
        else:
            settings.multimodal_model = full_model

    # The vlm engine crops figures locally (--images/--embed-images/--full);
    # --enhance and the Mistral Batch API remain mistral-engine features.
    if settings.ocr_engine == "vlm":
        if enhance_md:
            raise SystemExit(
                "Error: --enhance is a mistral-engine workflow and is not "
                "supported with --ocr-engine vlm."
            )
        if batch:
            console.print(
                "[dim]--batch applies only to the mistral engine; processing "
                "per-file with the vlm engine.[/dim]"
            )
            batch = False

    # --index implies --chunks; set custom index path if provided
    if index_name:
        chunks = True
    if index_path:
        from pdfcancel.index import set_index_path
        set_index_path(Path(index_path))

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

    pdf_paths = gather_pdfs(input_path, exclude=list(exclude) if exclude else None)

    # Install cooperative SIGINT/SIGTERM handling so a long run stops cleanly
    # (in-flight pages checkpoint) rather than dying mid-write.
    with stop_on_signals() as stop:
        if batch and len(pdf_paths) > 1:
            # Batch pipeline: upload all → batch OCR → batch vision → local post-process
            pairs = convert_batch_pipeline(
                pdf_paths, settings,
                output_dir=out, plaintext=plaintext, extract_images=images,
                embed_images=embed_images, full=full, force=force,
                no_clean=no_clean, produce_chunks=chunks,
                chunk_size=chunk_size, chunker_type=chunker, verbose=verbose,
                preserve_pages=preserve_pages,
            )
            _finish_multi(pairs, index_name, chunk_size, chunker, via=" via batch")
        elif len(pdf_paths) == 1:
            from pdfcancel.validate import OcrValidationError
            try:
                result = convert_single(
                    pdf_paths[0], settings,
                    output_dir=out, plaintext=plaintext, extract_images=images,
                    embed_images=embed_images, full=full, force=force,
                    no_clean=no_clean, produce_chunks=chunks,
                    chunk_size=chunk_size, chunker_type=chunker,
                    preserve_pages=preserve_pages, stop=stop,
                )
            except OcrValidationError as e:
                console.print(f"\n[red]Error:[/red] {e}")
                raise SystemExit(1)
            except ConversionIncomplete as e:
                console.print(f"\n[yellow]Incomplete:[/yellow] {e}")
                raise SystemExit(2)
            if index_name:
                _index_file(result, pdf_paths[0].name, index_name, chunk_size, chunker)
            console.print(f"\n[green]Cancelled.[/green] → {result}")
        else:
            pairs = convert_batch(
                pdf_paths, settings,
                output_dir=out, plaintext=plaintext, extract_images=images,
                embed_images=embed_images, full=full, force=force,
                no_clean=no_clean, produce_chunks=chunks,
                chunk_size=chunk_size, chunker_type=chunker, verbose=verbose,
                preserve_pages=preserve_pages, stop=stop,
            )
            _finish_multi(pairs, index_name, chunk_size, chunker)


def _finish_multi(
    pairs: list[tuple[Path, Path | None]],
    index_name: str | None,
    chunk_size: int,
    chunker: str,
    *,
    via: str = "",
) -> None:
    """Index successful conversions and exit non-zero if any file failed."""
    successes = [(pdf, res) for pdf, res in pairs if res is not None]
    failures = [pdf for pdf, res in pairs if res is None]

    if index_name:
        for pdf_path, res_path in successes:
            _index_file(res_path, pdf_path.name, index_name, chunk_size, chunker)

    console.print(f"\n[green]Cancelled {len(successes)} PDF(s){via}.[/green]")
    if failures:
        console.print(
            f"[red]{len(failures)} PDF(s) failed:[/red] "
            + ", ".join(p.name for p in failures)
        )
        raise SystemExit(1)


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

@main.command(help="Search across previously indexed documents. "
               "Supports hybrid, semantic, and text search modes.")
@click.argument("index_name")
@click.argument("query")
@click.option("-k", "--top-k", type=int, default=10,
              help="Number of search results to return. Default: 10. "
                   "Higher values return more matches but may be slower.")
@click.option("--mode", type=click.Choice(["hybrid", "semantic", "text"]),
              default="hybrid",
              help="Search mode. 'hybrid': combine keyword and semantic search (default). "
                   "'semantic': pure embedding-based search for meaning. "
                   "'text': traditional keyword search only.")
@click.option("--context", is_flag=True,
              help="Include neighboring chunks before and after each match for more "
                   "context. Useful for understanding the full passage around a match.")
@click.option("--index-path", type=click.Path(), default=None,
              help="Custom filesystem path for the index database. "
                   "Default: ~/.local/share/pdfcancel/indexes/<index_name>.db")
def search(index_name, query, top_k, mode, context, index_path):
    """Search across previously indexed documents.
    
    Perform semantic and keyword search across previously indexed PDF documents.
    Requires documents to be indexed using --index option during conversion.
    
    \b
    BASIC USAGE:
      pdfcancel search my_index "machine learning"      # Search 'my_index'
      pdfcancel search research "threat intelligence"   # Search 'research' index
    
    \b
    SEARCH MODES:
      Hybrid (default): Combines traditional keyword matching with semantic
                      understanding for best results.
      Semantic:        Pure meaning-based search using embeddings. Best for
                      conceptual queries.
      Text:            Traditional keyword search. Fastest, good for exact matches.
    
    \b
    EXAMPLES:
      pdfcancel search cti "threat intelligence sharing"  # Basic search
      pdfcancel search cti "ATT&CK" --mode semantic       # Semantic-only
      pdfcancel search cti "risk management" -k 5         # Limit to 5 results
      pdfcancel search cti "APT29" --context              # Show surrounding context
      pdfcancel search research "climate change" --mode hybrid --top-k 20
    
    \b
    TIPS:
      - Use quotes for exact phrase matching
      - Hybrid mode works best for most use cases
      - Use --context to see the full passage around matches
      - Combine with --mode semantic for research/academic queries
    """
    from pdfcancel.index import search as do_search, set_index_path

    if index_path:
        set_index_path(Path(index_path))

    results = do_search(query, index_name, top_k=top_k, mode=mode, context=context)

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    console.print(
        f"\n[bold]Results for[/bold] '{query}' "
        f"[dim](index: {index_name}, mode: {mode})[/dim]\n"
    )

    for i, r in enumerate(results, 1):
        w_score = f"{r.get('weighted_score', r['score']):.3f}"
        method = r.get("method", "")
        content_type = r.get("content_type", "")
        source = Path(r["source"]).stem
        section = r.get("section", "")
        doc_title = r.get("doc_title", "")

        # Page citation (available when indexed with --preserve-pages)
        page_start = r.get("page_start")
        page_end = r.get("page_end")
        if page_start is not None:
            if page_end is not None and page_end != page_start:
                page_tag = f" [magenta]pp. {page_start}–{page_end}[/magenta]"
            else:
                page_tag = f" [magenta]p. {page_start}[/magenta]"
        else:
            page_tag = ""

        # Header line with score, method, content type, source, page
        type_tag = f" [{content_type}]" if content_type and content_type != "prose" else ""
        console.print(
            f"[bold cyan]#{i}[/bold cyan] [green]{w_score}[/green] "
            f"[{method}]{type_tag}  [dim]{source}[/dim]{page_tag}"
        )
        if doc_title and doc_title != source:
            console.print(f"   [dim]┃ {doc_title}[/dim]")
        if section:
            console.print(f"   [dim]§ {section}[/dim]")

        # Context before (dimmed)
        if context and r.get("context_before"):
            ctx = r["context_before"][:120].replace("\n", " ").strip()
            console.print(f"   [dim]┌ {ctx}...[/dim]")

        # Main chunk text
        preview = r["text"][:250].replace("\n", " ").strip()
        if len(r["text"]) > 250:
            preview += "..."
        console.print(f"   {preview}")

        # Context after (dimmed)
        if context and r.get("context_after"):
            ctx = r["context_after"][:120].replace("\n", " ").strip()
            console.print(f"   [dim]└ {ctx}...[/dim]")

        if context and r.get("figure_context"):
            ctx = r["figure_context"][:160].replace("\n", " ").strip()
            console.print(f"   [dim]▣ {ctx}...[/dim]")

        console.print()


# ── Indexes command ───────────────────────────────────────────────────────

@main.command(help="List all search indexes with their document counts. "
               "Shows chunk statistics for each indexed source.")
def indexes():
    """List all search indexes with their document counts.
    
    Display information about all existing search indexes, including the
    documents they contain and the number of chunks indexed from each.
    
    \b
    USAGE:
      pdfcancel indexes                          # List all indexes
    
    \b
    OUTPUT:
      For each index, shows a table with:
        - Source: The original document filename
        - Chunks: Number of text chunks from that document
        - Total: Summary of total chunks in the index
    
    \b
    EXAMPLE:
      $ pdfcancel indexes
      
      Index: cti
      ┌─────────────────┬─────────┐
      │ Source          │  Chunks │
      │ threat_report.pdf │    150 │
      │ malware_analysis.md │     85 │
      │ TOTAL           │   [bold]235[/bold] │
      └─────────────────┴─────────┘
      
    \b
    RELATED COMMANDS:
      pdfcancel paper.pdf --index my_index      # Create/index a document
      pdfcancel search my_index "query"         # Search an index
    """
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


# ── Backlog command (resumable directory run) ────────────────────────────

@main.command(help="Resumable batch PDF conversion for directories. "
               "Checkpointed, signal-safe, with progress tracking.")
@click.argument("path", type=click.Path(exists=True, file_okay=False))
@click.option("-o", "--output", "output_dir", type=click.Path(), default=None,
              help="Output directory for all converted files. "
                   "Default: each PDF is converted alongside its source file.")
@click.option("--exclude", multiple=True,
              help="Exclude PDFs whose filename contains this text. Can be repeated. "
                   "Example: --exclude draft --exclude temp skips files with "
                   "'draft' or 'temp' in their names.")
@click.option("--ocr-engine", type=click.Choice(["mistral", "vlm"]), default=None,
              help="OCR engine to use. 'mistral' (default) uses Mistral's native OCR API. "
                   "'vlm' uses OpenAI-compatible vision models. Auto-switches to 'vlm' "
                   "when --api-base is specified.")
@click.option("--api-base", default=None,
              help="OpenAI-compatible server URL for vlm engine. "
                   "Examples: http://localhost:11434 (Ollama), "
                   "http://localhost:8000/v1 (vLLM). Implicitly sets --ocr-engine vlm.")
@click.option("--api-key", default=None,
              help="Bearer token for vlm endpoint authentication. "
                   "Not needed for local Ollama. Overrides PDFCANCEL_API_KEY env var.")
@click.option("--ca-bundle", type=click.Path(exists=True), default=None,
              help="CA bundle (PEM file) for verifying self-hosted server TLS certificates. "
                   "Required for HTTPS endpoints with self-signed certificates.")
@click.option("--model", default=None,
              help="Model override for the active engine. "
                   "For mistral: OCR model (default: mistral-ocr-latest). "
                   "For vlm: vision model (default: google/gemma-4-31B-it).")
@click.option("--ocr-dpi", type=int, default=None,
              help="VLM only: Page rasterization DPI. Higher = better quality but slower. "
                   "Default: 200. Range: 72-600 recommended.")
@click.option("--ocr-concurrency", type=int, default=None,
              help="VLM only: Number of concurrent page requests. "
                   "Default: 8. Reduce if hitting rate limits.")
@click.option("--ocr-retries", type=int, default=None,
              help="VLM only: Retry attempts for failed requests. "
                   "Default: 5. Increase for unstable connections.")
@click.option("--ocr-timeout", type=float, default=None,
              help="VLM only: Request timeout in seconds. "
                   "Default: 180 (3 minutes). Increase for slow endpoints or large pages.")
@click.option("--chunks", is_flag=True,
              help="Also produce chunked JSONL output. Each chunk is a JSON line "
                   "with text and metadata. Required for search indexing.")
@click.option("--chunk-size", type=int, default=1024,
              help="Maximum tokens per chunk. Default: 1024. "
                   "Larger values create fewer, larger chunks.")
@click.option("--chunker",
              type=click.Choice(["recursive", "semantic", "sentence", "hierarchical"]),
              default="recursive",
              help="Chunking strategy. 'recursive': balanced sentence/paragraph splits. "
                   "'semantic': uses embeddings for content-aware splitting. "
                   "'sentence': one chunk per sentence (most granular). "
                   "'hierarchical': preserves document structure. Default: recursive.")
@click.option("--preserve-pages", is_flag=True,
              help="Inject page markers into chunks with page_start/page_end metadata. "
                   "Essential for legal/academic documents needing page-level citations.")
@click.option("--no-clean", is_flag=True,
              help="Skip post-OCR cleanup. Produces raw output without fixing "
                   "headers/footers, page numbers, hyphenated breaks. Faster but less polished.")
@click.option("--plaintext", is_flag=True,
              help="Output plain text instead of markdown. Disables all formatting.")
@click.option("--images", is_flag=True,
              help="Extract figures and images to <name>_images/ directories.")
@click.option("--full", is_flag=True,
              help="Full multimodal: extract text + AI-describe charts, figures, diagrams. "
                   "Requires vision model. Implies --images. Adds AI-generated descriptions "
                   "as alt-text for each figure.")
@click.option("--full-model", default=None,
              help="Vision model for --full descriptions. "
                   "Default: pixtral-large-latest. For vlm: any vision model "
                   "(llava, gemma-4-vision, etc.).")
@click.option("--force", is_flag=True,
              help="Force re-processing of all PDFs, even if already converted. "
                   "Useful when changing options or after fixing configuration issues.")
@click.option("--caffeinate", is_flag=True,
              help="macOS only: Prevent system sleep during the run. "
                   "Recommended for long-running conversions on laptops.")
def backlog(path, output_dir, exclude, ocr_engine, api_base, api_key, ca_bundle,
            model, ocr_dpi, ocr_concurrency, ocr_retries, ocr_timeout, chunks,
            chunk_size, chunker, preserve_pages, no_clean, plaintext, images,
            full, full_model, force, caffeinate):
    """Resumable batch PDF conversion for directories.
    
    Resumable batch PDF conversion with checkpointing. Already-converted files
    are skipped on subsequent runs. Each page is cached as it transcribes, so
    Ctrl-C (SIGINT) or SIGTERM stops cleanly and re-running resumes from where
    it left off.
    
    Progress is tracked in a run-state file (<dir>/.pdfcancel_run.json) that
    can be inspected with: pdfcancel status <dir>
    
    \b
    RESUMABLE WORKFLOW:
      1. pdfcancel backlog ./papers/           # Start conversion
      2. Ctrl-C to interrupt at any time       # Stops cleanly
      3. pdfcancel backlog ./papers/           # Resumes from checkpoint
      4. pdfcancel status ./papers/            # Check current progress
    
    \b
    EXAMPLES:
      pdfcancel backlog ./documents/           # Convert all PDFs in directory
      pdfcancel backlog ./papers/ -o ./markdown/ # Output to different directory
      pdfcancel backlog ./books/ --full        # Full multimodal conversion
      pdfcancel backlog ./scans/ --vr-engine vlm --api-base http://localhost:11434
      pdfcancel backlog ./all/ --exclude draft  # Skip draft files
      pdfcancel backlog ./corpus/ --index research  # Convert and index for search
    
    \b
    PERFORMANCE TIPS:
      - Use --ocr-concurrency to match your API rate limits
      - Increase --ocr-timeout for slow endpoints or large pages
      - Use --exclude to skip files you don't need
      - Monitor with --verbose for detailed progress
    """
    _maybe_reexec_caffeinate(caffeinate)
    root = Path(path)
    settings = Settings()
    _apply_vlm_overrides(
        settings, ocr_engine=ocr_engine, api_base=api_base, api_key=api_key,
        ca_bundle=ca_bundle, model=model, ocr_dpi=ocr_dpi,
        ocr_concurrency=ocr_concurrency, ocr_retries=ocr_retries,
        ocr_timeout=ocr_timeout,
    )
    if full_model:
        if settings.ocr_engine == "vlm":
            settings.vlm_full_model = full_model
        else:
            settings.multimodal_model = full_model
    images = images or full  # --full implies --images, as in the main command

    pdf_paths = gather_pdfs(root, exclude=list(exclude) if exclude else None)
    out_ext = ".txt" if plaintext else ".md"
    state = RunState(root / RUN_STATE_NAME, total=len(pdf_paths))
    console.print(f"[bold]Backlog:[/bold] {len(pdf_paths)} PDF(s) under {root}")
    # One shared output folder: same-named PDFs need distinct output names.
    stems = output_stems(pdf_paths) if output_dir else {p: p.stem for p in pdf_paths}

    interrupted = False
    with stop_on_signals() as stop:
        for idx, pdf in enumerate(pdf_paths, 1):
            if stop.is_set():
                interrupted = True
                break
            out_dir = Path(output_dir) if output_dir else pdf.parent
            existing = out_dir / f"{stems[pdf]}{out_ext}"
            if not force and existing.exists():
                state.record(pdf.name, "skipped")
                continue
            state.set_current(pdf.name)
            console.print(f"  [dim]({idx}/{len(pdf_paths)})[/dim] {pdf.name}")
            try:
                convert_single(
                    pdf, settings, output_dir=out_dir, plaintext=plaintext,
                    extract_images=images, full=full,
                    no_clean=no_clean, produce_chunks=chunks,
                    chunk_size=chunk_size, chunker_type=chunker,
                    preserve_pages=preserve_pages, force=force, stop=stop,
                    output_stem=stems[pdf],
                )
                state.record(pdf.name, "done")
            except ConversionIncomplete as e:
                state.record(pdf.name, "incomplete")
                console.print(f"    [yellow]◐ {e}[/yellow]")
                if e.interrupted:
                    interrupted = True
                    break
            except Exception as e:  # noqa: BLE001 - keep going on a bad file
                state.record(pdf.name, "failed")
                console.print(f"    [red]✗ {e}[/red]")
    state.finish()

    d = state.data
    console.print(
        f"\n[bold]Done:[/bold] {d['done']} converted, {d['skipped']} skipped, "
        f"{d['incomplete']} incomplete, {d['failed']} failed (of {d['total']})"
    )
    if d["failures"]:
        console.print("  [red]Failed:[/red] " + ", ".join(d["failures"]))
    if interrupted:
        console.print(
            "  [yellow]Interrupted — re-run the same command to resume.[/yellow]"
        )
    if interrupted or d["incomplete"] or d["failed"]:
        raise SystemExit(2)


# ── Status command ────────────────────────────────────────────

@main.command(help="Show progress of a backlog conversion run. "
               "Displays detailed statistics for a directory.")
@click.argument("path", type=click.Path(exists=True, file_okay=False))
def status(path):
    """Show progress of a backlog conversion run."
    
    Display the current state of a resumable batch conversion started with
    'pdfcancel backlog'. Shows which files have been processed, skipped,
    failed, or remain to be converted.
    
    \b
    USAGE:
      pdfcancel status ./papers/                 # Check conversion progress
    
    \b
    OUTPUT:
      Displays a status table with:
        - State: 'finished' or 'in progress'
        - Done: Successfully converted files
        - Skipped: Already existed (no --force)
        - Incomplete: Partially processed (can resume)
        - Failed: Error during conversion
        - Remaining: Files yet to process
        - Total: Total files in directory
        - Current: Currently processing file
        - Elapsed: Time since run started
        - Updated: Last update timestamp
    
    \b
    RESUMING AFTER INTERRUPTION:
      1. Start:  pdfcancel backlog ./papers/
      2. Stop:   Ctrl-C (stops cleanly, checkpoints saved)
      3. Resume: pdfcancel backlog ./papers/ (continues from checkpoint)
      4. Check:  pdfcancel status ./papers/ (view progress at any time)
    
    \b
    EXAMPLE:
      $ pdfcancel backlog ./documents/
      [Output truncated, Ctrl-C pressed]
      
      $ pdfcancel status ./documents/
      Backlog status — documents
      ┌──────────────┬───────┐
      │ Field        │ Value │
      │ State        │ in progress │
      │ Done         │ 12    │
      │ Skipped      │ 2     │
      │ Incomplete   │ 1     │
      │ Failed       │ 0     │
      │ Remaining    │ 5     │
      │ Total        │ 20    │
      │ Current      │ report_q2.pdf │
      │ Elapsed      │ 2h34m │
      └──────────────┴───────┘
    """
    state_path = Path(path) / RUN_STATE_NAME
    if not state_path.exists():
        console.print(f"[yellow]No run state found at {state_path}[/yellow]")
        console.print("  Start one with:  pdfcancel backlog <dir> ...")
        return
    data = json.loads(state_path.read_text())

    elapsed = "?"
    try:
        start = datetime.fromisoformat(data["started_at"])
        updated = datetime.fromisoformat(data["updated_at"])
        secs = int((updated - start).total_seconds())
        elapsed = f"{secs // 3600}h{(secs % 3600) // 60:02d}m"
    except (KeyError, ValueError):
        pass

    total = data.get("total", 0)
    accounted = (
        data.get("done", 0) + data.get("skipped", 0)
        + data.get("incomplete", 0) + data.get("failed", 0)
    )
    remaining = max(0, total - accounted)

    table = Table(title=f"Backlog status — {Path(path).name}", show_header=False)
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    table.add_row("State", "finished" if data.get("finished") else "in progress")
    table.add_row("Done", str(data.get("done", 0)))
    table.add_row("Skipped", str(data.get("skipped", 0)))
    table.add_row("Incomplete", str(data.get("incomplete", 0)))
    table.add_row("Failed", str(data.get("failed", 0)))
    table.add_row("Remaining", str(remaining))
    table.add_row("Total", str(total))
    table.add_row("Current", str(data.get("current") or "—"))
    table.add_row("Elapsed", elapsed)
    table.add_row("Updated", data.get("updated_at", "?"))
    console.print(table)
    if data.get("failures"):
        console.print("  [red]Failed:[/red] " + ", ".join(data["failures"]))
