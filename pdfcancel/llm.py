"""OpenAI-compatible vision backend for the ``vlm`` OCR engine.

This module speaks the OpenAI ``/v1/chat/completions`` interface, so it works
with any compatible server the user has — a local vLLM instance, a LiteLLM
proxy, Ollama, OpenAI, or a dedicated OCR VLM. A chat model cannot
ingest a PDF or expose ``/v1/ocr``, so OCR here means: rasterize each page to
an image and transcribe it to Markdown via the ``chat_vision`` primitive, which
also powers ``--full`` figure descriptions (figures are cropped locally by
``pdfcancel.figures``).

The Mistral ``/v1/ocr`` fast-path lives elsewhere (the default ``mistral``
engine); nothing in this module touches it.
"""

from __future__ import annotations

import base64
import hashlib
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from threading import Event

from rich.console import Console

from pdfcancel.config import Settings
from pdfcancel.errors import ConversionIncomplete
from pdfcancel.images import OcrResult
from pdfcancel.pagecache import PageCache
from pdfcancel.pages import join_pages
from pdfcancel.retry import with_retry
from pdfcancel.vlmtext import clean_latex, clean_prefix, collapse_loops, find_loop

console = Console()

# Transcription prompt: the page image in, clean Markdown out. Kept strict so
# the model does not summarize, translate, or wrap the whole page in a fence.
# Tuned to be thorough on dense academic/technical pages: the previous terse
# prompt let the vlm engine drop text embedded in figures/charts and
# under-transcribe long reference lists, diverging from the Mistral baseline.
VLM_OCR_PROMPT = (
    "You are a meticulous OCR engine for dense academic and technical "
    "documents. Transcribe this page into clean, GitHub-flavored Markdown, "
    "capturing all of its textual content completely and verbatim.\n"
    "- Preserve the natural reading order, headings, paragraphs, and lists. "
    "For multi-column layouts, transcribe each column fully in reading order "
    "(left column before right).\n"
    "- Transcribe every word exactly; do not summarize, paraphrase, truncate, "
    "translate, or comment. Transcribe the whole page through its final line.\n"
    "- Render tables as GitHub-flavored Markdown tables, preserving every row "
    "and cell.\n"
    "- Use LaTeX ($...$ or $$...$$) only for mathematical expressions and "
    "equations. Write ordinary words, numbers, symbols (such as ✓, ×, "
    "→), and spacing as plain text; never use math mode, \\text{}, or "
    "\\quad for layout or alignment.\n"
    "- Transcribe the text inside figures, charts, diagrams, and infographics "
    "(titles, axis labels, legends, data values, and annotations), and include "
    "each figure's and table's caption. Use a single `![figure](figure)` "
    "placeholder only for a purely pictorial image with no legible text; never "
    "invent content that is not visibly present.\n"
    "- Include footnotes, endnotes, in-text citations, and the full reference "
    "or bibliography list.\n"
    "- Output only the page's Markdown: no surrounding code fence, no preamble, "
    "and no trailing notes."
)

# Appended for a born-digital page: its embedded text layer, so exact strings
# (identifiers like CVE vs CWE, names, numbers) come from the PDF rather than
# from the model's reading of pixels. Layout still comes from the image.
VLM_TEXT_LAYER_PROMPT = (
    "\n\nThe page's embedded PDF text layer follows. Treat it as the "
    "authoritative spelling of every word, name, identifier, acronym, and "
    "number on the page. Take reading order, headings, tables, and structure "
    "from the image: the text layer may be out of order, break words at line "
    "ends, or lack text drawn inside images, which you transcribe from the "
    "image. Do not copy text that is not visible on the page.\n\n"
    'TEXT LAYER:\n"""\n{text}\n"""'
)
TEXT_LAYER_MIN_CHARS = 200     # less than this is a scan, a figure page, or a cover
TEXT_LAYER_MAX_CHARS = 12000   # keep the hint within a modest prompt budget

# A page transcription that loops (pdfcancel.vlmtext) is retried once, then
# continued from its last clean text up to this many times. Sampling
# penalties were tried and rejected: they turn loops into letter soup.
VLM_LOOP_CONTINUATIONS = 2
VLM_CONTINUE_PROMPT = (
    "Your transcription stopped partway and began repeating itself. Continue "
    "transcribing the page from exactly where the text above ends, through "
    "the page's final line. Output only the continuation; do not repeat "
    "anything already transcribed."
)

# Per-page output cap. Dense academic pages (figures with embedded text, long
# reference lists) can exceed the old 1200-token cap and get truncated
# mid-page; give the model room to transcribe the whole page.
VLM_OCR_MAX_TOKENS = 4096

# A short fingerprint of the prompt. It is recorded in each page cache so that
# changing the prompt automatically invalidates stale cached pages.
VLM_PROMPT_SHA = hashlib.sha256(
    (VLM_OCR_PROMPT + VLM_TEXT_LAYER_PROMPT).encode("utf-8")
).hexdigest()[:12]


def _file_sha256(path: Path) -> str:
    """Stream a file's SHA-256 (used to key the page cache to its source PDF)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _auth_headers(settings: Settings) -> dict[str, str]:
    """Bearer-auth headers (vLLM accepts any token; default is ``EMPTY``)."""
    return {"Authorization": f"Bearer {settings.api_key or 'EMPTY'}"}


def _require_api_base(settings: Settings) -> str:
    """Return the normalized api_base or raise a clear error."""
    if not settings.api_base:
        raise SystemExit(
            "Error: the 'vlm' OCR engine needs a server URL.\n"
            "  Set it with --api-base or PDFCANCEL_API_BASE, e.g.\n"
            "  --api-base https://your-host/litellm/v1"
        )
    return settings.api_base


def _extract_content(data: dict) -> str:
    """Pull assistant text out of a chat/completions response payload."""
    try:
        message = data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise SystemExit(
            f"Error: unexpected chat/completions response shape: {data!r}"
        ) from exc
    content = message.get("content", "")
    # Some servers return content as a list of typed parts rather than a string.
    if isinstance(content, list):
        content = "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict)
        )
    return (content or "").strip()


def resolve_vlm_model(settings: Settings, *, http_client=None) -> str:
    """Resolve the model id for the vlm engine.

    Returns ``vlm_model`` when set. Otherwise queries ``/v1/models`` and uses
    the result only when the endpoint exposes exactly one model; a multi-model
    proxy (e.g. LiteLLM) must be given an explicit ``--model``.
    """
    if settings.vlm_model:
        return settings.vlm_model

    base = _require_api_base(settings)
    url = f"{base}/v1/models"
    own_client = http_client is None
    client = http_client or settings.build_vision_http_client()
    try:
        def _do():
            resp = client.get(url, headers=_auth_headers(settings))
            resp.raise_for_status()
            return resp.json()

        data = with_retry(
            _do,
            description="list models",
            attempts=settings.ocr_retries,
            base_delay=settings.ocr_backoff,
        )
    finally:
        if own_client:
            client.close()

    models = [
        m.get("id")
        for m in (data.get("data") or [])
        if isinstance(m, dict) and m.get("id")
    ]
    if len(models) == 1:
        return models[0]
    if not models:
        raise SystemExit(
            f"Error: no models returned by {url}; "
            "name one with --model (PDFCANCEL_VLM_MODEL)."
        )
    raise SystemExit(
        "Error: the endpoint exposes multiple models; choose one with --model "
        "(PDFCANCEL_VLM_MODEL).\n  Available: " + ", ".join(sorted(models))
    )


def chat_vision(
    settings: Settings,
    *,
    image_bytes: bytes,
    prompt: str,
    model: str | None = None,
    max_tokens: int = 1200,
    mime: str = "image/png",
    http_client=None,
    history: list[dict] | None = None,
) -> str:
    """Send one image + prompt to an OpenAI-compatible chat model, return text.

    When ``http_client`` is provided the caller owns its lifecycle (useful for
    reusing one connection pool across many pages); otherwise a client is built
    and closed per call. ``history`` appends later turns (e.g. a partial
    answer and a request to continue it) after the image message.
    """
    base = _require_api_base(settings)
    model = model or resolve_vlm_model(settings, http_client=http_client)

    b64 = base64.b64encode(image_bytes).decode("ascii")
    data_uri = f"data:{mime};base64,{b64}"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            },
            *(history or []),
        ],
        "temperature": 0.1,
        "max_tokens": max_tokens,
    }
    url = f"{base}/v1/chat/completions"

    own_client = http_client is None
    client = http_client or settings.build_vision_http_client()
    try:
        def _do():
            resp = client.post(url, json=payload, headers=_auth_headers(settings))
            resp.raise_for_status()
            return resp.json()

        data = with_retry(
            _do,
            description=f"vlm chat ({model})",
            attempts=settings.ocr_retries,
            base_delay=settings.ocr_backoff,
        )
    finally:
        if own_client:
            client.close()

    return _extract_content(data)


def rasterize_pdf(pdf_path: Path, dpi: int = 200) -> list[bytes]:
    """Render each PDF page to PNG bytes via PyMuPDF (lazy import)."""
    try:
        import pymupdf as fitz  # PyMuPDF >= 1.24 (preferred import name)
    except ImportError:
        try:
            import fitz  # older PyMuPDF alias
        except ImportError as exc:
            raise SystemExit(
                "Error: the 'vlm' OCR engine needs PyMuPDF to rasterize PDF "
                "pages.\n  Install it with:  pip install pymupdf\n"
                "  (or install the extra:  pip install 'pdfcancel[vlm]')"
            ) from exc

    zoom = max(dpi, 1) / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    pages: list[bytes] = []
    with fitz.open(str(pdf_path)) as doc:
        for page in doc:
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            pages.append(pix.tobytes("png"))
    return pages


def _usable_text_layer(text: str) -> str | None:
    """The page's text layer when it is real prose, else ``None``.

    Scans have none; broken font encodings yield replacement characters or
    symbol soup, which would mislead rather than help.
    """
    text = text.strip()
    if len(text) < TEXT_LAYER_MIN_CHARS or text.count("\ufffd") > 0.01 * len(text):
        return None
    tokens = text.split()
    wordy = sum(1 for t in tokens if sum(c.isalpha() for c in t) >= 2)
    if wordy < 0.5 * len(tokens):
        return None
    return text[:TEXT_LAYER_MAX_CHARS]


def page_text_layers(pdf_path: Path) -> list[str | None]:
    """Each page's usable embedded text (see :func:`_usable_text_layer`)."""
    import pymupdf

    flags = pymupdf.TEXTFLAGS_TEXT | pymupdf.TEXT_DEHYPHENATE
    with pymupdf.open(str(pdf_path)) as doc:
        return [_usable_text_layer(page.get_text("text", flags=flags)) for page in doc]


def ocr_prompt(text_layer: str | None) -> str:
    """The transcription prompt, with the page's text layer when it has one."""
    if not text_layer:
        return VLM_OCR_PROMPT
    return VLM_OCR_PROMPT + VLM_TEXT_LAYER_PROMPT.format(text=text_layer)


def repair_page(markdown: str) -> str:
    """Collapse repetition loops and unwrap layout-only LaTeX in one page."""
    return clean_latex(collapse_loops(markdown))


def _transcribe_page(ask, page_no: int) -> str:
    """Transcribe one page with ``ask``, recovering from repetition loops.

    Loops are sampling accidents, so a looping page is retried once. If that
    loops too, the transcription is continued from its last clean text: a
    loop runs to the token cap, so everything after it is otherwise lost.
    Whatever loops in the end is cut by :func:`repair_page`.
    """
    md = ask()
    if not find_loop(md):
        return md
    console.print(f"    [yellow]page {page_no}: repetition loop, retrying[/yellow]")
    retry = ask()
    if not find_loop(retry):
        return retry
    text = max((md, retry), key=lambda t: len(clean_prefix(t)))
    for _ in range(VLM_LOOP_CONTINUATIONS):
        done = clean_prefix(text)
        console.print(f"    [yellow]page {page_no}: continuing after a loop[/yellow]")
        more = ask([{"role": "assistant", "content": done},
                    {"role": "user", "content": VLM_CONTINUE_PROMPT}])
        added = clean_prefix(more)
        text = done + "\n" + more.lstrip()
        if not find_loop(more) or len(added.split()) < 20:
            break
    return text


def vlm_ocr_pdf(
    pdf_path: Path,
    settings: Settings,
    *,
    preserve_pages: bool = False,
    cache_dir: Path | None = None,
    stop: Event | None = None,
    keep_cache: bool = False,
) -> OcrResult:
    """OCR a PDF with the vlm engine: rasterize pages, transcribe, assemble.

    Resumable and fail-safe for long runs:

    - Each page is checkpointed to a per-document cache (under ``cache_dir``)
      as soon as it transcribes, so a crash/kill/sleep costs at most the pages
      still in flight; a re-run reloads finished pages and only does the rest.
    - A page that fails after exhausting retries does not abort the document;
      the remaining pages still run and the finished ones stay cached.
    - When ``stop`` is set (SIGINT/SIGTERM), no new pages are scheduled, the
      in-flight pages finish and checkpoint, and the run ends cleanly.

    If any page is still missing at the end (interrupted or failed), raises
    :class:`~pdfcancel.errors.ConversionIncomplete` without assembling output,
    leaving the cache in place to resume. On full success the cache is removed,
    unless ``keep_cache`` is set: then ``OcrResult.cleanup`` removes it, and the
    caller runs that once later, failure-prone steps (figure descriptions)
    have finished and the output is written.

    ``images`` is always empty: page transcription does not produce cropped
    figure images (that is a ``mistral``-engine feature).
    """
    page_images = rasterize_pdf(pdf_path, dpi=settings.ocr_dpi)
    n = len(page_images)
    text_layers: list[str | None] = [None] * n
    if settings.vlm_text_layer:
        try:
            layers = page_text_layers(pdf_path)
        except Exception as exc:  # the hint is optional; OCR from pixels alone
            console.print(f"    [yellow]No text layer hints ({type(exc).__name__}: {exc})[/yellow]")
        else:
            if len(layers) == n:
                text_layers = layers
    if n == 0:
        return OcrResult(markdown="", images={}, page_markdowns=[])

    model = resolve_vlm_model(settings)
    workers = max(1, int(settings.ocr_concurrency or 1))

    # Page cache keyed by the inputs that determine the output; a change to any
    # of them resets the cache so stale pages are never reused.
    cache: PageCache | None = None
    page_markdowns: list[str | None] = [None] * n
    if cache_dir is not None:
        meta = {
            "pdf_sha256": _file_sha256(pdf_path),
            "dpi": int(settings.ocr_dpi),
            "model": model,
            "prompt_sha": VLM_PROMPT_SHA,
            "text_layer": bool(settings.vlm_text_layer),
            "page_count": n,
        }
        cache = PageCache(cache_dir, stem=pdf_path.stem, meta=meta)
        cache.validate_or_reset()
        for i in range(n):
            cached = cache.load_page(i)
            if cached is not None:
                page_markdowns[i] = cached

    completed = sum(1 for md in page_markdowns if md is not None)
    console.print(
        f"    Transcribing {n} page(s) via vlm model '{model}' "
        f"(dpi={settings.ocr_dpi}, workers={workers})"
    )
    if completed:
        console.print(f"    Resuming \u2014 {completed}/{n} page(s) already cached")

    last_emit = time.monotonic()

    def _emit(force: bool = False) -> None:
        nonlocal last_emit
        now = time.monotonic()
        if force or now - last_emit >= 10.0:
            pct = int(completed * 100 / n) if n else 100
            console.print(f"    \u2026 {completed}/{n} pages ({pct}%)")
            last_emit = now

    todo = [i for i in range(n) if page_markdowns[i] is None]
    failed: list[int] = []
    interrupted = False

    if todo:
        if stop is not None and stop.is_set():
            interrupted = True
        else:
            client = settings.build_vision_http_client()

            def _transcribe(index: int) -> tuple[int, str]:
                def ask(history: list[dict] | None = None) -> str:
                    return chat_vision(
                        settings,
                        image_bytes=page_images[index],
                        prompt=ocr_prompt(text_layers[index]),
                        model=model,
                        max_tokens=VLM_OCR_MAX_TOKENS,
                        mime="image/png",
                        http_client=client,
                        **({"history": history} if history else {}),
                    )

                return index, _transcribe_page(ask, index + 1)

            try:
                pending = iter(todo)
                inflight: dict = {}
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    # Prime up to `workers` pages, then refill one-for-one as
                    # each completes so a stop can halt new work immediately.
                    for _ in range(workers):
                        try:
                            idx = next(pending)
                        except StopIteration:
                            break
                        inflight[executor.submit(_transcribe, idx)] = idx

                    while inflight:
                        done_set, _ = wait(inflight, return_when=FIRST_COMPLETED)
                        for fut in done_set:
                            idx = inflight.pop(fut)
                            try:
                                _, md = fut.result()
                                page_markdowns[idx] = md
                                if cache is not None:
                                    cache.save_page(idx, md)
                                completed += 1
                            except Exception as exc:
                                failed.append(idx)
                                console.print(
                                    f"    [red]page {idx + 1} failed[/red] "
                                    f"({type(exc).__name__}: {exc})"
                                )
                            _emit()

                        if stop is not None and stop.is_set():
                            interrupted = True
                            continue  # drain in-flight; schedule nothing new

                        while len(inflight) < workers:
                            try:
                                idx = next(pending)
                            except StopIteration:
                                break
                            inflight[executor.submit(_transcribe, idx)] = idx
            finally:
                client.close()

    _emit(force=True)

    missing = [i + 1 for i in range(n) if page_markdowns[i] is None]
    if missing:
        raise ConversionIncomplete(
            total=n,
            completed=n - len(missing),
            failed_pages=sorted(missing),
            interrupted=interrupted,
        )

    # Repair at assembly, so pages cached by an earlier run are repaired too.
    page_markdowns = [repair_page(md) for md in page_markdowns]
    markdown = join_pages(page_markdowns, preserve_pages=preserve_pages)
    result = OcrResult(markdown=markdown, images={}, page_markdowns=page_markdowns)
    if cache is not None:
        if keep_cache:
            result.cleanup = cache.cleanup  # caller removes it after writing output
        else:
            cache.cleanup()
    return result
