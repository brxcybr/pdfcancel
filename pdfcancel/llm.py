"""OpenAI-compatible vision backend for the ``vlm`` OCR engine.

This module speaks the OpenAI ``/v1/chat/completions`` interface, so it works
with any compatible server the user has — a local vLLM instance, the WIRE
LiteLLM proxy, Ollama, OpenAI, or a dedicated OCR VLM. A chat model cannot
ingest a PDF or expose ``/v1/ocr``, so OCR here means: rasterize each page to
an image and transcribe it to Markdown via the same ``chat_vision`` primitive
that can later power ``--full``.

The Mistral ``/v1/ocr`` fast-path lives elsewhere (the default ``mistral``
engine); nothing in this module touches it.
"""

from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rich.console import Console

from pdfcancel.config import Settings
from pdfcancel.images import OcrResult
from pdfcancel.pages import join_pages
from pdfcancel.retry import with_retry

console = Console()

# Transcription prompt: the page image in, clean Markdown out. Kept strict so
# the model does not summarize, translate, or wrap the whole page in a fence.
VLM_OCR_PROMPT = (
    "You are an OCR engine. Transcribe this document page into clean, "
    "GitHub-flavored Markdown.\n"
    "- Preserve the natural reading order, headings, paragraphs, and lists.\n"
    "- Render tables as Markdown tables and math as LaTeX ($...$ or $$...$$).\n"
    "- Transcribe the text exactly; do not summarize, translate, or comment.\n"
    "- For a figure, chart, or photo, emit a single placeholder line "
    "`![figure](figure)` instead of inventing its contents.\n"
    "- Output only the page's Markdown: no surrounding code fence, no preamble, "
    "and no trailing notes."
)


def _auth_headers(settings: Settings) -> dict[str, str]:
    """Bearer-auth headers (vLLM accepts any token; default is ``EMPTY``)."""
    return {"Authorization": f"Bearer {settings.api_key or 'EMPTY'}"}


def _require_api_base(settings: Settings) -> str:
    """Return the normalized api_base or raise a clear error."""
    if not settings.api_base:
        raise SystemExit(
            "Error: the 'vlm' OCR engine needs a server URL.\n"
            "  Set it with --api-base or PDFCANCEL_API_BASE, e.g.\n"
            "  --api-base https://api.wirecui.westpoint.edu/litellm/v1"
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

        data = with_retry(_do, description="list models")
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
) -> str:
    """Send one image + prompt to an OpenAI-compatible chat model, return text.

    When ``http_client`` is provided the caller owns its lifecycle (useful for
    reusing one connection pool across many pages); otherwise a client is built
    and closed per call.
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
            }
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

        data = with_retry(_do, description=f"vlm chat ({model})")
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


def vlm_ocr_pdf(
    pdf_path: Path,
    settings: Settings,
    *,
    preserve_pages: bool = False,
) -> OcrResult:
    """OCR a PDF with the vlm engine: rasterize pages, transcribe, assemble.

    Returns an :class:`~pdfcancel.images.OcrResult` carrying the joined
    markdown and per-page markdown. ``images`` is always empty: page
    transcription does not produce cropped figure images (that is a
    ``mistral``-engine feature).
    """
    page_images = rasterize_pdf(pdf_path, dpi=settings.ocr_dpi)
    if not page_images:
        return OcrResult(markdown="", images={}, page_markdowns=[])

    model = resolve_vlm_model(settings)
    workers = max(1, int(settings.ocr_concurrency or 1))
    console.print(
        f"    Transcribing {len(page_images)} page(s) via vlm model "
        f"'{model}' (dpi={settings.ocr_dpi}, workers={workers})"
    )

    page_markdowns: list[str] = [""] * len(page_images)
    client = settings.build_vision_http_client()
    try:
        def _transcribe(item: tuple[int, bytes]) -> tuple[int, str]:
            index, img = item
            md = chat_vision(
                settings,
                image_bytes=img,
                prompt=VLM_OCR_PROMPT,
                model=model,
                mime="image/png",
                http_client=client,
            )
            return index, md

        items = list(enumerate(page_images))
        if workers == 1:
            for item in items:
                index, md = _transcribe(item)
                page_markdowns[index] = md
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                for index, md in executor.map(_transcribe, items):
                    page_markdowns[index] = md
    finally:
        client.close()

    markdown = join_pages(page_markdowns, preserve_pages=preserve_pages)
    return OcrResult(markdown=markdown, images={}, page_markdowns=page_markdowns)
