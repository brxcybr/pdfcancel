"""Batch processing pipeline for pdfcancel.

Uses the Mistral Batch API to process multiple PDFs in parallel at 50% cost.
Supports both OCR batch jobs (/v1/ocr) and vision description batch jobs
(/v1/chat/completions) for --full mode.

Flow:
  1. Upload each PDF → get file IDs and signed URLs
  2. Submit OCR batch job (inline) → poll until complete
  3. Parse OCR results → markdown + image maps per PDF
  4. (--full) Build vision prompts per image → submit chat batch → poll
  5. Return results for local cleanup, injection, chunking, indexing
"""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from pdfcancel.config import Settings

console = Console()

# Poll interval and timeout for batch jobs
_POLL_INTERVAL = 10  # seconds
_POLL_TIMEOUT = 7200  # 2 hours max


class BatchResult:
    """Result for a single PDF from a batch OCR job."""

    def __init__(
        self,
        pdf_path: Path,
        custom_id: str,
        markdown: str,
        images: dict[str, bytes],
    ):
        self.pdf_path = pdf_path
        self.custom_id = custom_id
        self.markdown = markdown
        self.images = images


def batch_ocr(
    pdf_paths: list[Path],
    settings: Settings,
    *,
    include_images: bool = False,
) -> list[BatchResult]:
    """Submit all PDFs as a single OCR batch job and return results.

    Args:
        pdf_paths: List of PDF files to process.
        settings: Settings with API key and OCR model.
        include_images: If True, request image base64 data (for --full mode).

    Returns:
        List of BatchResult with markdown and optional image maps.
    """
    from mistralai.client import Mistral

    client = Mistral(api_key=settings.require_api_key())
    total = len(pdf_paths)

    # Step 1: Upload PDFs and get signed URLs
    console.print(f"\n[bold]Uploading {total} PDF(s) for batch OCR...[/bold]")
    uploads: list[dict[str, Any]] = []  # [{custom_id, file_id, signed_url, pdf_path}]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Uploading...", total=total)
        for idx, pdf_path in enumerate(pdf_paths):
            custom_id = f"ocr-{idx}"
            try:
                with open(pdf_path, "rb") as f:
                    uploaded = client.files.upload(
                        file={"file_name": pdf_path.name, "content": f.read()},
                        purpose="ocr",
                    )
                signed = client.files.get_signed_url(
                    file_id=uploaded.id, expiry=1
                )
                uploads.append({
                    "custom_id": custom_id,
                    "file_id": uploaded.id,
                    "signed_url": signed.url,
                    "pdf_path": pdf_path,
                })
            except Exception as e:
                console.print(f"  [red]✗[/red] Upload failed for {pdf_path.name}: {e}")
            progress.update(task, advance=1, description=f"Uploading ({idx+1}/{total})")

    if not uploads:
        console.print("[red]No files uploaded successfully.[/red]")
        return []

    # Step 2: Build batch requests (inline batching for <10k)
    console.print(f"[bold]Submitting OCR batch job ({len(uploads)} files)...[/bold]")
    requests = []
    for u in uploads:
        body: dict[str, Any] = {
            "model": settings.ocr_model,
            "document": {
                "type": "document_url",
                "document_url": u["signed_url"],
            },
        }
        if include_images:
            body["include_image_base64"] = True
        requests.append({
            "custom_id": u["custom_id"],
            "body": body,
        })

    job = client.batch.jobs.create(
        requests=requests,
        model=settings.ocr_model,
        endpoint="/v1/ocr",
        metadata={"job_type": "pdfcancel_ocr"},
    )
    console.print(f"  Job ID: [cyan]{job.id}[/cyan]  Status: {job.status}")

    # Step 3: Poll for completion
    job = _poll_batch_job(client, job.id)

    if job.status != "SUCCESS":
        console.print(f"[red]Batch OCR job failed: {job.status}[/red]")
        if job.errors:
            for err in job.errors:
                console.print(f"  [red]{err}[/red]")
        return []

    # Step 4: Retrieve results
    console.print("[bold]Retrieving OCR results...[/bold]")
    results = _retrieve_ocr_results(client, job, uploads, include_images)

    console.print(f"[green]✓ {len(results)} PDF(s) processed via batch OCR.[/green]")
    return results


def _poll_batch_job(client: Any, job_id: str) -> Any:
    """Poll a batch job until it completes or times out."""
    start = time.time()
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Waiting for batch job...", total=None)
        while True:
            job = client.batch.jobs.get(job_id=job_id)
            elapsed = int(time.time() - start)
            desc = (
                f"Batch: {job.status} "
                f"({job.succeeded_requests}/{job.total_requests} done) "
                f"[{elapsed}s]"
            )
            progress.update(task, description=desc)

            if job.status in ("SUCCESS", "FAILED", "TIMEOUT_EXCEEDED", "CANCELLED"):
                return job

            if time.time() - start > _POLL_TIMEOUT:
                console.print("[yellow]Batch job timed out, cancelling...[/yellow]")
                client.batch.jobs.cancel(job_id=job_id)
                return client.batch.jobs.get(job_id=job_id)

            time.sleep(_POLL_INTERVAL)


def _retrieve_ocr_results(
    client: Any,
    job: Any,
    uploads: list[dict[str, Any]],
    include_images: bool,
) -> list[BatchResult]:
    """Parse OCR batch job outputs into BatchResult objects."""
    # Build custom_id → upload mapping
    id_to_upload = {u["custom_id"]: u for u in uploads}

    results = []

    # Get results inline
    completed_job = client.batch.jobs.get(job_id=job.id, inline=True)

    if not completed_job.outputs:
        console.print("[yellow]No outputs in batch job response.[/yellow]")
        return results

    for output in completed_job.outputs:
        custom_id = output.custom_id
        upload = id_to_upload.get(custom_id)
        if not upload:
            continue

        response = output.response
        if not response or response.status_code != 200:
            err = getattr(response, "body", "unknown error") if response else "no response"
            console.print(
                f"  [red]✗[/red] {upload['pdf_path'].name}: {err}"
            )
            continue

        # Parse OCR response body
        body = response.body
        markdown_parts = []
        images: dict[str, bytes] = {}

        pages = body.get("pages", []) if isinstance(body, dict) else getattr(body, "pages", [])
        for page in pages:
            md = page.get("markdown", "") if isinstance(page, dict) else getattr(page, "markdown", "")
            markdown_parts.append(md)

            if include_images:
                page_images = page.get("images", []) if isinstance(page, dict) else getattr(page, "images", [])
                for img in page_images:
                    img_id = img.get("id", "") if isinstance(img, dict) else getattr(img, "id", "")
                    img_b64 = img.get("image_base64", "") if isinstance(img, dict) else getattr(img, "image_base64", "")
                    if img_id and img_b64:
                        if "," in img_b64:
                            img_b64 = img_b64.split(",", 1)[1]
                        try:
                            images[img_id] = base64.b64decode(img_b64)
                        except Exception:
                            pass

        markdown = "\n\n".join(markdown_parts)
        results.append(BatchResult(
            pdf_path=upload["pdf_path"],
            custom_id=custom_id,
            markdown=markdown,
            images=images,
        ))

    return results


def batch_vision(
    image_requests: list[dict[str, Any]],
    settings: Settings,
) -> dict[str, str]:
    """Submit image description prompts as a batch chat/completions job.

    Args:
        image_requests: List of dicts with keys:
            - custom_id: unique ID like "pdf_stem::img_id"
            - prompt: the text prompt
            - data_uri: the base64 data URI for the image

    Returns:
        Mapping of custom_id → description text.
    """
    if not image_requests:
        return {}

    from mistralai.client import Mistral
    client = Mistral(api_key=settings.require_api_key())

    total = len(image_requests)
    console.print(f"\n[bold]Submitting batch vision job ({total} image(s))...[/bold]")

    requests = []
    for req in image_requests:
        requests.append({
            "custom_id": req["custom_id"],
            "body": {
                "model": settings.multimodal_model,
                "max_tokens": 500,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": req["prompt"]},
                            {"type": "image_url", "image_url": {"url": req["data_uri"]}},
                        ],
                    }
                ],
            },
        })

    job = client.batch.jobs.create(
        requests=requests,
        model=settings.multimodal_model,
        endpoint="/v1/chat/completions",
        metadata={"job_type": "pdfcancel_vision"},
    )
    console.print(f"  Job ID: [cyan]{job.id}[/cyan]  Status: {job.status}")

    # Poll for completion
    job = _poll_batch_job(client, job.id)

    if job.status != "SUCCESS":
        console.print(f"[red]Batch vision job failed: {job.status}[/red]")
        return {}

    # Retrieve results
    completed = client.batch.jobs.get(job_id=job.id, inline=True)
    descriptions: dict[str, str] = {}

    if not completed.outputs:
        return descriptions

    for output in completed.outputs:
        response = output.response
        if not response or response.status_code != 200:
            descriptions[output.custom_id] = "[Image description unavailable]"
            continue

        body = response.body
        try:
            choices = body.get("choices", []) if isinstance(body, dict) else getattr(body, "choices", [])
            if choices:
                choice = choices[0]
                message = choice.get("message", {}) if isinstance(choice, dict) else getattr(choice, "message", None)
                content = message.get("content", "") if isinstance(message, dict) else getattr(message, "content", "")
                descriptions[output.custom_id] = content.strip()
            else:
                descriptions[output.custom_id] = "[Image description unavailable]"
        except Exception:
            descriptions[output.custom_id] = "[Image description unavailable]"

    console.print(f"[green]✓ {len(descriptions)} image description(s) retrieved.[/green]")
    return descriptions
