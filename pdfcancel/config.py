"""Configuration management for pdfcancel."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


# Load .env from cwd (or parents) on import
load_dotenv()


def _normalize_base_url(url: str | None) -> str:
    """Normalize a custom Mistral server URL for the mistralai SDK.

    The SDK appends ``/v1/...`` to its base URL (and only strips a trailing
    slash), so a configured value ending in ``/v1`` would be doubled into
    ``/v1/v1/...``. We strip surrounding whitespace, trailing slashes, and a
    single trailing ``/v1`` segment. A blank value normalizes to ``""``,
    meaning "use the SDK's default hosted endpoint".
    """
    url = (url or "").strip().rstrip("/")
    if url.endswith("/v1"):
        url = url[: -len("/v1")].rstrip("/")
    return url


@dataclass
class Settings:
    """Runtime settings resolved from env vars, CLI flags, and defaults."""

    mistral_api_key: str = field(default_factory=lambda: os.getenv("MISTRAL_API_KEY", ""))
    ocr_model: str = field(
        default_factory=lambda: os.getenv("PDFCANCEL_OCR_MODEL", "mistral-ocr-latest")
    )
    multimodal_model: str = field(
        default_factory=lambda: os.getenv("PDFCANCEL_MULTIMODAL_MODEL", "pixtral-large-latest")
    )
    mistral_base_url: str = field(
        default_factory=lambda: _normalize_base_url(os.getenv("MISTRAL_BASE_URL", ""))
    )
    output_dir: Path | None = None  # None means "same directory as input"

    def require_api_key(self) -> str:
        """Return the API key or raise with a helpful message."""
        if not self.mistral_api_key:
            raise SystemExit(
                "Error: MISTRAL_API_KEY is required.\n"
                "Set it via environment variable or a .env file.\n"
                "  export MISTRAL_API_KEY='your-key-here'\n"
                "Get a free key at https://console.mistral.ai"
            )
        return self.mistral_api_key

    def build_client(self):
        """Construct a Mistral SDK client, honoring a custom server URL.

        Requires an API key. When ``mistral_base_url`` is set, the client is
        pointed at that server (e.g. a local vLLM instance); otherwise the
        SDK's default hosted endpoint is used.
        """
        from mistralai.client import Mistral

        api_key = self.require_api_key()
        base_url = _normalize_base_url(self.mistral_base_url)
        if base_url:
            return Mistral(api_key=api_key, server_url=base_url)
        return Mistral(api_key=api_key)
