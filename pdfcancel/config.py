"""Configuration management for pdfcancel."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


# Load .env from cwd (or parents) on import
load_dotenv()


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
