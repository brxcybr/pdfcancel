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


def _int_env(name: str, default: int) -> int:
    """Parse an int from an env var, falling back to default on absence/garbage."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    """Parse a float from an env var, falling back to default on absence/garbage."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw.strip())
    except ValueError:
        return default


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
    mistral_ca_bundle: str = field(
        default_factory=lambda: os.getenv("MISTRAL_CA_BUNDLE", "")
    )

    # OCR engine selection + generic OpenAI-compatible (vlm) provider.
    ocr_engine: str = field(
        default_factory=lambda: (
            os.getenv("PDFCANCEL_OCR_ENGINE", "mistral").strip().lower() or "mistral"
        )
    )
    api_base: str = field(
        default_factory=lambda: _normalize_base_url(os.getenv("PDFCANCEL_API_BASE", ""))
    )
    api_key: str = field(
        default_factory=lambda: os.getenv("PDFCANCEL_API_KEY", "EMPTY") or "EMPTY"
    )
    vlm_model: str = field(
        default_factory=lambda: os.getenv("PDFCANCEL_VLM_MODEL", "").strip()
    )
    ocr_dpi: int = field(default_factory=lambda: _int_env("PDFCANCEL_OCR_DPI", 200))
    ocr_concurrency: int = field(
        default_factory=lambda: _int_env("PDFCANCEL_OCR_CONCURRENCY", 4)
    )
    # Network resilience for the vlm engine (long/flaky runs). Retries apply
    # per request (page transcription, model listing); timeout is the per-page
    # read/write budget.
    ocr_retries: int = field(default_factory=lambda: _int_env("PDFCANCEL_OCR_RETRIES", 5))
    ocr_backoff: float = field(
        default_factory=lambda: _float_env("PDFCANCEL_OCR_BACKOFF", 2.0)
    )
    ocr_timeout: float = field(
        default_factory=lambda: _float_env("PDFCANCEL_OCR_TIMEOUT", 180.0)
    )
    output_dir: Path | None = None  # None means "same directory as input"

    def __post_init__(self) -> None:
        # Normalize a directly-assigned api_base, then infer the engine: an
        # api_base configured via env without an explicit engine selects vlm.
        self.api_base = _normalize_base_url(self.api_base)
        if (
            self.api_base
            and self.ocr_engine == "mistral"
            and not os.getenv("PDFCANCEL_OCR_ENGINE")
        ):
            self.ocr_engine = "vlm"
        if self.ocr_engine not in ("mistral", "vlm"):
            raise SystemExit(
                f"Error: invalid OCR engine '{self.ocr_engine}'. "
                "Use 'mistral' or 'vlm'."
            )

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

        When ``mistral_ca_bundle`` is set, TLS certificates are verified
        against that CA bundle (e.g. an internal/self-hosted CA) instead of
        the default trust store, without modifying any global trust store.
        Verification always stays enabled; a missing bundle path is a hard
        error rather than a silent fallback to an unverified connection.
        """
        from mistralai.client import Mistral

        api_key = self.require_api_key()
        kwargs: dict = {"api_key": api_key}

        base_url = _normalize_base_url(self.mistral_base_url)
        if base_url:
            kwargs["server_url"] = base_url

        ca_bundle = (self.mistral_ca_bundle or "").strip()
        if ca_bundle:
            if not Path(ca_bundle).exists():
                raise SystemExit(
                    f"Error: MISTRAL_CA_BUNDLE path does not exist: {ca_bundle}"
                )
            import httpx

            # Verify against the provided CA bundle. This keeps verification
            # ON and is scoped to this client only (no changes to certifi or
            # the OS trust store). Mirror the SDK default follow_redirects.
            kwargs["client"] = httpx.Client(verify=ca_bundle, follow_redirects=True)

        return Mistral(**kwargs)

    def build_vision_http_client(self):
        """Construct an ``httpx.Client`` for OpenAI-compatible (vlm) calls.

        TLS is verified against the shared CA bundle (``mistral_ca_bundle``)
        when set — e.g. an internal/self-hosted root CA — scoped to this
        client only, without touching any global trust store. Verification
        always stays enabled; a missing bundle path is a hard error.
        """
        import httpx

        verify: object = True
        ca_bundle = (self.mistral_ca_bundle or "").strip()
        if ca_bundle:
            if not Path(ca_bundle).exists():
                raise SystemExit(
                    f"Error: CA bundle path does not exist: {ca_bundle}"
                )
            verify = ca_bundle
        # Split timeouts: fail fast on a dead connection, but allow a long read
        # for a slow page transcription. ``read``/``write`` use ocr_timeout.
        timeout = httpx.Timeout(
            connect=15.0,
            read=self.ocr_timeout,
            write=self.ocr_timeout,
            pool=30.0,
        )
        return httpx.Client(verify=verify, follow_redirects=True, timeout=timeout)
