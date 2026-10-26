"""Tests for the OpenAI-compatible ``vlm`` OCR engine (config.py + llm.py)."""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

from pdfcancel import llm
from pdfcancel.config import Settings


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Isolate tests from any PDFCANCEL_* values in the environment / .env."""
    for var in (
        "PDFCANCEL_OCR_ENGINE",
        "PDFCANCEL_API_BASE",
        "PDFCANCEL_API_KEY",
        "PDFCANCEL_VLM_MODEL",
        "PDFCANCEL_OCR_DPI",
        "PDFCANCEL_OCR_CONCURRENCY",
    ):
        monkeypatch.delenv(var, raising=False)
    yield


# ── Fakes ────────────────────────────────────────────────────────────────────

class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _FakeClient:
    """Captures request args and returns a canned JSON payload."""

    def __init__(self, payload: dict, captured: dict):
        self._payload = payload
        self._captured = captured

    def post(self, url, json=None, headers=None):
        self._captured.update(url=url, json=json, headers=headers)
        return _FakeResponse(self._payload)

    def get(self, url, headers=None):
        self._captured.update(url=url, headers=headers)
        return _FakeResponse(self._payload)

    def close(self):
        self._captured["closed"] = True


# ── Config: engine + provider resolution ─────────────────────────────────────

def test_api_base_env_implies_vlm(monkeypatch):
    monkeypatch.setenv("PDFCANCEL_API_BASE", "https://host/litellm/v1")
    settings = Settings()
    assert settings.ocr_engine == "vlm"
    assert settings.api_base == "https://host/litellm"  # trailing /v1 stripped


def test_explicit_engine_overrides_inference(monkeypatch):
    monkeypatch.setenv("PDFCANCEL_API_BASE", "https://host/litellm/v1")
    monkeypatch.setenv("PDFCANCEL_OCR_ENGINE", "mistral")
    assert Settings().ocr_engine == "mistral"


def test_default_engine_is_mistral():
    settings = Settings()
    assert settings.ocr_engine == "mistral"
    assert settings.api_base == ""


def test_api_key_defaults_to_empty():
    assert Settings().api_key == "EMPTY"


def test_vlm_model_and_tuning_from_env(monkeypatch):
    monkeypatch.setenv("PDFCANCEL_VLM_MODEL", "gemma4")
    monkeypatch.setenv("PDFCANCEL_OCR_DPI", "150")
    monkeypatch.setenv("PDFCANCEL_OCR_CONCURRENCY", "8")
    settings = Settings()
    assert settings.vlm_model == "gemma4"
    assert settings.ocr_dpi == 150
    assert settings.ocr_concurrency == 8


def test_ocr_dpi_garbage_falls_back(monkeypatch):
    monkeypatch.setenv("PDFCANCEL_OCR_DPI", "not-an-int")
    assert Settings().ocr_dpi == 200


def test_invalid_engine_raises():
    with pytest.raises(SystemExit):
        Settings(ocr_engine="bogus")


def test_build_vision_http_client_uses_shared_ca_bundle(monkeypatch, tmp_path):
    import httpx

    ca = tmp_path / "ca.pem"
    ca.write_text("dummy")
    captured: dict = {}

    class FakeHttpxClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(httpx, "Client", FakeHttpxClient)
    Settings(mistral_ca_bundle=str(ca)).build_vision_http_client()
    assert captured.get("verify") == str(ca)
    assert captured.get("follow_redirects") is True


# ── chat_vision: OpenAI-compatible request shape ─────────────────────────────

def test_chat_vision_builds_openai_request(monkeypatch):
    settings = Settings(api_base="https://host/litellm/v1", ocr_engine="vlm")
    captured: dict = {}
    payload = {"choices": [{"message": {"content": "# Hello"}}]}
    monkeypatch.setattr(
        settings, "build_vision_http_client", lambda: _FakeClient(payload, captured)
    )

    out = llm.chat_vision(
        settings, image_bytes=b"abc", prompt="Transcribe", model="gemma"
    )

    assert out == "# Hello"
    assert captured["url"] == "https://host/litellm/v1/chat/completions"
    body = captured["json"]
    assert body["model"] == "gemma"
    content = body["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "Transcribe"}
    expected_uri = "data:image/png;base64," + base64.b64encode(b"abc").decode("ascii")
    assert content[1]["image_url"]["url"] == expected_uri
    assert captured["headers"]["Authorization"] == "Bearer EMPTY"
    assert captured.get("closed") is True


def test_chat_vision_uses_api_key(monkeypatch):
    settings = Settings(api_base="https://h/v1", ocr_engine="vlm", api_key="sekret")
    captured: dict = {}
    monkeypatch.setattr(
        settings,
        "build_vision_http_client",
        lambda: _FakeClient({"choices": [{"message": {"content": "x"}}]}, captured),
    )
    llm.chat_vision(settings, image_bytes=b"z", prompt="p", model="m")
    assert captured["headers"]["Authorization"] == "Bearer sekret"


def test_chat_vision_content_as_parts_list(monkeypatch):
    settings = Settings(api_base="https://h/v1", ocr_engine="vlm")
    payload = {"choices": [{"message": {"content": [{"text": "a"}, {"text": "b"}]}}]}
    monkeypatch.setattr(
        settings, "build_vision_http_client", lambda: _FakeClient(payload, {})
    )
    assert llm.chat_vision(settings, image_bytes=b"z", prompt="p", model="m") == "ab"


def test_chat_vision_requires_api_base():
    settings = Settings(ocr_engine="vlm", api_base="")
    with pytest.raises(SystemExit):
        llm.chat_vision(settings, image_bytes=b"x", prompt="p", model="m")


# ── resolve_vlm_model ────────────────────────────────────────────────────────

def test_resolve_model_returns_configured():
    settings = Settings(api_base="https://h/v1", ocr_engine="vlm", vlm_model="my-model")
    assert llm.resolve_vlm_model(settings) == "my-model"


def test_resolve_model_single_model(monkeypatch):
    settings = Settings(api_base="https://h/v1", ocr_engine="vlm")
    captured: dict = {}
    monkeypatch.setattr(
        settings,
        "build_vision_http_client",
        lambda: _FakeClient({"data": [{"id": "only-model"}]}, captured),
    )
    assert llm.resolve_vlm_model(settings) == "only-model"
    assert captured["url"] == "https://h/v1/models"


def test_resolve_model_multiple_raises(monkeypatch):
    settings = Settings(api_base="https://h/v1", ocr_engine="vlm")
    monkeypatch.setattr(
        settings,
        "build_vision_http_client",
        lambda: _FakeClient({"data": [{"id": "a"}, {"id": "b"}]}, {}),
    )
    with pytest.raises(SystemExit):
        llm.resolve_vlm_model(settings)


# ── vlm_ocr_pdf: rasterize + transcribe + assemble ───────────────────────────

def _stub_ocr(monkeypatch, pages: list[bytes]):
    monkeypatch.setattr(llm, "rasterize_pdf", lambda pdf_path, dpi=200: list(pages))
    monkeypatch.setattr(
        llm,
        "chat_vision",
        lambda s, *, image_bytes, prompt, model, mime="image/png", http_client=None: (
            f"PAGE[{image_bytes.decode()}]"
        ),
    )


def test_vlm_ocr_pdf_assembles_pages(monkeypatch):
    settings = Settings(
        api_base="https://h/v1", ocr_engine="vlm", vlm_model="m", ocr_concurrency=1
    )
    _stub_ocr(monkeypatch, [b"img0", b"img1"])
    monkeypatch.setattr(settings, "build_vision_http_client", lambda: _FakeClient({}, {}))

    result = llm.vlm_ocr_pdf(Path("x.pdf"), settings, preserve_pages=False)

    assert result.page_markdowns == ["PAGE[img0]", "PAGE[img1]"]
    assert result.markdown == "PAGE[img0]\n\nPAGE[img1]"
    assert result.images == {}


def test_vlm_ocr_pdf_preserve_pages_markers(monkeypatch):
    settings = Settings(
        api_base="https://h/v1", ocr_engine="vlm", vlm_model="m", ocr_concurrency=1
    )
    _stub_ocr(monkeypatch, [b"a", b"b"])
    monkeypatch.setattr(settings, "build_vision_http_client", lambda: _FakeClient({}, {}))

    result = llm.vlm_ocr_pdf(Path("x.pdf"), settings, preserve_pages=True)

    assert "<!-- pdfcancel-page: 1 -->" in result.markdown
    assert "<!-- pdfcancel-page: 2 -->" in result.markdown


# ── rasterization (requires PyMuPDF) ─────────────────────────────────────────

def test_rasterize_pdf_smoke(tmp_path):
    pymupdf = pytest.importorskip("pymupdf")
    pdf = tmp_path / "doc.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hello OCR")
    doc.save(str(pdf))
    doc.close()

    pages = llm.rasterize_pdf(pdf, dpi=100)
    assert len(pages) == 1
    assert pages[0][:8] == b"\x89PNG\r\n\x1a\n"


# ── Engine guard in convert_single ───────────────────────────────────────────

def test_convert_single_vlm_rejects_full(tmp_path):
    from pdfcancel.convert import convert_single

    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    settings = Settings(api_base="https://h/v1", ocr_engine="vlm", vlm_model="m")
    with pytest.raises(SystemExit):
        convert_single(pdf, settings, output_dir=tmp_path, full=True)
