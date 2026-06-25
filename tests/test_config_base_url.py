"""Tests for custom Mistral server URL configuration (config.py)."""

from __future__ import annotations

import pytest

from pdfcancel.config import Settings, _normalize_base_url


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("https://host/vllm-mistral/v1", "https://host/vllm-mistral"),
        ("https://host/vllm-mistral/v1/", "https://host/vllm-mistral"),
        ("https://host/vllm-mistral/", "https://host/vllm-mistral"),
        ("https://host/vllm-mistral", "https://host/vllm-mistral"),
        ("  https://host/v1  ", "https://host"),
        ("", ""),
        ("   ", ""),
        (None, ""),
    ],
)
def test_normalize_base_url(raw, expected):
    assert _normalize_base_url(raw) == expected


def test_settings_reads_and_normalizes_env(monkeypatch):
    monkeypatch.setenv("MISTRAL_BASE_URL", "https://host/vllm-mistral/v1")
    assert Settings().mistral_base_url == "https://host/vllm-mistral"


def test_settings_base_url_empty_when_env_unset(monkeypatch):
    monkeypatch.delenv("MISTRAL_BASE_URL", raising=False)
    assert Settings().mistral_base_url == ""


def _patch_fake_client(monkeypatch) -> dict:
    """Replace mistralai.client.Mistral with a kwargs-capturing fake."""
    captured: dict = {}

    class FakeMistral:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import mistralai.client as mc

    monkeypatch.setattr(mc, "Mistral", FakeMistral)
    return captured


def test_build_client_passes_server_url_when_set(monkeypatch):
    captured = _patch_fake_client(monkeypatch)
    settings = Settings(
        mistral_api_key="key", mistral_base_url="https://host/vllm-mistral/v1"
    )
    settings.build_client()
    assert captured["api_key"] == "key"
    assert captured["server_url"] == "https://host/vllm-mistral"


def test_build_client_omits_server_url_when_unset(monkeypatch):
    captured = _patch_fake_client(monkeypatch)
    settings = Settings(mistral_api_key="key", mistral_base_url="")
    settings.build_client()
    assert captured["api_key"] == "key"
    assert "server_url" not in captured


def test_build_client_requires_api_key(monkeypatch):
    _patch_fake_client(monkeypatch)
    settings = Settings(mistral_api_key="", mistral_base_url="")
    with pytest.raises(SystemExit):
        settings.build_client()
