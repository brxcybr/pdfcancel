"""Tests for the long-run / fail-safe layer added to pdfcancel.

Covers atomic writes (fsio), the per-page checkpoint cache (pagecache), durable
run state + cooperative signal stop (runstate), and the resumable / fail-safe
behavior of the vlm engine (llm.vlm_ocr_pdf) plus retry-config wiring.
"""

from __future__ import annotations

import json
import os
import signal
from pathlib import Path
from threading import Event

import pytest

from pdfcancel import llm
from pdfcancel.config import Settings
from pdfcancel.errors import ConversionIncomplete
from pdfcancel.fsio import write_json_atomic, write_text_atomic
from pdfcancel.pagecache import CACHE_DIRNAME, PageCache
from pdfcancel.runstate import RUN_STATE_NAME, RunState, stop_on_signals


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in (
        "PDFCANCEL_OCR_ENGINE", "PDFCANCEL_API_BASE", "PDFCANCEL_API_KEY",
        "PDFCANCEL_VLM_MODEL", "PDFCANCEL_OCR_DPI", "PDFCANCEL_OCR_CONCURRENCY",
        "PDFCANCEL_OCR_RETRIES", "PDFCANCEL_OCR_BACKOFF", "PDFCANCEL_OCR_TIMEOUT",
    ):
        monkeypatch.delenv(var, raising=False)
    yield


# ── fsio: atomic writes ──────────────────────────────────────────────────────

def test_write_text_atomic_creates_and_overwrites(tmp_path):
    target = tmp_path / "sub" / "out.md"
    write_text_atomic(target, "first")
    assert target.read_text() == "first"
    write_text_atomic(target, "second")
    assert target.read_text() == "second"
    # No leftover temp files in the directory.
    assert [p.name for p in target.parent.iterdir()] == ["out.md"]


def test_write_json_atomic_roundtrip(tmp_path):
    target = tmp_path / "data.json"
    write_json_atomic(target, {"b": 2, "a": 1})
    assert json.loads(target.read_text()) == {"b": 2, "a": 1}


def test_write_text_atomic_preserves_original_on_failure(tmp_path, monkeypatch):
    target = tmp_path / "keep.md"
    target.write_text("original")

    def _boom(src, dst):
        raise OSError("replace failed")

    monkeypatch.setattr(os, "replace", _boom)
    with pytest.raises(OSError):
        write_text_atomic(target, "new content")

    assert target.read_text() == "original"  # untouched
    # The temp file was cleaned up, leaving only the original.
    assert [p.name for p in tmp_path.iterdir()] == ["keep.md"]


# ── pagecache ────────────────────────────────────────────────────────────────

def _meta(**over):
    base = {"pdf_sha256": "abc", "dpi": 200, "model": "m", "prompt_sha": "p", "page_count": 2}
    base.update(over)
    return base


def test_pagecache_save_and_load(tmp_path):
    cache = PageCache(tmp_path, stem="doc", meta=_meta())
    cache.validate_or_reset()
    assert cache.load_page(0) is None
    cache.save_page(0, "page one")
    assert cache.load_page(0) == "page one"


def test_pagecache_resets_on_meta_change(tmp_path):
    first = PageCache(tmp_path, stem="doc", meta=_meta(model="old"))
    first.validate_or_reset()
    first.save_page(0, "stale")

    # A different model fingerprint must invalidate the cached page.
    second = PageCache(tmp_path, stem="doc", meta=_meta(model="new"))
    second.validate_or_reset()
    assert second.load_page(0) is None


def test_pagecache_keeps_on_same_meta(tmp_path):
    first = PageCache(tmp_path, stem="doc", meta=_meta())
    first.validate_or_reset()
    first.save_page(1, "keep me")

    second = PageCache(tmp_path, stem="doc", meta=_meta())
    second.validate_or_reset()
    assert second.load_page(1) == "keep me"


def test_pagecache_cleanup(tmp_path):
    cache = PageCache(tmp_path, stem="doc", meta=_meta())
    cache.validate_or_reset()
    cache.save_page(0, "x")
    assert cache.dir.exists()
    cache.cleanup()
    assert not cache.dir.exists()


# ── runstate ─────────────────────────────────────────────────────────────────

def test_runstate_counts_and_persists(tmp_path):
    path = tmp_path / RUN_STATE_NAME
    state = RunState(path, total=4)
    state.record("a.pdf", "done")
    state.record("b.pdf", "skipped")
    state.record("c.pdf", "incomplete")
    state.record("d.pdf", "failed")
    state.finish()

    data = json.loads(path.read_text())
    assert data["total"] == 4
    assert data["done"] == 1
    assert data["skipped"] == 1
    assert data["incomplete"] == 1
    assert data["failed"] == 1
    assert data["failures"] == ["d.pdf"]
    assert data["finished"] is True
    assert data["current"] is None


def test_stop_on_signals_sets_event_not_raises():
    with stop_on_signals() as stop:
        assert not stop.is_set()
        signal.raise_signal(signal.SIGINT)  # handled cooperatively, no KeyboardInterrupt
        assert stop.is_set()


# ── vlm_ocr_pdf: resume / failure / interrupt ────────────────────────────────

def _make_pdf(tmp_path: Path) -> Path:
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake content for hashing")
    return pdf


class _NoopClient:
    def close(self):
        pass


def _vlm_settings(**over):
    kw = dict(api_base="https://h/v1", ocr_engine="vlm", vlm_model="m", ocr_concurrency=1)
    kw.update(over)
    return Settings(**kw)


def _expected_meta(pdf: Path, settings: Settings, page_count: int) -> dict:
    return {
        "pdf_sha256": llm._file_sha256(pdf),
        "dpi": int(settings.ocr_dpi),
        "model": settings.vlm_model,
        "prompt_sha": llm.VLM_PROMPT_SHA,
        "page_count": page_count,
    }


def test_vlm_resume_skips_cached_pages(tmp_path, monkeypatch):
    pdf = _make_pdf(tmp_path)
    settings = _vlm_settings()
    monkeypatch.setattr(llm, "rasterize_pdf", lambda p, dpi=200: [b"img0", b"img1"])
    monkeypatch.setattr(settings, "build_vision_http_client", lambda: _NoopClient())

    # Pre-seed page 0 in a cache that matches what vlm_ocr_pdf will compute.
    cache = PageCache(tmp_path, stem=pdf.stem, meta=_expected_meta(pdf, settings, 2))
    cache.validate_or_reset()
    cache.save_page(0, "CACHED0")

    calls: list[int] = []

    def _fake_chat(s, *, image_bytes, prompt, model, max_tokens=None, mime="image/png", http_client=None):
        calls.append(image_bytes)
        return f"FRESH[{image_bytes.decode()}]"

    monkeypatch.setattr(llm, "chat_vision", _fake_chat)

    result = llm.vlm_ocr_pdf(pdf, settings, cache_dir=tmp_path)

    assert result.page_markdowns == ["CACHED0", "FRESH[img1]"]
    assert calls == [b"img1"]  # page 0 came from cache; only page 1 transcribed
    # Cache is cleaned up after a fully successful assembly.
    assert not (tmp_path / CACHE_DIRNAME / pdf.stem).exists()


def test_vlm_one_bad_page_is_incomplete_and_keeps_cache(tmp_path, monkeypatch):
    pdf = _make_pdf(tmp_path)
    settings = _vlm_settings()
    monkeypatch.setattr(llm, "rasterize_pdf", lambda p, dpi=200: [b"img0", b"img1"])
    monkeypatch.setattr(settings, "build_vision_http_client", lambda: _NoopClient())

    def _chat_fail_page2(s, *, image_bytes, prompt, model, max_tokens=None, mime="image/png", http_client=None):
        if image_bytes == b"img1":
            raise RuntimeError("boom")
        return f"OK[{image_bytes.decode()}]"

    monkeypatch.setattr(llm, "chat_vision", _chat_fail_page2)

    with pytest.raises(ConversionIncomplete) as exc:
        llm.vlm_ocr_pdf(pdf, settings, cache_dir=tmp_path)

    assert exc.value.interrupted is False
    assert exc.value.failed_pages == [2]
    assert exc.value.completed == 1
    # The good page is cached; the bad one is not — so a re-run retries only it.
    cache_dir = tmp_path / CACHE_DIRNAME / pdf.stem
    assert (cache_dir / "page-00001.md").exists()
    assert not (cache_dir / "page-00002.md").exists()

    # Second run: page 2 now succeeds, resumes from the cached page 1.
    calls: list[bytes] = []

    def _chat_ok(s, *, image_bytes, prompt, model, max_tokens=None, mime="image/png", http_client=None):
        calls.append(image_bytes)
        return f"OK[{image_bytes.decode()}]"

    monkeypatch.setattr(llm, "chat_vision", _chat_ok)
    result = llm.vlm_ocr_pdf(pdf, settings, cache_dir=tmp_path)
    assert result.page_markdowns == ["OK[img0]", "OK[img1]"]
    assert calls == [b"img1"]  # only the previously-failed page re-run


def test_vlm_interrupted_before_start_is_incomplete(tmp_path, monkeypatch):
    pdf = _make_pdf(tmp_path)
    settings = _vlm_settings()
    monkeypatch.setattr(llm, "rasterize_pdf", lambda p, dpi=200: [b"img0", b"img1"])

    called = []
    monkeypatch.setattr(
        llm, "chat_vision",
        lambda *a, **k: called.append(1) or "x",
    )

    stop = Event()
    stop.set()  # already interrupted
    with pytest.raises(ConversionIncomplete) as exc:
        llm.vlm_ocr_pdf(pdf, settings, cache_dir=tmp_path, stop=stop)

    assert exc.value.interrupted is True
    assert exc.value.completed == 0
    assert called == []  # no transcription attempted


def test_chat_vision_uses_settings_retry_config(tmp_path, monkeypatch):
    settings = _vlm_settings(ocr_retries=7, ocr_backoff=3.0)
    captured: dict = {}

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

    class _Client:
        def post(self, url, json=None, headers=None):
            return _Resp()

        def close(self):
            pass

    monkeypatch.setattr(settings, "build_vision_http_client", lambda: _Client())

    def _fake_retry(fn, *, description="", attempts=3, base_delay=2.0, **kw):
        captured["attempts"] = attempts
        captured["base_delay"] = base_delay
        return fn()

    monkeypatch.setattr(llm, "with_retry", _fake_retry)

    out = llm.chat_vision(settings, image_bytes=b"z", prompt="p", model="m")
    assert out == "ok"
    assert captured["attempts"] == 7
    assert captured["base_delay"] == 3.0
