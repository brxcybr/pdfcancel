"""Shared test setup."""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """Keep a developer's .env / shell config out of every test.

    ``pdfcancel.config`` calls ``load_dotenv()`` at import, so a local ``.env``
    (e.g. a ``MISTRAL_CA_BUNDLE`` path) would otherwise leak into Settings()
    defaults. Tests that need a value set it explicitly with monkeypatch.
    """
    for var in list(os.environ):
        if var.startswith(("PDFCANCEL_", "MISTRAL_")):
            monkeypatch.delenv(var, raising=False)
