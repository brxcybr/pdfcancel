"""Tests for the agent skill's model-name resolution (skills/pdfcancel/scripts)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "skills/pdfcancel/scripts/pdfcancel_skill.py"
_spec = importlib.util.spec_from_file_location("pdfcancel_skill", _PATH)
skill = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(skill)

IDS = ["gemma-4-31B-it", "Mistral-Medium-3.5-128B"]
BACKENDS = {"gemma-4-31B-it": "google/gemma-4-31B-it",
            "Mistral-Medium-3.5-128B": "mistralai/Mistral-Medium-3.5-128B"}


@pytest.mark.parametrize("name,backends,want", [
    ("gemma-4-31B-it", BACKENDS, "gemma-4-31B-it"),           # LiteLLM alias as-is
    ("google/gemma-4-31B-it", BACKENDS, "gemma-4-31B-it"),    # HF id -> alias
    ("google/gemma-4-31B-it", {}, "gemma-4-31B-it"),          # /model/info unavailable
    ("nope", BACKENDS, None),
])
def test_resolve_litellm(name, backends, want):
    assert skill.resolve(name, IDS, backends) == want


def test_resolve_direct_vllm_keeps_hf_id():
    assert skill.resolve("google/gemma-4-31B-it", ["google/gemma-4-31B-it"], {}) == \
        "google/gemma-4-31B-it"


def test_resolve_ambiguous_tail_is_rejected():
    ids = ["a/model-x", "b/model-x"]
    assert skill.resolve("c/model-x", ids, {}) is None
