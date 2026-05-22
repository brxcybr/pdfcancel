"""Plugin system for pdfcancel.

Plugins extend pdfcancel with additional capabilities like Zotero
integration, custom output formats, or alternative OCR providers.

Plugins are discovered from this directory. Each plugin is a Python
module with a `register(cli)` function that adds Click commands or
options to the CLI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


PLUGIN_DIR = Path(__file__).parent


def discover_plugins() -> list[str]:
    """Return names of available plugin modules in the plugins directory."""
    plugins = []
    for path in sorted(PLUGIN_DIR.glob("*.py")):
        if path.name.startswith("_"):
            continue
        plugins.append(path.stem)
    return plugins


def load_plugin(name: str) -> Any:
    """Import and return a plugin module by name."""
    import importlib
    return importlib.import_module(f"pdfcancel.plugins.{name}")
