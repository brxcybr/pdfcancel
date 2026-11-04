"""Endpoint helper for the pdfcancel agent skill.

Subcommands (run by pdfcancel-skill.sh with the profile's environment):

  check                       list served models (and alias -> backend on LiteLLM)
  resolve NAME                print the model name the endpoint accepts for NAME
  ask PDF PAGE PROMPT [opts]  send one rendered page + prompt, print the answer

Environment: PDFCANCEL_API_BASE, PDFCANCEL_API_KEY, MISTRAL_CA_BUNDLE.

LiteLLM routes by its own alias (``gemma-4-31B-it``) and rejects the backend's
Hugging Face id (``google/gemma-4-31B-it``); a direct vLLM server wants the HF
id. ``resolve`` accepts either form and returns what this endpoint serves.
"""

from __future__ import annotations

import argparse
import base64
import os
import sys

import httpx


def _root() -> str:
    base = os.environ.get("PDFCANCEL_API_BASE", "").rstrip("/")
    if not base:
        sys.exit("PDFCANCEL_API_BASE is not set (see the skill profile)")
    return base.removesuffix("/v1")


def _client() -> httpx.Client:
    ca = os.environ.get("MISTRAL_CA_BUNDLE") or True
    return httpx.Client(
        verify=ca,
        headers={"Authorization": f"Bearer {os.environ.get('PDFCANCEL_API_KEY') or 'EMPTY'}"},
        timeout=httpx.Timeout(300, connect=15),
        follow_redirects=True,
    )


def _served(client: httpx.Client) -> tuple[list[str], dict[str, str]]:
    """Return (model ids from /v1/models, {alias: backend id} from LiteLLM)."""
    r = client.get(f"{_root()}/v1/models")
    r.raise_for_status()
    ids = [m["id"] for m in r.json().get("data", []) if m.get("id")]
    backends: dict[str, str] = {}
    try:  # LiteLLM only; may be admin-only, so failure just means "no mapping"
        info = client.get(f"{_root()}/model/info")
        if info.status_code == 200:
            for m in info.json().get("data", []):
                backend = m.get("litellm_params", {}).get("model", "")
                backends[m["model_name"]] = backend.split("/", 1)[-1] if "/" in backend else backend
    except (httpx.HTTPError, ValueError, KeyError):
        pass
    return ids, backends


def resolve(name: str, ids: list[str], backends: dict[str, str]) -> str | None:
    if name in ids:
        return name
    for alias, backend in backends.items():
        if name == backend:
            return alias
    # Without /model/info, match an HF id to an alias by its last path segment.
    tail = name.rsplit("/", 1)[-1]
    matches = [i for i in ids if i == tail or i.rsplit("/", 1)[-1] == tail]
    return matches[0] if len(matches) == 1 else None


def cmd_check(_: argparse.Namespace) -> None:
    with _client() as c:
        ids, backends = _served(c)
    print(f"endpoint: {_root()}/v1")
    for i in ids:
        print(f"  {i:<28} -> {backends[i]}" if i in backends else f"  {i}")


def cmd_resolve(a: argparse.Namespace) -> None:
    with _client() as c:
        ids, backends = _served(c)
    got = resolve(a.name, ids, backends)
    if not got:
        sys.exit(f"unknown model {a.name!r}; served: {', '.join(ids)}")
    print(got)


def cmd_ask(a: argparse.Namespace) -> None:
    import pymupdf

    doc = pymupdf.open(a.pdf)
    if not 1 <= a.page <= doc.page_count:
        sys.exit(f"page {a.page} out of range (1-{doc.page_count})")
    png = doc[a.page - 1].get_pixmap(dpi=a.dpi).tobytes("png")
    body = {
        "model": a.model or os.environ["PDFCANCEL_VLM_MODEL"],
        "max_tokens": a.max_tokens,
        "temperature": 0,
        "messages": [{"role": "user", "content": [
            {"type": "image_url",
             "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}},
            {"type": "text", "text": a.prompt},
        ]}],
    }
    with _client() as c:
        r = c.post(f"{_root()}/v1/chat/completions", json=body)
    if r.status_code != 200:
        sys.exit(f"HTTP {r.status_code}: {r.text[:500]}")
    print(r.json()["choices"][0]["message"]["content"])


def main() -> None:
    ap = argparse.ArgumentParser(description="pdfcancel skill endpoint helper")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check").set_defaults(fn=cmd_check)
    r = sub.add_parser("resolve")
    r.add_argument("name")
    r.set_defaults(fn=cmd_resolve)
    q = sub.add_parser("ask")
    q.add_argument("pdf")
    q.add_argument("page", type=int)
    q.add_argument("prompt")
    q.add_argument("--dpi", type=int, default=200)
    q.add_argument("--model")
    q.add_argument("--max-tokens", type=int, default=4096)
    q.set_defaults(fn=cmd_ask)
    a = ap.parse_args()
    try:
        a.fn(a)
    except httpx.HTTPError as e:
        sys.exit(f"endpoint error: {e}")


if __name__ == "__main__":
    main()
