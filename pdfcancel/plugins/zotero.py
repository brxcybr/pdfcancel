"""Zotero plugin for pdfcancel.

Connects to a running Zotero instance via its local HTTP API (port 23119)
to enumerate PDFs and bibliographic metadata. Supports both personal
libraries and group libraries.

Usage:
    pdfcancel zotero list                          # list groups + collections
    pdfcancel zotero sync CTI -o ./output/         # cancel all PDFs in 'CTI'
    pdfcancel zotero sync CTI -o ./out/ --index cti  # cancel + index
    pdfcancel zotero sync CTI -o ./out/ --group 6503832  # explicit group ID

Requires Zotero 7+ running with:
  Settings > Advanced > "Allow other applications on this computer to
  communicate with Zotero"
"""

from __future__ import annotations

import json as _json
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

console = Console()

ZOTERO_LOCAL = "http://localhost:23119/api"
ZOTERO_STORAGE = Path.home() / "Zotero" / "storage"


def _api(endpoint: str) -> Any:
    """GET a Zotero local API endpoint and return parsed JSON."""
    url = f"{ZOTERO_LOCAL}{endpoint}"
    try:
        with urllib.request.urlopen(url) as resp:
            return _json.loads(resp.read())
    except Exception as e:
        raise SystemExit(
            f"Error: Could not connect to Zotero at {url}\n"
            f"  {e}\n"
            "  Ensure Zotero is running with:\n"
            "  Settings > Advanced > Allow other applications to communicate with Zotero"
        )


@dataclass
class ZoteroItem:
    """A bibliographic item from Zotero with metadata and PDF paths."""
    key: str
    item_type: str
    title: str = ""
    date: str = ""
    doi: str = ""
    abstract: str = ""
    url: str = ""
    publication: str = ""
    creators: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    pdf_paths: list[Path] = field(default_factory=list)
    collections: list[str] = field(default_factory=list)

    @property
    def first_author(self) -> str:
        if self.creators:
            return self.creators[0].split(",")[0].strip()
        return "Unknown"

    @property
    def year(self) -> str:
        if self.date:
            for part in self.date.replace("/", "-").split("-"):
                if len(part) == 4 and part.isdigit():
                    return part
        return ""

    def frontmatter(self) -> str:
        """Generate YAML frontmatter for markdown output."""
        lines = ["---"]
        lines.append(f'title: "{self.title}"')
        if self.creators:
            lines.append(f"authors: [{', '.join(self.creators)}]")
        if self.year:
            lines.append(f'year: "{self.year}"')
        if self.date:
            lines.append(f'date: "{self.date}"')
        if self.doi:
            lines.append(f'doi: "{self.doi}"')
        if self.publication:
            lines.append(f'publication: "{self.publication}"')
        if self.url:
            lines.append(f'url: "{self.url}"')
        if self.tags:
            lines.append(f"tags: [{', '.join(self.tags)}]")
        if self.collections:
            lines.append(f"collections: [{', '.join(self.collections)}]")
        lines.append(f'zotero_key: "{self.key}"')
        lines.append("---\n")
        return "\n".join(lines)


def list_groups() -> list[dict[str, Any]]:
    """List all Zotero groups the user belongs to."""
    groups = _api("/users/0/groups")
    return [
        {
            "id": g["data"]["id"],
            "name": g["data"]["name"],
            "items": g["meta"]["numItems"],
        }
        for g in groups
    ]


def list_collections(group_id: int) -> list[dict[str, Any]]:
    """List all collections in a group."""
    colls = _api(f"/groups/{group_id}/collections")
    return [
        {
            "key": c["key"],
            "name": c["data"]["name"],
            "parent": c["data"].get("parentCollection", ""),
            "items": c["meta"].get("numItems", 0),
        }
        for c in colls
    ]


def find_group(name_or_id: str) -> int:
    """Find a group by name or ID."""
    if name_or_id.isdigit():
        return int(name_or_id)
    groups = list_groups()
    for g in groups:
        if g["name"].lower() == name_or_id.lower():
            return g["id"]
    raise SystemExit(
        f"Error: Group '{name_or_id}' not found.\n"
        "  Run: pdfcancel zotero list"
    )


def find_collection_key(group_id: int, name: str) -> str:
    """Find a collection key by name within a group."""
    colls = list_collections(group_id)
    for c in colls:
        if c["name"].lower() == name.lower():
            return c["key"]
    raise SystemExit(
        f"Error: Collection '{name}' not found in group {group_id}.\n"
        f"  Available: {', '.join(c['name'] for c in colls)}"
    )


def get_collection_items(
    group_id: int,
    collection_key: str,
) -> list[ZoteroItem]:
    """Get all bibliographic items with PDFs from a collection via the API."""
    # Fetch items (excluding attachments/notes/annotations), paginated
    items_data = []
    start = 0
    while True:
        batch = _api(
            f"/groups/{group_id}/collections/{collection_key}/items"
            f"?itemType=-attachment+-note+-annotation&limit=100&start={start}"
        )
        if not batch:
            break
        items_data.extend(batch)
        if len(batch) < 100:
            break
        start += 100

    # Build ZoteroItems and find their PDFs
    items = []
    for raw in items_data:
        item = _build_item_from_api(raw, group_id)
        if item.pdf_paths:
            items.append(item)

    return items


def _build_item_from_api(raw: dict, group_id: int) -> ZoteroItem:
    """Build a ZoteroItem from a Zotero API response dict."""
    d = raw["data"]
    item = ZoteroItem(
        key=d["key"],
        item_type=d.get("itemType", ""),
        title=d.get("title", ""),
        date=d.get("date", ""),
        doi=d.get("DOI", ""),
        abstract=d.get("abstractNote", ""),
        url=d.get("url", ""),
        publication=d.get("publicationTitle", ""),
    )

    # Creators
    for c in d.get("creators", []):
        last = c.get("lastName", "")
        first = c.get("firstName", "")
        if last and first:
            item.creators.append(f"{last}, {first}")
        elif last:
            item.creators.append(last)
        elif c.get("name"):
            item.creators.append(c["name"])

    # Tags
    item.tags = [t["tag"] for t in d.get("tags", [])]

    # Collections (names would require extra lookups; store keys for now)
    item.collections = d.get("collections", [])

    # Find PDF attachments via children endpoint
    children = _api(f"/groups/{group_id}/items/{d['key']}/children")
    for child in children:
        cd = child["data"]
        if cd.get("contentType") == "application/pdf":
            att_key = cd["key"]
            filename = cd.get("filename", "")
            if filename:
                pdf_path = ZOTERO_STORAGE / att_key / filename
                if pdf_path.exists():
                    item.pdf_paths.append(pdf_path)

    return item


def register(cli: Any) -> None:
    """Register the Zotero subcommands with the pdfcancel CLI."""
    import click
    from pdfcancel.config import Settings
    from pdfcancel.convert import convert_single

    @cli.group()
    def zotero():
        """Zotero library integration — sync collections to markdown."""
        pass

    @zotero.command(name="list")
    @click.option("--group", "group_name", default=None,
                  help="Show collections for a specific group.")
    def zotero_list(group_name):
        """List Zotero groups and their collections."""
        groups = list_groups()
        if not groups:
            console.print("[yellow]No groups found.[/yellow]")
            return

        if group_name:
            gid = find_group(group_name)
            colls = list_collections(gid)
            table = Table(title=f"Collections in '{group_name}'")
            table.add_column("Key", style="dim")
            table.add_column("Name", style="cyan")
            table.add_column("Items", justify="right")
            for c in colls:
                table.add_row(c["key"], c["name"], str(c["items"]))
            console.print(table)
        else:
            table = Table(title="Zotero Groups")
            table.add_column("ID", justify="right")
            table.add_column("Name", style="cyan")
            table.add_column("Items", justify="right")
            for g in groups:
                table.add_row(str(g["id"]), g["name"], str(g["items"]))
            console.print(table)
            console.print("\n  [dim]Tip: pdfcancel zotero list --group MI-CTI[/dim]")

    @zotero.command(name="sync")
    @click.argument("collection")
    @click.option("-o", "--output", "output_dir", type=click.Path(), required=True,
                  help="Output directory for markdown files.")
    @click.option("--group", "group_name", default=None,
                  help="Zotero group name or ID (auto-detected if collection name is unique).")
    @click.option("--full", is_flag=True, help="Include multimodal image descriptions.")
    @click.option("--chunks", is_flag=True, help="Also produce chunked JSONL.")
    @click.option("--index", "index_name", default=None,
                  help="Add to a named search index (implies --chunks).")
    @click.option("--force", is_flag=True, help="Re-process even if unchanged.")
    @click.option("--verbose", is_flag=True, help="Show detailed progress.")
    def zotero_sync(collection, output_dir, group_name, full, chunks, index_name, force, verbose):
        """Sync a Zotero collection — cancel all PDFs to markdown with metadata.

        \b
        Examples:
          pdfcancel zotero sync CTI -o ./cti_markdown/ --group MI-CTI
          pdfcancel zotero sync CTI -o ./output/ --index cti --group MI-CTI
        """
        settings = Settings()
        out = Path(output_dir)

        # Find the group
        if group_name:
            gid = find_group(group_name)
        else:
            # Auto-detect: search all groups for a matching collection name
            gid = _auto_find_group(collection)

        coll_key = find_collection_key(gid, collection)

        console.print(f"[bold]Scanning Zotero collection:[/bold] {collection} (group {gid})")
        items = get_collection_items(gid, coll_key)

        if not items:
            console.print("[yellow]No items with PDFs found in this collection.[/yellow]")
            return

        console.print(f"  Found {len(items)} item(s) with PDFs\n")

        if index_name:
            chunks = True

        results = []
        total = len(items)
        for idx, item in enumerate(items, 1):
            pdf_path = item.pdf_paths[0]
            console.print(f"  [dim]({idx}/{total})[/dim] ", end="")

            try:
                md_path = convert_single(
                    pdf_path, settings,
                    output_dir=out, full=full, force=force,
                    extract_images=full,
                    produce_chunks=chunks,
                    chunk_size=1024, chunker_type="recursive",
                )

                # Inject frontmatter
                _inject_frontmatter(md_path, item)

                if verbose:
                    console.print(f"  [green]✓[/green] {item.title[:60]}")

                results.append(md_path)

                if index_name:
                    from pdfcancel.cli import _index_file
                    _index_file(md_path, pdf_path.name, index_name, 1024, "recursive")

            except Exception as e:
                console.print(f"  [red]✗[/red] {item.title[:60]}: {e}")

        console.print(f"\n[green]Synced {len(results)}/{total} item(s) from '{collection}'.[/green]")

    def _auto_find_group(collection_name: str) -> int:
        """Search all groups for a collection matching the given name."""
        groups = list_groups()
        for g in groups:
            colls = list_collections(g["id"])
            for c in colls:
                if c["name"].lower() == collection_name.lower():
                    return g["id"]
        raise SystemExit(
            f"Error: Collection '{collection_name}' not found in any group.\n"
            "  Specify --group explicitly, or run: pdfcancel zotero list"
        )


def _inject_frontmatter(md_path: Path, item: ZoteroItem) -> None:
    """Inject YAML frontmatter with Zotero metadata at the top of a markdown file."""
    content = md_path.read_text()
    if content.startswith("---\n"):
        return

    frontmatter = item.frontmatter()
    md_path.write_text(frontmatter + content)

    # Rename to Author - Year - Title format if current name is unhelpful
    stem = md_path.stem
    if item.title and not any(c.isalpha() for c in stem[:3]):
        new_name = f"{item.first_author} - {item.year} - {item.title[:80]}.md"
        new_name = "".join(c if c.isalnum() or c in " -_.,()'" else "_" for c in new_name)
        new_path = md_path.parent / new_name
        if not new_path.exists():
            md_path.rename(new_path)
