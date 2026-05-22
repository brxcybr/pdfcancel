# pdfcancel

Cancel your PDFs — convert them to clean, structured markdown using [Mistral OCR](https://mistral.ai/news/mistral-ocr/).

Built on [Chonkie](https://docs.chonkie.ai) + [Mistral OCR](https://docs.mistral.ai/capabilities/document/) for high-fidelity document conversion with AI-powered image descriptions, semantic search, and Zotero integration.

## Why?

Tools like `pdftotext` flatten PDFs into unstructured strings, losing all document hierarchy. `pdfcancel` produces **clean markdown** with:

- Heading hierarchy preserved (`#`, `##`, `###`, etc.)
- Tables reconstructed as markdown tables
- Lists, blockquotes, and code blocks maintained
- Mathematical equations in LaTeX format
- AI-generated descriptions of charts, figures, and diagrams
- Classification marking handling (CUI, SECRET, UNCLASSIFIED)
- Automatic removal of page headers, footers, watermarks, and page numbers
- Multilingual support (99%+ accuracy across 25+ languages)

## Install

```bash
git clone <repo-url> && cd pdfcancel

# Create a Python 3.12 venv (required — Python 3.14 not yet supported by deps)
python3.12 -m venv .venv
source .venv/bin/activate

# Install core
pip install .

# Install with indexing/search support
pip install ".[indexing]"
```

> **Note:** After editing source code, run `make install` to reinstall.

## Quick Start

```bash
# Cancel a single PDF
pdfcancel paper.pdf

# Cancel an entire directory
pdfcancel ./papers/

# The works: OCR + AI image descriptions + chunks + search index
pdfcancel paper.pdf --full --chunks --index myproject
```

## Configuration

Set your Mistral API key:

```bash
export MISTRAL_API_KEY="your-key-here"
```

Or create a `.env` file:

```
MISTRAL_API_KEY=your-key-here
```

**Get a free key** at [console.mistral.ai](https://console.mistral.ai) — new accounts get $5 free credit (~5,000 pages).

| Variable | Default | Description |
|----------|---------|-------------|
| `MISTRAL_API_KEY` | *(required)* | Mistral API key for OCR |
| `PDFCANCEL_OCR_MODEL` | `mistral-ocr-latest` | OCR model to use |
| `PDFCANCEL_MULTIMODAL_MODEL` | `pixtral-large-latest` | Vision model for `--full` |

## Usage

### Basic Conversion

```bash
pdfcancel paper.pdf                    # → paper.md
pdfcancel paper.pdf -o ./output/       # → ./output/paper.md
pdfcancel paper.pdf --plaintext        # → paper.txt
pdfcancel ./doctrine/                  # cancel all PDFs recursively
pdfcancel paper.pdf --force            # re-process even if unchanged
```

### Image Handling

```bash
pdfcancel paper.pdf --images           # extract to paper_images/
pdfcancel paper.pdf --embed-images     # inline as base64 data URIs
```

### Multimodal Extraction (`--full`)

Sends each extracted image to a vision model for AI-generated descriptions. Descriptions are context-enriched — the vision model receives surrounding document text to ground its output in the paper's terminology.

```bash
pdfcancel paper.pdf --full
pdfcancel paper.pdf --full --full-model pixtral-large-latest
```

Produces descriptions like:

```markdown
![img-0.jpeg](paper_images/img-0.jpeg)
> **Figure description:** This flowchart depicts the CQA workflow for
> an AI-driven interactive chart transformation system. Components include
> a query input, Content-based QA model, and report generator...
```

Descriptions are cached by image content hash — re-runs skip already-described images.

### Enhance Mode (`--enhance`)

Upgrade an existing markdown file without regenerating from scratch:

```bash
pdfcancel paper.pdf --enhance paper.md             # apply cleanup
pdfcancel paper.pdf --enhance paper.md --full      # + add image descriptions
```

### RAG Chunking (`--chunks`)

Produce structured JSONL alongside markdown for RAG/LLM pipelines:

```bash
pdfcancel paper.pdf --chunks
pdfcancel paper.pdf --chunks --chunk-size 512
pdfcancel paper.pdf --chunks --chunker semantic
```

Chunks include rich metadata:
- Section heading breadcrumb (`Strategy > Risk Management > Information Sharing`)
- Content type classification (`prose`, `figure`, `abstract`, `table`, `references`)
- Document-level metadata (`doc_title`, `doc_author`, `doc_year`, `doc_doi`)
- Figure blocks kept atomic — images and their descriptions are never split

### Search Index (`--index`)

Build per-project search indexes with hybrid text + semantic search:

```bash
# Cancel + add to index
pdfcancel paper.pdf --index cti
pdfcancel ./papers/ --index cti --verbose

# Search
pdfcancel search cti "threat intelligence sharing"
pdfcancel search cti "adversary TTPs" --mode semantic
pdfcancel search cti "risk management" --context -k 5

# List indexes
pdfcancel indexes
```

Search features:
- **Hybrid mode** (default): BM25 full-text + semantic cosine similarity
- **Weighted scoring**: abstracts boosted 1.5×, figures 1.3×, references demoted 0.5×
- **Context expansion** (`--context`): shows neighboring chunks for each result
- Local embeddings via model2vec — no API cost for search

### Zotero Integration

Sync entire Zotero collections to markdown with citation metadata:

```bash
# List groups and collections
pdfcancel zotero list
pdfcancel zotero list --group MI-CTI

# Sync a collection
pdfcancel zotero sync CTI -o ./output/ --group MI-CTI
pdfcancel zotero sync CTI -o ./output/ --group MI-CTI --index cti
```

Each synced file gets YAML frontmatter with Zotero metadata:

```yaml
---
title: "TAXII Version 2.1"
authors: [Jordan, Bret, Varner, Drew]
year: "2021"
url: "https://docs.oasis-open.org/cti/taxii/v2.1/os/taxii-v2.1-os.pdf"
zotero_key: "I9C6QY75"
---
```

Requires Zotero 7+ with: Settings > Advanced > "Allow other applications to communicate with Zotero".

### Classification Markings

Government/military documents with classification banners (CUI, SECRET, etc.) are handled automatically:
- Per-page banners stripped from body
- Single classification notice at document top: `> **Classification:** CUI`
- Portion markings like `(U)` preserved inline
- `Page N of M` page numbers removed

### Post-OCR Cleanup

Automatic cleanup runs on every conversion (disable with `--no-clean`):
- Download watermarks (`Downloaded by University of...`)
- Repeating page headers/footers (journal names, author abbreviations)
- Bare page numbers
- Broken sentences rejoined across page boundaries
- Excessive blank lines collapsed

## Full CLI Reference

```
pdfcancel <path> [options]              # cancel PDFs (default action)
pdfcancel search <index> <query>        # search indexed documents
pdfcancel indexes                       # list all search indexes
pdfcancel zotero list [--group NAME]    # list Zotero groups/collections
pdfcancel zotero sync COLL -o DIR       # sync Zotero collection

Cancel options:
  -o, --output DIR       Output directory (default: same as input)
  --images               Extract images to companion directory
  --embed-images         Embed images as base64 in markdown
  --plaintext            Output plaintext instead of markdown
  --full                 AI-describe charts, figures, and diagrams
  --full-model MODEL     Vision model for --full
  --enhance FILE.md      Enhance an existing markdown file
  --model MODEL          OCR model override
  --chunks               Produce chunked JSONL for RAG/LLM
  --chunk-size N         Max tokens per chunk (default: 1024)
  --chunker TYPE         recursive | semantic | sentence
  --index NAME           Add to a named search index
  --no-clean             Skip post-OCR cleanup
  --force                Re-process even if unchanged
  --verbose              Show detailed progress

Search options:
  -k, --top-k N          Number of results (default: 10)
  --mode TYPE            hybrid | semantic | text
  --context              Include neighboring chunks
```

## Development

```bash
source .venv/bin/activate
make install          # reinstall after code changes
make dev-install      # install with dev deps
make test             # smoke test on ref/ PDFs
make clean            # remove build artifacts
```

### Project Structure

```
pdfcancel/
├── pdfcancel/
│   ├── __init__.py      # Version
│   ├── cli.py           # Click CLI with DefaultGroup routing
│   ├── config.py        # Settings, env loading, API key validation
│   ├── convert.py       # Core OCR pipeline
│   ├── clean.py         # Post-OCR cleanup (5 passes + classification)
│   ├── images.py        # Image extraction + direct Mistral SDK
│   ├── multimodal.py    # Context-enriched vision descriptions
│   ├── enhance.py       # Upgrade existing markdown
│   ├── chunks.py        # Figure-aware chunking + content classification
│   ├── index.py         # SQLite FTS5 + semantic search + weighted scoring
│   └── plugins/
│       ├── __init__.py  # Plugin loader
│       └── zotero.py   # Zotero local API integration
├── ref/                 # Reference PDFs for testing
├── pyproject.toml
├── Makefile
└── README.md
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `chonkie[mistral]` | Mistral OCR + chunking pipeline |
| `click` | CLI framework |
| `rich` | Terminal formatting and progress bars |
| `python-dotenv` | `.env` file loading |
| `model2vec` | Local embeddings for semantic search (optional) |
| `numpy` | Cosine similarity for search (optional) |

## Roadmap

- [x] **Phase 1:** Core conversion — `pdfcancel paper.pdf` → `paper.md`
- [x] **Phase 2:** `--full` — context-enriched multimodal image descriptions
- [x] **Phase 3:** `--enhance` — upgrade existing markdown files
- [x] **Phase 4:** `--chunks` — figure-aware RAG chunking with content classification
- [x] **Phase 5:** `--index` + `search` — hybrid search with weighted scoring and context expansion
- [x] **Phase 6:** `zotero` — Zotero library sync with citation frontmatter
- [x] Classification marking handling (CUI, SECRET, UNCLASSIFIED)
- [x] Post-OCR cleanup (watermarks, headers, page numbers, broken sentences)
- [x] Document-level metadata propagation to all chunks
- [ ] Vega-Lite chart spec extraction from figures
- [ ] Structured data table extraction from chart images

## License

MIT
