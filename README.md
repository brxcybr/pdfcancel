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
- Automatic removal of page headers, footers, watermarks, and page numbers
- Multilingual support (99%+ accuracy across 25+ languages)

## Install

```bash
git clone https://github.com/brxcybr/pdfcancel.git && cd pdfcancel

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
| `MISTRAL_BASE_URL` | *(unset)* | Custom Mistral-compatible server URL (e.g. local vLLM); a trailing `/v1` is added automatically |

### Self-hosted / local Mistral server

If your organization runs a Mistral-compatible server (e.g. a local [vLLM](https://docs.vllm.ai) instance), point pdfcancel at it to keep large or sensitive documents off the public API:

```bash
export MISTRAL_BASE_URL="https://your-host/vllm-mistral"   # or pass --mistral-url
pdfcancel paper.pdf --enhance paper.md --full \
  --full-model mistralai/Mistral-Medium-3.5-128B \
  --mistral-url https://your-host/vllm-mistral
```

- The URL is the server **root**; the SDK appends `/v1/...`, and a trailing `/v1` you include is stripped automatically (both `.../vllm-mistral` and `.../vllm-mistral/v1` work).
- Set the served model name with `--full-model` (vision) and/or `--model` (OCR); local model IDs differ from the hosted defaults.
- **Capability caveat:** an OpenAI-compatible server such as vLLM serves `--full` image descriptions (chat/vision) but typically does **not** implement Mistral's OCR, file-upload, or batch endpoints. The base PDF→markdown OCR step, `--images` / `--embed-images` / `--preserve-pages`, and `--batch` therefore still require Mistral's hosted API. To run `--full` entirely against a local server, use `--enhance` on documents that already have an extracted `<name>_images/` folder — pdfcancel describes those images in place without re-running OCR.

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
When a chart, graph, matrix, or visual table contains extractable data, `--full`
also asks the vision model for a strict JSON payload. pdfcancel renders that
payload as searchable markdown table data and preserves it as hidden metadata for
downstream chunking and indexing.

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
> **Structured chart data:**
> bar_chart; Model accuracy; confidence=estimated
>
> | Series | X | Y | Label |
> | --- | --- | --- | --- |
> | Accuracy | CNN | ~91.5 | CNN |
```

Descriptions are cached by image content hash — re-runs skip already-described images.

### Enhance Mode (`--enhance`)

Upgrade an existing markdown file without regenerating from scratch:

```bash
pdfcancel paper.pdf --enhance paper.md             # apply cleanup
pdfcancel paper.pdf --enhance paper.md --full      # + add image descriptions
```

When the markdown already has an extracted `<name>_images/` folder, `--enhance --full` describes those images directly — with no OCR call — so it can run entirely against a self-hosted vision server (see [Self-hosted / local Mistral server](#self-hosted--local-mistral-server)).

### Page-Level Citations (`--preserve-pages`)

By default, cleanup strips page numbers and page boundaries are lost. For
academic work that needs page-level citations, `--preserve-pages` injects an
HTML comment marker (`<!-- pdfcancel-page: N -->`, 1-indexed) before each
page's markdown. Markers survive post-OCR cleanup, and the chunker records
`page_start` / `page_end` metadata on every chunk (stripping the comments
from the chunk text). Indexed chunks store these as columns, and search
results show the page citation (`p. 12` / `pp. 12–14`).

```bash
pdfcancel paper.pdf --preserve-pages --index myproject
pdfcancel search myproject "threat intelligence"   # results cite pages
```

Existing indexes keep working — chunks indexed without `--preserve-pages`
simply have no page info.

### OCR Output Validation

After every OCR run, pdfcancel validates the response before writing output:

- The response must contain at least one page.
- Extracted text must average at least ~20 characters per page (catches
  empty/truncated OCR responses).
- If [pypdf](https://pypi.org/project/pypdf/) can read the source PDF, the
  OCR page count is compared to the actual page count and a warning is
  printed on mismatch.
- Images that fail base64 decoding are counted and reported instead of
  silently dropped.

On validation failure the output file is **not** written, the failure is
reported per file, and the CLI exits non-zero if any file failed.

### RAG Chunking (`--chunks`)

Produce structured JSONL alongside markdown for RAG/LLM pipelines:

```bash
pdfcancel paper.pdf --chunks
pdfcancel paper.pdf --chunks --chunk-size 512
pdfcancel paper.pdf --chunks --chunker semantic
pdfcancel paper.pdf --chunks --chunker hierarchical
```

Chunks include rich metadata:
- Section heading breadcrumb (`Strategy > Risk Management > Information Sharing`)
- Hierarchical chunk IDs for section-aware retrieval (`chunk_id`, `parent_id`, sibling IDs)
- Content type classification (`prose`, `figure`, `abstract`, `table`, `references`)
- Document-level metadata (`doc_title`, `doc_author`, `doc_year`, `doc_doi`)
- Figure blocks kept atomic — images and their descriptions are never split
- Structured chart/table metadata from `--full`, including extracted data and
  approximate Vega-Lite specs where a chart can be reconstructed

Chunking strategies:
- `recursive`: fast markdown-aware splitting for general use
- `semantic`: small semantic chunks for high-recall retrieval
- `sentence`: sentence-bounded chunks
- `hierarchical`: semantic chunks bounded by markdown sections, with parent/sibling metadata so a retrieval hit can be expanded to neighboring chunks and same-section figures

For documents where charts, diagrams, or screenshots carry definitional context, combine
`--full` with `--chunker hierarchical`. `--full` injects vision-generated figure
descriptions before chunking, and hierarchical metadata keeps those figure chunks attached
to the surrounding section.

### Search Index (`--index`)

Build per-project search indexes with hybrid text + semantic search:

```bash
# Cancel + add to index
pdfcancel paper.pdf --index collection
pdfcancel ./papers/ --index collection --verbose

# Search
pdfcancel search collection "DevSecOps"
pdfcancel search collection "Agentic AI" --mode semantic
pdfcancel search collection "AI Governance" --context -k 5

# List indexes
pdfcancel indexes
```

Search features:
- **Hybrid mode** (default): BM25 full-text + semantic cosine similarity
- **Weighted scoring**: abstracts boosted 1.5×, figures 1.3×, references demoted 0.5×
- **Context expansion** (`--context`): shows neighboring chunks for each result
- Structured chart metadata available in search result dictionaries
- Local embeddings via model2vec — no API cost for search

### Zotero Integration

Sync entire Zotero collections to markdown with citation metadata:

```bash
# List groups and collections
pdfcancel zotero list
pdfcancel zotero list --group ORG

# Sync a collection
pdfcancel zotero sync COLLECTION -o ./output/ --group ORG
pdfcancel zotero sync COLLECTION -o ./output/ --group ORG --index collection
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
  --mistral-url URL      Custom Mistral-compatible server URL (local vLLM)
  --enhance FILE.md      Enhance an existing markdown file
  --model MODEL          OCR model override
  --chunks               Produce chunked JSONL for RAG/LLM
  --chunk-size N         Max tokens per chunk (default: 1024)
  --chunker TYPE         recursive | semantic | sentence | hierarchical
  --index NAME           Add to a named search index
  --no-clean             Skip post-OCR cleanup
  --preserve-pages       Keep page boundaries; chunks get page_start/page_end
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
│   ├── charts.py        # Structured chart/table data + Vega-Lite helpers
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
| `pypdf` | Page-count validation of OCR output |
| `model2vec` | Local embeddings for semantic search (optional) |
| `numpy` | Cosine similarity for search (optional) |

## Roadmap

- [x] **Phase 1:** Core conversion — `pdfcancel paper.pdf` → `paper.md`
- [x] **Phase 2:** `--full` — context-enriched multimodal image descriptions
- [x] **Phase 3:** `--enhance` — upgrade existing markdown files
- [x] **Phase 4:** `--chunks` — figure-aware RAG chunking with content classification
- [x] **Phase 5:** `--index` + `search` — hybrid search with weighted scoring and context expansion
- [x] **Phase 6:** `zotero` — Zotero library sync with citation frontmatter
- [x] Post-OCR cleanup (watermarks, headers, page numbers, broken sentences)
- [x] Document-level metadata propagation to all chunks
- [x] Vega-Lite chart spec extraction from figures
- [x] Structured data table extraction from chart images

## License

MIT
