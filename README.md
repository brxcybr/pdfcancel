# pdfcancel

Cancel your PDFs — convert them to clean, structured markdown using [Mistral OCR](https://mistral.ai/news/mistral-ocr/).

Built on [Chonkie](https://docs.chonkie.ai) + [Mistral OCR](https://docs.mistral.ai/capabilities/document/) for high-fidelity document conversion that preserves headings, tables, lists, equations, and figure references.

## Why?

Tools like `pdftotext` flatten PDFs into unstructured strings, losing all document hierarchy. `pdfcancel` produces **clean markdown** with:

- Heading hierarchy preserved (`#`, `##`, `###`, etc.)
- Tables reconstructed as markdown tables
- Lists, blockquotes, and code blocks maintained
- Mathematical equations in LaTeX format
- Figure/image references retained
- Multilingual support (99%+ accuracy across 25+ languages)

## Install

```bash
# Clone the repo
git clone <repo-url> && cd pdfcancel

# Create a Python 3.12 venv (required — Python 3.14 not yet supported by deps)
python3.12 -m venv .venv
source .venv/bin/activate

# Install
pip install .
```

> **Note:** After editing source code, run `make install` to reinstall.

## Quick Start

```bash
# Cancel a single PDF
pdfcancel paper.pdf

# Cancel an entire directory (recursive)
pdfcancel ./papers/

# Output to a specific directory
pdfcancel paper.pdf -o ./output/
```

## Configuration

### API Key (required)

pdfcancel uses the Mistral OCR API. Set your key via environment variable:

```bash
export MISTRAL_API_KEY="your-key-here"
```

Or create a `.env` file in your working directory:

```
MISTRAL_API_KEY=your-key-here
```

**Get a free key** at [console.mistral.ai](https://console.mistral.ai) — new accounts get $5 free credit (~5,000 pages).

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MISTRAL_API_KEY` | *(required)* | Mistral API key for OCR |
| `PDFCANCEL_OCR_MODEL` | `mistral-ocr-latest` | OCR model to use |
| `PDFCANCEL_MULTIMODAL_MODEL` | `pixtral-large-latest` | Vision model for `--full` |

## Usage

### Basic Conversion

```bash
# PDF → markdown (default)
pdfcancel paper.pdf                    # → paper.md (same directory)
pdfcancel paper.pdf -o ./output/       # → ./output/paper.md

# PDF → plaintext
pdfcancel paper.pdf --plaintext        # → paper.txt

# Entire directory (recursive *.pdf)
pdfcancel ./doctrine/                  # → one .md per PDF
pdfcancel ./doctrine/ -o ./converted/  # all output in ./converted/
```

### Image Handling

```bash
# Extract images as separate files
pdfcancel paper.pdf --images
# → paper.md + paper_images/img-0.jpeg, img-1.png, ...

# Embed images as base64 data URIs (self-contained markdown)
pdfcancel paper.pdf --embed-images
# → paper.md (larger file, no companion directory)
```

### Multimodal Extraction (future)

```bash
# AI-describe all charts, figures, and diagrams
pdfcancel paper.pdf --full

# Use a specific vision model
pdfcancel paper.pdf --full --full-model pixtral-large-latest
```

> `--full` is not yet implemented. When available, it will send each extracted
> image to a vision model and insert text descriptions in the markdown.

### Enhance Mode (future)

```bash
# Upgrade an existing markdown file with multimodal descriptions
pdfcancel paper.pdf --enhance existing_paper.md
```

> `--enhance` is not yet implemented.

### Model Override

```bash
# Use a specific OCR model
pdfcancel paper.pdf --model mistral-ocr-2505
```

### Skip Processing

pdfcancel tracks processed files by SHA-256 hash. On re-run, unchanged PDFs are skipped automatically. Use `--force` to override:

```bash
pdfcancel paper.pdf --force    # re-process even if unchanged
```

## Full CLI Reference

```
Usage: pdfcancel [OPTIONS] PATH

  Cancel your PDFs — convert them to clean markdown.

Options:
  -o, --output PATH     Output directory (default: same as input).
  --images              Extract images to companion directory.
  --embed-images        Embed images as base64 in markdown.
  --plaintext           Output plaintext instead of markdown.
  --full                Full multimodal extraction (future).
  --full-model TEXT     Vision model for --full.
  --enhance PATH        Enhance existing markdown (future).
  --model TEXT          OCR model override.
  --force               Re-process even if already converted.
  --verbose             Show detailed progress.
  --version             Show version and exit.
  --help                Show this message and exit.
```

## Development

```bash
# Activate venv
source .venv/bin/activate

# Install with dev deps
make dev-install

# Reinstall after code changes
make install

# Run smoke test
make test

# Clean build artifacts
make clean
```

### Project Structure

```
pdfcancel/
├── pdfcancel/           # Main package
│   ├── __init__.py      # Version
│   ├── cli.py           # Click CLI entry point
│   ├── config.py        # Settings, env loading, API key validation
│   ├── convert.py       # PDF → markdown/plaintext conversion
│   └── images.py        # Image extraction and embedding
├── ref/                 # Reference PDFs for testing
├── pyproject.toml       # Package metadata and dependencies
├── Makefile             # Dev helpers (install, test, clean)
└── README.md
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `chonkie[mistral]` | Mistral OCR integration + pipeline framework |
| `click` | CLI framework |
| `rich` | Terminal formatting and progress bars |
| `python-dotenv` | `.env` file loading |

## Roadmap

- [x] **Phase 1:** Core conversion — `pdfcancel paper.pdf` → `paper.md`
- [ ] **Phase 2:** `--full` — multimodal image descriptions via vision LLM
- [ ] **Phase 3:** `--enhance` — upgrade existing markdown files
- [ ] **Phase 4:** `--chunks` — RAG/LLM chunked output (JSONL)
- [ ] **Phase 5:** `--index` — per-project searchable index database
- [ ] **Phase 6:** `--zotero` — Zotero library integration with citation metadata

## License

MIT
