---
name: pdfcancel
description: Convert PDFs into clean, page-cited markdown and RAG chunks with pdfcancel on a self-hosted vision model (vLLM, LiteLLM, or Ollama), including local figure extraction and AI descriptions of charts, graphs, tables, diagrams, and infographics with structured chart data. Use when extracting context from PDFs, especially figure-heavy, scanned, dense academic, or sensitive documents that must stay on your network; when the user mentions pdfcancel, --full, figure descriptions, OCR, page-level citations, or converting a reference library; when asking a local vision model about a specific PDF page; or when ordinary PDF text extraction loses tables and figure content.
compatibility: Any agent with a shell. Source at https://github.com/brxcybr/pdfcancel (skills/pdfcancel). Needs bash, pdfcancel installed with the [vlm] extra (install.sh does this), a profile at ~/.config/pdfcancel/skill.env, and network reach to the configured endpoint.
---

# pdfcancel on a self-hosted vision endpoint

pdfcancel turns PDFs into markdown that keeps headings, tables, LaTeX, and
`<!-- pdfcancel-page: N -->` page markers. It writes optional
`_chunks.jsonl` for retrieval. On the `vlm` engine everything runs against
your own OpenAI-compatible server: page OCR, local figure cropping, and figure
descriptions.

**Always go through the wrapper:**

```bash
W=~/.agents/skills/pdfcancel/scripts/pdfcancel-skill.sh
```

The wrapper loads the site profile (`~/.config/pdfcancel/skill.env`, or
`$PDFCANCEL_SKILL_PROFILE`): endpoint, models, token, and CA bundle. It exports
those values so a stray `.env` can't redirect a run, and it translates model
names into the form the endpoint accepts.

## Setup on a new workstation

Source: **https://github.com/brxcybr/pdfcancel** (public). The skill lives in `skills/pdfcancel/` there.

If `$W check` exits `127` (pdfcancel missing) or `78` (no profile), the
workstation isn't set up yet. **Ask the user before installing anything**,
then do one of the following:

- **Team bundle** (`pdfcancel-skill*.tar.gz`, shared by a teammate): it ships
  the profile and CA certificates, so nothing has to be filled in. pdfcancel
  itself is pulled from the repository.
  ```bash
  tar xzf pdfcancel-skill*.tar.gz && ./pdfcancel-skill/install.sh
  ```
- **From the repository:**
  ```bash
  git clone https://github.com/brxcybr/pdfcancel ~/src/pdfcancel
  ~/src/pdfcancel/skills/pdfcancel/install.sh          # creates ~/.config/pdfcancel/skill.env
  ```
  Then fill in `~/.config/pdfcancel/skill.env`: `PDFCANCEL_API_BASE`,
  `PDFCANCEL_VLM_MODEL`, and, for an internal CA, `PDFCANCEL_CA_BUNDLE`. Get
  these values from the user or their team; don't guess endpoints. Re-run
  `install.sh --ca root.crt --ca issuing.crt` to install CA certificates.

`install.sh` creates a virtualenv with the latest `pdfcancel[vlm,chunking]`. It
installs the skill to `~/.agents/skills/pdfcancel`, linked into
`~/.claude/skills`, and finishes with an endpoint check. The source is
`$PDFCANCEL_REPO` at `$PDFCANCEL_REF` (default: the URL above, `main`; either
can be set in the profile).

### Updating

```bash
$W update     # latest pdfcancel + skill files from the repository; keeps profile and CA
```

Suggest an update to the user, rather than working around the problem, when
something this skill documents is missing: pdfcancel rejects an option such as
`--full` on the vlm engine, or a mode listed here doesn't exist. A skill linked
to a developer checkout is updated there instead (`git pull && make install`).

## Data-handling boundary

- Every wrapper mode stays on the configured endpoint. Figure detection and
  cropping run on this machine.
- Don't run bare `pdfcancel` without `--ocr-engine vlm` / `--api-base`. Its
  default engine is Mistral's **hosted cloud API**. Never send sensitive or
  controlled documents there, and ask the user before sending anything to the
  cloud.
- Never disable TLS verification. If TLS fails, fix `PDFCANCEL_CA_BUNDLE` in the
  profile.

## Step 1: check the endpoint

```bash
$W check     # lists served models (alias -> backend on LiteLLM); fails fast on TLS/network
```

If this fails, stop and report the error. Don't fall back to another route.

## Step 2: pick a mode

| Goal | Command |
|---|---|
| **Everything**: text, tables, and every figure cropped and described, with structured chart data | `$W convert <pdf-or-dir> [-o DIR]` |
| Text, tables, and in-figure labels only (faster; no crops or descriptions) | `$W ocr <pdf-or-dir> [-o DIR]` |
| A targeted question about one page ("give Table 3 as CSV") | `$W ask <pdf> <page> "prompt"` |
| Search an index, check status, other pdfcancel commands | `$W <pdfcancel args>` |
| Get the latest pdfcancel and skill | `$W update` |

Default to `convert` when figures, charts, or diagrams carry meaning.

### `convert`: full local extraction

Runs `pdfcancel <path> --full --preserve-pages --chunks --chunker hierarchical`
on the vlm engine. Per PDF:

1. **OCR:** each page is rendered and transcribed to markdown by
   `PDFCANCEL_VLM_MODEL`. That includes tables, LaTeX, captions, and the text
   inside figures.
2. **Figures:** detected locally with PyMuPDF. That covers embedded images, vector
   drawing clusters (most charts), and nearby axis and legend labels. Each
   figure is cropped to `<stem>_images/pNNN-fKK.png` and placed where the OCR
   left its figure placeholder. Running headers, repeated logos, ruled tables,
   and shaded code or text boxes are skipped, because the OCR already
   transcribed them. A scanned page with a figure placeholder gets a
   full-page render instead.
3. **Describe:** each crop goes to `PDFCANCEL_VLM_FULL_MODEL` (default: the OCR
   model) along with the surrounding text. The result is a
   `> **Figure description:**` block, plus a data table and hidden chart JSON
   for charts and visual tables.
4. **Chunk:** hierarchical chunks, with figure blocks kept whole and tied to
   their section.

```bash
$W convert paper.pdf                         # → paper.md, paper_images/, paper_chunks.jsonl
$W convert ./library/ -o ./markdown/ --exclude DRAFT
$W convert paper.pdf --ocr-dpi 300           # small chart text (PDFCANCEL_FIGURE_DPI for crops)
$W convert paper.pdf --force                 # redo a document already converted
$W convert paper.pdf --index mylib           # also add to a search index
```

- **Resumable:** finished PDFs are skipped as unchanged, and pages are
  checkpointed under `.pdfcancel_cache/`. After an interruption, re-run the
  same command.
- **Cost:** OCR takes roughly 15–20 s per page at concurrency 4. Each figure
  description takes roughly 10 s. A 30-page paper with 20 figures takes
  several minutes, so run directories in the background.

### `ocr`: text-only pass

Same as `convert` without `--full`. Figures stay as `![figure](figure)`
placeholders, though their visible text is still transcribed.

### `ask`: one page, one question

```bash
$W ask paper.pdf 12 "Transcribe Table 3 as CSV with its header row. Output only the CSV."
$W ask paper.pdf 7 "List every data point in Figure 2 as series,x,y. Mark estimates with ~." --dpi 300
```

Pages are 1-indexed, matching the page markers. The answer goes to stdout. Ask
for exact transcription and tell the model not to guess values it can't read.
Use this when a figure is missing or ambiguous in the converted output.

## Step 3: read the output

- `<stem>.md`: read one page with
  `awk '/pdfcancel-page: 12 -->/,/pdfcancel-page: 13 -->/' paper.md`. List
  figures with `grep -n -A12 'Figure description' paper.md`. The linked PNG is
  the exact crop, so open it if you can read images.
- `<stem>_chunks.jsonl`: one `{"text","metadata"}` per line. Metadata includes
  `section`, `content_type` (`prose`, `figure`, `table`, `abstract`,
  `references`), `has_figure`, `page_start`, `page_end`, and `doc_title`.

  ```bash
  jq -c 'select(.metadata.content_type=="table" or .metadata.has_figure)
         | {p: .metadata.page_start, s: .metadata.section, text}' paper_chunks.jsonl
  ```
- Search: `$W search mylib "query" --context -k 5`. Results cite pages.
- Cite pages from the markers. Say so when a figure is still a bare
  `![figure](figure)` (nothing was detected), then use `ask` for that page.

## Exit codes

| Code | Meaning | Action |
|---|---|---|
| 0 | Done | Read the outputs |
| 1 | Endpoint or validation error, or an unknown model (served models listed) | Report it; fix the profile or retry with `--force` |
| 2 | Incomplete: interrupted, or pages failed after retries; the cache is kept | Re-run the same command |
| 64 | Wrapper usage error | Fix the arguments |
| 78 | No profile | See [Setup on a new workstation](#setup-on-a-new-workstation) |
| 127 | pdfcancel isn't installed in `PDFCANCEL_VENV` | See [Setup on a new workstation](#setup-on-a-new-workstation) |

## Troubleshooting

- **`400 Invalid model name` (LiteLLM):** LiteLLM routes by its own alias
  (`gemma-4-31B-it`), not the backend's Hugging Face id (`google/gemma-4-31B-it`).
  The wrapper resolves either form. Run `$W check` to see what's served.
- **`404 model does not exist`:** a direct vLLM URL got a name it doesn't host.
  Use the backend id, or point the profile at the proxy.
- **TLS errors:** `PDFCANCEL_CA_BUNDLE` must be a PEM file containing the
  internal root and issuing CAs.
- **Truncated pages:** per-page OCR output is capped at 4096 tokens. Split
  dense pages with `ask`.
- **Timeouts:** raise `--ocr-timeout 300` and lower `--ocr-concurrency 2`.
- **Wrong or noisy crops:** delete `<stem>_images/` and re-run with `--force`.
  Use `ask` for anything detection missed.
