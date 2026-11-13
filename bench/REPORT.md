# pdfcancel vs. docling: feature and performance comparison

*2026-11-04, updated 2026-11-05 and 2026-11-13. Corpus: 8 non-CUI documents, 137 pages: published CTI papers, a public report, the OASIS TAXII 2.1 spec, and one synthetic scan. The 2026-11-13 update adds the CTI library's 44 public academic papers (1,369 pages) as a second, larger test set.*

## Final evaluation (2026-11-13)

pdfcancel as used for the CTI library conversion (`main` 3e68c5b with PRs #7 and #8, plus `fix/vlm-max-completion-tokens`; gemma-4-31B-it pages, Mistral-Medium-3.5-128B figures, WIRE vLLM) against docling-serve 1.38.1 / docling 2.142.1 (unchanged since 2026-11-05).

### Bench (8 documents, 137 pages)

| Category | pdfcancel (`out/pdfcancel-v6`) | docling std (`out/docling-std-v2`) | Leader |
|---|---|---|---|
| Born-digital recall / precision / order (129 pp.) | **0.959** / 0.977 / **0.921** | 0.933 / 0.978 / 0.899 | pdfcancel (precision tied) |
| Scan recall / precision / order (8 pp.) | 0.948 / 0.927 / **0.923** | **0.968 / 0.983** / 0.779 | split, as before |
| Infographic report recall (Chi) | **0.969** | 0.838 | pdfcancel |
| Numbered section headings found (160) | 148 (0.925) | **152 (0.950)** | docling, this run (pdfcancel v4: 159) |
| Heading hierarchy | **141 / 141 = 1.00** | 88 / 145 = 0.61 | pdfcancel |
| Chart values correct (Cascavilla, 44) | **43 / 44** | 41 / 44 (3-page enriched probe) | pdfcancel |
| Tables / cells | **63 / 3,342** | 52 / 1,909 | pdfcancel |
| Throughput (wall) | 4.38 s/page | **1.93 s/page** | docling |

pdfcancel v6 matches v4 on every text metric. Two differences:

- **Headings 159 → 148.** In Alaeifar the model ran nine section titles (3.5–3.13) straight into their paragraphs this time, and `headings.normalize_headings` only promotes a numbered line with a blank line on both sides. The same pattern appears 6 times in 3 of the library's academic outputs (2 of them inside a table of contents, which a fix must leave alone). Model variance plus a strict rule; v4's run happened to get it right.
- **Charts 44 → 43.** One radar value (Cascavilla p. 9) fell outside its ±6 tolerance.

**docling enriched** (picture description, chart extraction, classification) now runs on the GPU, but only **5 of 8 documents converted**: Cascavilla hit CUDA OOM twice, and Alaeifar and the scan returned "Internal processing error" twice. On the 5 documents both converted (80 pp.):

| | pdfcancel | docling std | docling enriched |
|---|---|---|---|
| Recall / precision / order | **0.950** / 0.983 / **0.896** | 0.912 / **0.994** / 0.876 | 0.935 / 0.947* / 0.880 |
| s/page | 3.12 | **1.59** | 5.33 |
| Figures kept / described | 23 / 23 | 46 / 0 | 46 / 24 |
| Words per description | **111** | – | 25 |
| Charts with data | 13 | – | 8 |

\* docling writes its generated descriptions into the markdown, which lowers its text precision; use the standard run for text quality.

Of docling's 46 pictures, its own classifier calls 18 icons and 6 logos. Its descriptions are one sentence ("In this image we can see a pie chart."). pdfcancel's name the chart, its categories, and its values: of 46 percentages quoted in its Chi descriptions, 44 are printed on the same page and the other 2 are marked estimates (`~45–52%`) read off unlabeled bars.

### CTI library, academic papers (44 papers, 1,369 pages)

Scored the same way: each PDF's own text layer is the truth (`scripts/academic_eval.py`). docling std converted all 44. The pdfcancel column is the final library run (2026-11-09 to 11-13, scores in `out/academic_metrics-final.json`), including the Möller 2020 paper re-converted after the name-collision fix (0.99 / 0.98 / 0.99). An earlier conversion of 43 of the papers (2026-11-05, `out/academic_metrics-nov5.json`) scored within 0.02 of the final run on every paper and metric.

| | pdfcancel (final library run) | docling std |
|---|---|---|
| Recall / precision / order, whole documents | **0.965** / 0.929 / **0.948** | 0.953 / **0.971** / 0.944 |
| Same, text-layer pages only (41 docs, 1,087 pp.) | **0.974** / 0.944 / **0.954** | 0.955 / **0.983** / 0.950 |
| Tables / cells | **394 / 15,140** | 252 / 8,579 |
| Figures | 484, all described, 175 with chart data | 611 detected, none described (std) |
| Throughput | not measured (see below) | 1.34 s/page |

- **Recall**: pdfcancel's biggest leads are the Handbook's cover and contents (0.96 vs 0.33), Chi (0.96 vs 0.82), the Handbook introduction (0.99 vs 0.91), and Schechter (0.92 vs 0.86). docling's biggest lead on any paper is under 0.01.
- **Precision** (breakdown from the 2026-11-05 conversion): 72% of the gap is one book (Bautista 2018, 301 pp.), then Chi (10%) and Schechter (9%). The extra words sit on figure pages: text drawn inside diagrams, which has no text layer and which pdfcancel transcribes by design, plus 218 `<br>` tags in table cells counted as words.
- **Against the previous pdfcancel version** (22 papers with an archived output): 0.930 / 0.934 / 0.929 → **0.948 / 0.959 / 0.941**.
- **Defects** (both conversions): no repetition loops, no description lines leaked into body text, every page marked. 4 empty pages: three "intentionally left blank" pages and one TAXII page whose only content was a footer and a continued example URL. LaTeX spans (1,686 in all 44 papers; 1,470 in 43 on 2026-11-05) are footnote superscripts (484), circle markers in Chi (423 `$\bigcirc$`), and real math, not the old spacing misuse.
- **Scanned pages**: 44 pages have no text layer (Handbook chapters). pdfcancel transcribed them; they are left out of the text-layer-only scores.

### Findings

1. **Output name collisions lose documents.** Outputs go to one flat folder named by file stem. Two different PDFs in the library share a stem (one in `academic/`), so one of them has no conversion; a second shared stem is an identical copy. **Fixed in PR #9**: batches append the folder name to the second, different PDF of a name. Re-running the library with the fix converted just those two documents; all 184 PDFs now have their own output (183 names, the identical copy sharing one).
2. **Numbered headings run into their paragraph aren't recovered** (see Bench). **Fixed in PR #9**: a multi-level number (`3.5.`) followed directly by a paragraph now counts, outside contents pages. Applied to this run's output: 148 → 159 of 160; on the 43 academic outputs, +4 real headings and no false ones.
3. **docling enriched isn't dependable on the shared GPU**: 3 of 8 bench documents failed twice each.
4. **The final library came from a separate full run.** It was written to local disk. The 23 documents converted on Ask Sage `gpt-5.6-terra-gov` during the WIRE outage went to an earlier output folder, since superseded and removed; the Ask Sage path was checked only by an academic smoke test (body text 0.90 word similarity to the WIRE output, all headings, figures described).
5. **Library throughput isn't measured.** The library run's log was out of scope for this evaluation, so speed comes from the bench: pdfcancel 4.4 s/page against docling's 1.3–1.9.

### Bottom line

On text, pdfcancel recovers more of each document (recall +0.01 to +0.02 on the academic pages, +0.13 on infographic pages) in better reading order and with far more table structure, while docling stays more precise (+0.035 on text-layer pages), mostly because pdfcancel also transcribes text inside figures. pdfcancel's figure descriptions and chart values are much more useful than docling's, and its heading levels follow the numbering where docling flattens them. docling is 2–3× faster and gives element bounding boxes and picture classification. For a figure-heavy CTI library meant for RAG, pdfcancel is the better tool; docling is the better fit when speed and exact text-layer fidelity matter more than figures.

### Reproduce (2026-11-13)

pdfcancel: `python -I scripts/run_pdfcancel_v6.py` with the vlm engine configured (`PDFCANCEL_OCR_ENGINE=vlm`, `PDFCANCEL_API_BASE`, models) and `MISTRAL_API_KEY` blanked so nothing falls back to the hosted Mistral API. docling: `DOCLING_URL=http://<host>:5001 scripts/run_docling_v2.py <docling-std|docling-full> <input dir> <out name>`. Scores: `scripts/metrics.py`, `scripts/bench_summary.py pdfcancel-v6 docling-std-v2 docling-full-v2`, `scripts/chart_score.py pdfcancel-v6 docling-enriched`, `CTI_REFERENCES=<library references/> scripts/academic_eval.py`. The corpus PDFs, their extracted text, and `out/` are kept out of git (copyrighted papers); `scripts/prepare.py` rebuilds the corpus from `BENCH_SOURCE_PDFS`.

## Standing on 2026-11-05 (after PR #6)

pdfcancel `feat/close-docling-gaps` (run `out/pdfcancel-v4`, default settings: concurrency 8) against docling-serve standard, same corpus and scoring. The sections below *Setup* are the original study and the PR #5 update, kept for history.

| Category | pdfcancel | docling | Leader |
|---|---|---|---|
| Born-digital recall / precision / order (129 pp.) | **0.96** / 0.98 / **0.92** | 0.93 / 0.98 / 0.90 | pdfcancel (precision tied) |
| Scan recall / precision / order (8 pp.) | 0.95 / 0.92 / **0.92** | **0.97 / 0.98** / 0.78 | split: docling on words, pdfcancel on order |
| Infographic report recall (Chi) | **0.97** | 0.84 | pdfcancel |
| Numbered section headings found (160, hand-checked) | **0.99** | 0.95 | pdfcancel |
| Heading hierarchy (levels follow the numbering) | **1.00** | 0.60 (all `##`) | pdfcancel |
| Chart values correct (7 Cascavilla charts, 44 values) | **44 / 44** | 41 / 44 (enriched probe) | pdfcancel |
| Figure descriptions | grounded in the paper's context | generic, sometimes invented | pdfcancel |
| Tables / cells | 63 / 3,326 | 52 / 1,909 | pdfcancel (more structure kept) |
| Throughput (all 137 pp., wall) | 4.8 s/page | **2.6 s/page** | docling |
| Figure detection on scans | whole-page render + description | **per-figure crops** | docling |
| Picture classification, bounding-box provenance, input formats | chart type only, page-level citations | **classifier, element bboxes, DOCX/PPTX/HTML** | docling |

What changed since PR #5:

- **Charts (12 → 44 of 44).** Two causes. (1) The crop missed the y-axis of the two line charts: their tick labels are drawn as 3 pt glyph outlines, which drawing clustering discards, so the model was reading values off unlabeled gridlines. Crops now take in small glyph paths near a figure. (2) A data-only pass with the OCR model (gemma-4-31B) at 300 DPI, prompted to list legend swatches before matching pie slices, replaces the description model's chart JSON; the description is written from its values. On the same crops, gemma read values more accurately than the Mistral describer, and 300 DPI fixed the shade-only pie that 200 DPI swapped (stable across repeated runs). docling's misses: one unlabeled pie slice and the radar chart (CrimeTypes 50 for 60, DetectionMethods 40 for ~32).
- **Headings (0.78 → 0.99 recall, 0.80 → 1.00 hierarchy).** A cleanup pass sets levels from section numbering and recovers numbered headings the model wrote as plain lines (Mavroeidis went from 2 headings to 14). Truth is `truth/headings.json`: numbered section headings from each text layer, filtered for strict section order and checked by hand. Hierarchy counts consecutive found headings whose level change matches their numbering-depth change.
- **Scan (0.87 / 0.92 / 0.83 → 0.95 / 0.92 / 0.92).** The reference page that looped on every attempt is now transcribed column by column after its retry loops; all 39 references come through. Replies are streamed and stopped the moment they loop, so the scan's OCR took 57 s instead of 246 s. Remaining precision gap: most extra words are the labels inside Fig. 1, an embedded image whose text the born-digital truth doesn't contain (pdfcancel transcribes figure text by design), plus a few misreads (Gregio → "Grego").
- **Speed (8.2 → 4.8 s/page).** Concurrency 4 → 8 and the loop cut-off. Throughput scales nearly linearly on this vLLM backend (20-page spec: 47 s at 4, 26 s at 8, 18 s at 16 workers), so `PDFCANCEL_OCR_CONCURRENCY=16` would roughly halve OCR time again; docling stays faster because it reads the text layer instead of running a 31B VLM on every page.
- **Figures.** TAXII's two API request callouts are no longer cropped as figures (a monospace line too long to be a diagram label marks a code box).

Still open: figure detection on scans (a VLM bounding-box probe placed one of two figures well and the other ~10% of the page too high, not reliable enough to crop from), and figure classification beyond chart type.

## Setup

| | pdfcancel | docling |
|---|---|---|
| Version | `main` + PR #4 (vlm engine, local figures, `--full`) | docling-serve 1.38.1 / docling 2.142.1 |
| Where | Local Mac, calling the WIRE LiteLLM proxy | Shared 80 GB GPU server |
| OCR / text | `gemma-4-31B-it` transcribes every rendered page | Standard pipeline: PDF text layer (docling-parse) + layout model + TableFormer; OCR only for bitmaps |
| Figures | PyMuPDF detection → crops → `Mistral-Medium-3.5-128B` description + chart JSON | Layout model; enrichment: Granite Vision 3.3-2B description, Granite Vision 4.1-4B chart extraction, EfficientNet classifier |
| Config compared | `convert` (full pipeline, hierarchical chunks) | **standard** (no enrichment) on all docs; **enriched** on 3 chart pages only (see caveats) |

The docling server doesn't allow custom model configs, so each tool runs with its own models, as deployed.

**Ground truth.** Text is scored against each PDF's embedded text layer. For the synthetic scan (Mavroeidis 2017 rendered to grayscale JPEG pages, no text layer), the original text layer is exact truth. Charts were checked by eye against the rendered crops. Both tools were scored on their markdown, which they emit in reading order; generated descriptions and chart data are excluded from the text scores.

## Results

### Text fidelity, structure, speed

| Metric (page-weighted) | pdfcancel | docling (std) |
|---|---|---|
| Recall, born-digital | 0.92 | **0.93** |
| Precision, born-digital | 0.79 ⚠ | **0.98** |
| Reading order, born-digital | 0.84 | **0.90** |
| Recall / precision / order, **scan** | 0.93 / 0.91 / **0.90** | **0.97 / 0.98** / 0.78 |
| Recall on infographic report (Chi) | **0.95** | 0.84 |
| Throughput | 8.2 s/page (OCR 5.4 + figures 2.8) | **2.6 s/page** |
| Headings found | 181 (levels inconsistent) | **299** (all flattened to `##`) |
| Tables / cells | 67 / 3,259 | 52 / 1,909 |
| Figures (detected / described / with chart data) | 55 / 55 / 31 | 86 / — / — (std) |

⚠ pdfcancel's precision is held down by three bugs found here (repetition loops, LaTeX misuse, description leak). See *Improvements*, P0.

### Figures and charts (Cascavilla pp. 7–9, both enriched)

| Chart | Truth | docling | pdfcancel |
|---|---|---|---|
| Venue bars | 7 / 15 / 10 | **7 / 15 / 10** | 7 / 15 / 9 |
| Study-type pie (printed %) | 33 / 20 / 20 / 7 / 20 | **correct** (one label dropped) | Case and Empirical **swapped**, marked `confidence=exact` |
| Grey-lit pie (no %s) | Experience-Report largest | 50 / 25 / 25, order right | 50 / 33 / 7, **wrong**, wrong figure title |
| Org bars | 8 / 10 / 10 / 13 | ✓ | ✓ |
| Radar | CrimeTypes 60, Detection ~38 | approximate (50 / 40) | no data |
| **Descriptions** | — | Generic, often **invented topic**: "number of people diagnosed with cancer", "percentage of people employed" | **Grounded** in the paper: correct subject, axes, trend, and relation to the text |
| Classification | — | **Accurate** (bar 0.996, pie 0.999, line 0.995) | none |

## Strengths and weaknesses

**docling**
- ➕ Fast, and **exact text on born-digital PDFs**, because it reads the text layer instead of re-OCRing.
- ➕ Strong heading detection; element-level page and bounding-box provenance; accurate picture classification.
- ➕ A **dedicated chart-extraction model** reads values more accurately.
- ➕ Many input formats (DOCX/PPTX/XLSX/HTML/audio); mature chunkers and framework integrations.
- ➖ **Drops text inside pictures.** Recall falls to 0.84 on the infographic report.
- ➖ Descriptions from small local VLMs are generic or hallucinated, with no document context.
- ➖ Flattens heading levels by default. Collapsed a complex evaluation-matrix table (Mavroeidis Table I) into one garbled row.
- ➖ GPU-hungry: the enriched pipeline **OOMed on the shared GPU** for every document.
- ➖ Reading order on the scan is weaker (0.78 vs 0.90).

**pdfcancel**
- ➕ **Context-grounded figure descriptions** plus structured chart data, useful for RAG on figure-heavy material.
- ➕ Captures text inside charts and infographics; better reading order on scans; kept the complex matrix table's structure.
- ➕ Page-cited markdown and chunks; runs on any OpenAI-compatible endpoint; packaged as an agent skill.
- ➖ ~3× slower, because every page goes through a 31B VLM.
- ➖ **Transcription errors where the text layer is exact**: "Sharing Standards, Ontologies and Ontologies" for "…, and Ontologies"; **NXD/CVE for NVD/CWE**.
- ➖ Chart values are less accurate, with overconfident `exact` labels.
- ➖ Headings missed, with levels inconsistent across pages (pages are transcribed independently).
- ➖ No figure detection on scans: detection relies on PDF vector/image objects.
- ➖ Bugs found here: VLM **repetition loops** (one Chi page repeated a phrase ~620×), **LaTeX for spacing** (`\quad` ×1,910 on Cascavilla), and **multi-line descriptions escaping their blockquote** into body text.

## Improvements for pdfcancel (prioritized)

**P0: bugs found by this study (small, high impact)**
1. **Repetition guard.** Detect looping n-grams in a page transcription. Retry with a frequency or repetition penalty, then truncate the loop if it recurs.
2. **Blockquote every description line** in `inject_descriptions`, so multi-line descriptions and lists stay out of the body text and chunks.
3. **LaTeX discipline.** Restrict the OCR prompt to real math, and add a cleaner that strips `\quad`/`\text{ }` spacing runs and unwraps `$\text{word}$`.

**P1: biggest accuracy and speed gains**
4. **Use the text layer on born-digital pages.** Give the VLM the page's extracted text as the authoritative spelling ("use these exact strings"), or skip the VLM for text-only pages. This fixes misread identifiers (NVD/CWE), cuts hallucinations, and recovers speed. docling's advantage comes almost entirely from this.
5. **Ground chart extraction.** Pass the text-layer strings found inside each crop (printed values and labels) to the describer. Use a separate data-only prompt at higher DPI. Only claim `confidence=exact` when every value matches a printed label.

**P2: structure**
6. **Consistent heading hierarchy.** A post-pass that normalizes levels from numbering (I./A./1.1) and the PDF outline, plus the previous page's heading context in the prompt.
7. **Figure detection on scans.** A small layout model (docling's DocLayNet-trained layout model is open source) or VLM-returned bounding boxes.
8. **Figure classification for routing.** Charts go to data extraction, diagrams to description, logos and icons are skipped. docling's EfficientNet figure classifier is open source.

**P3: complementary use**
9. **An optional `docling` engine**: docling-serve for fast, faithful text, tables and layout, with pdfcancel's grounded figure descriptions and chunking on top. That would combine the best of both measured here.

## Update: after the P0/P1 fixes (PR #5)

Same corpus and models, run into `out/pdfcancel-v3`:

| Born-digital, 129 pages | before | after | docling (std) |
|---|---|---|---|
| Recall / precision / order | 0.92 / 0.79 / 0.84 | **0.96 / 0.98 / 0.92** | 0.93 / 0.98 / 0.90 |
| s/page | 8.4 | 6.7 | **2.4** |
| LaTeX spans | 3,071 | 119 | – |
| Description lines leaked into body | 209 | 0 | – |

- Born-digital text now matches docling on precision and beats it on recall and order; docling remains ~3× faster.
- Scan: pages 1–7 are unchanged; page 8 (dense references) looped on every try in this run and continuation recovered only part of it (scan 0.87 / 0.92 / 0.83 vs 0.93 / 0.91 / 0.90). The page loops 2 of 3 times under the old prompt too. Still open: splitting a persistently looping page into regions.
- Chart grounding rarely changes anything here: most crops (e.g. all of Cascavilla's) draw their labels as vector outlines, with no text layer to check against. Only 2 `exact` claims were lowered. A separate data-only pass at higher DPI is the remaining lever for chart accuracy.

## Caveats
- One run per tool, on shared servers. Timings are indicative, not precise.
- The docling enriched pipeline failed with **CUDA OOM** on every document (another tenant held ~49 GB of the 80 GB GPU), so enrichment was judged from a successful 3-page probe. Re-run `scripts/run_docling.py docling-full` when the GPU is free.
- Text-layer ground truth favors text-layer readers on born-digital PDFs; the scan is the neutral OCR test.
- Different models on each side (31B / 128B LLMs vs 2B / 4B specialist models), so quality differences reflect models as well as pipelines.

## Reproduce
`scripts/prepare.py` (corpus + truth) → `scripts/run_pdfcancel.py` (original; `_v3`, `_v4` for later runs), `scripts/run_docling.py docling-std [docling-full]` → `scripts/metrics.py`. Heading truth: `scripts/heading_truth.py` → `truth/headings.json`. Chart values: `scripts/chart_score.py pdfcancel-v4 docling-enriched`. Outputs are in `out/` (`pdfcancel-v4-pre` is a run of an intermediate build, before the crop fix).
