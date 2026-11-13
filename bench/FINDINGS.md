# pdfcancel vs. docling: findings

*November 13, 2026. Full details and method: `REPORT.md`.*

## What was tested

- **pdfcancel** (`main` + PR #9): a 31B vision model (gemma-4-31B-it) transcribes every page, and a 128B model (Mistral-Medium-3.5) describes figures. It runs on the WIRE vLLM servers.
- **docling** (docling-serve 1.38.1): reads the PDF's text layer and runs layout and table models. The "enriched" mode adds small models for figure descriptions, chart data, and picture classification.

Two test sets, all public, non-CUI documents:

1. **Benchmark:** 8 documents, 137 pages. Published CTI papers, an infographic-heavy industry report, the OASIS TAXII 2.1 spec, and one scanned paper. Hand-checked truth for section headings and chart values.
2. **Academic library:** the 44 public papers from the CTI reference library, 1,369 pages, as converted in the final full library run.

Text is scored against each PDF's own embedded text: **recall** (how much of it was captured), **precision** (how much of the output is in it), and **order** (whether it's in reading order).

## Results

| | pdfcancel | docling |
|---|---|---|
| Text, academic papers (recall / precision / order) | **0.974** / 0.944 / **0.954** | 0.955 / **0.983** / 0.950 |
| Text, benchmark | **0.959** / 0.977 / **0.921** | 0.933 / 0.978 / 0.899 |
| Text on infographic pages (recall) | **0.97** | 0.84 |
| Scanned paper (recall / precision / order) | 0.95 / 0.93 / **0.92** | **0.97 / 0.98** / 0.78 |
| Section headings found (of 160) | **159**¹ | 152 |
| Heading levels correct | **100%** | 61% |
| Chart values correct (of 44) | **43** | 41 |
| Tables / cells kept (academic papers) | **394 / 15,140** | 252 / 8,579 |
| Figure descriptions | **detailed, with values** | one sentence |
| Speed | 4.4 s/page | **1.3 to 1.9 s/page** |

¹ With PR #9's heading fix applied to the benchmark output; 148 without it.

## What this means

**pdfcancel is better for a figure-heavy library meant for search and question answering.**

- It captures more of each document. On some papers docling missed up to two-thirds of the text, mostly scanned pages and text drawn inside charts and infographics. docling's best lead on any paper was under 1%.
- Its figure descriptions are useful. A typical one names the chart, its categories, and its values ("1-5 years 22.4%, 5-10 years 23.1%..."). In a spot check, 44 of 46 quoted percentages appear on the page, and the other 2 are marked as estimates. docling's say "In this image we can see a pie chart."
- Heading levels follow the document's numbering, so chunks keep their place in the document. docling flattens most headings to one level.
- It keeps far more table structure.
- It's consistent. Two separate conversions of the academic papers, four days apart, scored within 0.02 of each other on every paper, with no repetition loops in either.

**docling is better when speed and exact text matter more than figures.**

- It's 2 to 3 times faster, because it reads the text layer instead of running a large model on every page.
- Its text is more precise. pdfcancel's lower precision comes mostly from text inside figures, which pdfcancel transcribes on purpose and the scoring counts as extra.
- It gives bounding boxes for every element, classifies pictures, and reads Word, PowerPoint, and HTML as well as PDF.
- Its enriched mode wasn't reliable on the shared GPU: 3 of 8 benchmark documents failed twice each (out of memory, or internal errors).

## Issues found and fixed (PR #9)

1. **Same-named PDFs overwrote each other.** Outputs go into one folder named by file name, so two different PDFs with the same name in different folders collided, and one of the library's 184 documents was lost. Each now gets its own output, and the lost document has been re-converted.
2. **Some numbered section headings were missed** when the model ran a title straight into its paragraph. Headings found on the benchmark went from 148 back to 159 of 160.
3. **Runs on Ask Sage models.** Two new settings let pdfcancel use OpenAI reasoning models (such as GPT-5.6 Terra Gov) when the local servers are down.

## Limits of this evaluation

- Only public documents were scored. Library documents outside the academic set weren't reviewed.
- One run per tool on the benchmark, on shared servers, so timings are approximate. Speed for the full library run isn't included.
- Scoring against the text layer favors text-layer readers like docling on born-digital pages.
- The tools use different models (31B and 128B against 2B and 4B), so the results compare models as well as pipelines.
