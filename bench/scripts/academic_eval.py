"""Score the CTI library's academic outputs (public papers) the way metrics.py scores the bench.

Truth is each PDF's text layer (as prepare.py builds it). Compared per paper:
the library run's pdfcancel output, the previous pdfcancel version (archive,
where it exists), and docling-serve standard (out/academic-docling-std).
Outputs are opened by exact stem only; no folder outside academic/ is listed.
Writes out/academic_metrics.json and prints page-weighted totals.
"""
import json, os, re, sys
from pathlib import Path

import pymupdf

sys.path.insert(0, os.path.dirname(__file__))
from metrics import _leaked, md_tables, strip_docling, strip_pdfcancel, text_scores, words  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pdfcancel.vlmtext import find_loop  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFS = Path(os.path.expanduser(os.environ["CTI_REFERENCES"]))  # the library's references/ folder
# The library run being scored; ACADEMIC_OUT points at another run's flat output folder.
NEW = Path(os.path.expanduser(os.environ.get("ACADEMIC_OUT", str(REFS / "academic" / "markdown"))))
OLD = REFS / "archive/academic/markdown"
DOCLING = Path(ROOT) / "out/academic-docling-std"
PAGE = re.compile(r"<!--\s*pdfcancel-page:\s*(\d+)\s*-->")
MIN_WORDS = 20  # a page with fewer text-layer words is treated as having no text layer


def pdfcancel_pages(md: str) -> dict[int, str]:
    parts = PAGE.split(md)
    return dict(zip(map(int, parts[1::2]), parts[2::2]))


def docling_pages(md: str, pages: int) -> dict[int, str] | None:
    parts = md.split("<!-- page-break -->")
    return dict(enumerate(parts, 1)) if len(parts) == pages else None


def pdfcancel_extras(md: str, pages: int) -> dict:
    body = strip_pdfcancel(md)
    chunks = PAGE.split(md)  # [pre, n1, text1, n2, text2, ...]
    marked = {int(n) for n in chunks[1::2]}
    loops = sum(find_loop(strip_pdfcancel(t)) for t in chunks[2::2])
    tables, cells = md_tables(body)
    return {
        "pages_marked": len(marked & set(range(1, pages + 1))),
        "empty_pages": sum(not strip_pdfcancel(t).strip() for t in chunks[2::2]),
        "loop_pages": loops,
        "headings": len(re.findall(r"^#{1,6} ", md, re.M)),
        "tables": tables, "table_cells": cells,
        "figures": len(re.findall(r"!\[[^\]]*\]\([^)]*_images/[^)]*\)", md)),
        "described": md.count("> **Figure description:**"),
        "chart_data": md.count("pdfcancel-chart-data"),
        "latex": len(re.findall(r"\$[^$\n]+\$", body)),
        "leaked": _leaked(md),
    }


def main():
    from pdfcancel.convert import file_hash
    manifest = json.loads((NEW / ".pdfcancel_manifest.json").read_text())
    rows = []
    for pdf in sorted((REFS / "academic/pdfs").rglob("*.pdf")):
        stem = pdf.stem
        doc = pymupdf.open(pdf)
        pages = doc.page_count
        layer = [pg.get_text() for pg in doc]
        truth = "\f".join(layer)
        text_pages = [i for i, t in enumerate(layer, 1) if len(words(t)) >= MIN_WORDS]
        tp_truth = "\f".join(layer[i - 1] for i in text_pages)
        row = {"doc": stem, "pages": pages, "text_pages": len(text_pages)}
        new = NEW / f"{stem}.md"
        # The output folder is flat: a same-named PDF elsewhere can own this stem.
        if new.exists() and (manifest.get(stem) or {}).get("hash") != file_hash(pdf):
            row["collision"] = True
        elif new.exists():
            md = new.read_text()
            pp = pdfcancel_pages(md)
            row["new"] = {**text_scores(truth, strip_pdfcancel(md)), **pdfcancel_extras(md, pages)}
            row["new_tp"] = text_scores(tp_truth, strip_pdfcancel("\f".join(pp.get(i, "") for i in text_pages)))
        old = OLD / f"{stem}.md"
        if old.exists():
            md = old.read_text()
            row["old"] = {**text_scores(truth, strip_pdfcancel(md)), **pdfcancel_extras(md, pages)}
        dj = DOCLING / f"{stem}.json"
        if dj.exists():
            meta = json.loads(dj.read_text())
            d = meta.get("doc") or {}
            md = (DOCLING / f"{stem}.md").read_text()
            tables = d.get("tables", [])
            row["docling"] = {
                **text_scores(truth, strip_docling(md)),
                "status": meta.get("status"), "s_per_page": meta["wall"] / pages,
                "headings": sum(1 for t in d.get("texts", []) if t.get("label") in ("section_header", "title")),
                "tables": len(tables),
                "table_cells": sum(len(t.get("data", {}).get("table_cells", [])) for t in tables),
                "figures": len(d.get("pictures", [])),
            }
            dp = docling_pages(md, pages)
            if dp:
                row["docling_tp"] = text_scores(tp_truth, strip_docling("\f".join(dp[i] for i in text_pages)))
        rows.append(row)
        print(f"{stem[:60]:60} {pages:4}p  " + "  ".join(
            f"{k}: {r['recall']:.2f}/{r['precision']:.2f}/{r['order']:.2f}"
            for k, r in ((k, row.get(k)) for k in ("new", "old", "docling")) if r), flush=True)
    json.dump(rows, open(os.environ.get("ACADEMIC_METRICS", f"{ROOT}/out/academic_metrics.json"), "w"), indent=1)

    def weighted(key, rs, metric, w="pages"):
        sel = [r for r in rs if key in r]
        pg = sum(r[w] for r in sel)
        return sum(r[key][metric] * r[w] for r in sel) / pg if pg else float("nan")

    def show(label, key, rs, w="pages"):
        n = sum(key in r for r in rs); pg = sum(r[w] for r in rs if key in r)
        print(f"  {label:52} n={n:2} {w}={pg:5}  " + " / ".join(
            f"{weighted(key, rs, m, w):.3f}" for m in ("recall", "precision", "order")))

    with_old = [r for r in rows if "old" in r and "new" in r]
    both = [r for r in rows if "docling" in r and "new" in r]
    both_tp = [r for r in rows if "docling_tp" in r and "new_tp" in r]
    print("\nPage-weighted recall / precision / order, whole documents")
    show("pdfcancel library run, all", "new", rows)
    show("pdfcancel library run, docs with prior version", "new", with_old)
    show("pdfcancel prior version, same docs", "old", with_old)
    show("pdfcancel library run, docs docling converted", "new", both)
    show("docling std, same docs", "docling", both)
    print("\nText-layer pages only (pages with no text layer have no truth)")
    show("pdfcancel library run", "new_tp", both_tp or rows, "text_pages")
    show("docling std", "docling_tp", both_tp, "text_pages")
    print("\nexcluded, stem owned by a different PDF:", [r["doc"] for r in rows if r.get("collision")])
    print("pages with no text layer:", sum(r["pages"] - r["text_pages"] for r in rows))
    tot = lambda key, m: sum(r[key][m] for r in rows if key in r)
    print("\nlibrary run totals:", {m: tot("new", m) for m in (
        "pages_marked", "empty_pages", "loop_pages", "headings", "tables", "table_cells",
        "figures", "described", "chart_data", "latex", "leaked")},
        "of", sum(r["pages"] for r in rows if "new" in r), "pages")


if __name__ == "__main__":
    main()
