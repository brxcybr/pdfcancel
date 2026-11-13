"""Roll out/metrics.json up into REPORT.md's categories for the named runs.

    bench_summary.py pdfcancel-v6 docling-std-v2 docling-full-v2
"""
import json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRUTH = json.load(open(f"{ROOT}/truth/headings.json"))
rows = json.load(open(f"{ROOT}/out/metrics.json"))


def pw(rs, k):
    pg = sum(r["pages"] for r in rs)
    return sum(r[k] * r["pages"] for r in rs) / pg if pg else float("nan")


for run in sys.argv[1:]:
    rs = [r for r in rows if r["tool"] == run and r.get("status", "success") == "success"]
    bad = [r["doc"] for r in rows if r["tool"] == run and r not in rs]
    born = [r for r in rs if r["doc"] != "mavroeidis-scan"]
    scan = [r for r in rs if r["doc"] == "mavroeidis-scan"]
    chi = [r for r in rs if r["doc"] == "chi-report"]
    heads = [r for r in rs if r.get("h_recall") is not None]
    n_truth = sum(len(TRUTH.get(r["doc"], [])) for r in heads)
    found = sum(r["h_recall"] * len(TRUTH.get(r["doc"], [])) for r in heads)
    # weight each doc's hierarchy score by its consecutive-heading pairs
    pairs = [(r["h_hierarchy"], round(r["h_recall"] * len(TRUTH.get(r["doc"], []))) - 1)
             for r in heads if r.get("h_hierarchy") is not None]
    hier_ok, hier_n = sum(h * n for h, n in pairs), sum(n for _, n in pairs)
    pages = sum(r["pages"] for r in rs)
    print(f"== {run}  ({len(rs)} docs, {pages} pages{', not scored: ' + ', '.join(bad) if bad else ''})")
    print(f"  born-digital R/P/O ({sum(r['pages'] for r in born)} pp): "
          f"{pw(born, 'recall'):.3f} / {pw(born, 'precision'):.3f} / {pw(born, 'order'):.3f}")
    if scan:
        print(f"  scan R/P/O: {scan[0]['recall']:.3f} / {scan[0]['precision']:.3f} / {scan[0]['order']:.3f}")
    if chi:
        print(f"  Chi infographic recall: {chi[0]['recall']:.3f}")
    print(f"  numbered headings found: {found:.0f}/{n_truth} = {found / max(n_truth, 1):.3f}; "
          f"hierarchy: {hier_ok:.0f}/{hier_n} pairs = {hier_ok / max(hier_n, 1):.3f}")
    print(f"  tables / cells: {sum(r['tables'] for r in rs)} / {sum(r['table_cells'] for r in rs)}; "
          f"figures {sum(r['figures'] for r in rs)}, described {sum(r['described'] for r in rs)}, "
          f"chart data {sum(r['chart_data'] for r in rs)}")
    print(f"  throughput: {sum(r['s_per_page'] * r['pages'] for r in rs) / max(pages, 1):.2f} s/page (wall, page-weighted)")
    if "leaked" in rs[0]:
        print(f"  latex spans {sum(r['latex'] for r in rs)}, leaked description lines {sum(r['leaked'] for r in rs)}")
