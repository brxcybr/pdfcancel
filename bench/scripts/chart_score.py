"""Score pdfcancel's chart data on Cascavilla's seven charts (values read by eye).

Truth: bars read against printed ticks; pies from printed percentages (p8 f02)
or slice angles (p8 f01, ±8); line charts with no printed scale in gridlines
above the baseline; the radar's three clearest axes (±6). A value counts when
within 0.6 (or the stated tolerance) of the truth.
"""
import json, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRUTH = {
    "p007-f01.png": {"private": 8, "government": 10, "scientific": 10, "public": 13},
    "p009-f02.png": {"workshop": 7, "conference": 15, "journal": 10},
    "p008-f02.png": {"case": 33, "experimental": 20, "empirical": 20, "formal": 7, "none": 20},
    "p008-f01.png": {"case": 31, "experience": 42, "empirical": 27},
    "p007-f02.png": dict(zip(map(str, range(2006, 2019)), [1, 1, 0, 1, 4, 4, 4, 4, 5, 5, 5, 7, 8])),
    "p009-f01.png": dict(zip(map(str, range(2006, 2019)), [0, 0, 1, 7, 2, 1, 1, 5, 5, 5, 4, 7, 8])),
    "p009-f03.png": {"crime": 60, "detection": 32, "sourcerisk": 10},
}
TOL = {"p008-f01.png": 8, "p009-f03.png": 6}


def score(img, chart):
    """Best match over the chart's series (a trend line shares the x labels)."""
    groups = []
    for s in (chart or {}).get("series", []) or []:
        pts = {}
        for p in s.get("points", []) or []:
            key = p.get("x") or p.get("label") or s.get("name") or ""
            pts[str(key).lower().replace(" ", "")] = p.get("y")
        groups.append(pts)
    if groups and all(len(g) == 1 for g in groups):  # one series per pie slice
        groups = [{k: v for g in groups for k, v in g.items()}]
    rows = {}
    for r in ((chart or {}).get("table") or {}).get("rows", []) or []:
        vals = list(r.values())
        if len(vals) >= 2:
            rows[str(vals[0]).lower().replace(" ", "")] = vals[1]
    groups.append(rows)
    best = 0
    for pts in groups:
        ok = 0
        for key, want in TRUTH[img].items():
            got = next((v for k, v in pts.items() if key in k), None)
            try:
                ok += abs(float(str(got).strip("~%")) - want) <= TOL.get(img, 0.6)
            except (TypeError, ValueError):
                pass
        best = max(best, ok)
    return best


def charts(md: str) -> dict:
    """{image name: chart JSON} from pdfcancel's hidden chart-data comments."""
    out = {}
    for m in re.finditer(r"!\[([^\]]+)\]\([^)]*\)\n(.*?)(?=\n!\[|\Z)", md, re.S):
        c = re.search(r"<!--\s*pdfcancel-chart-data:\s*(.*?)\s*-->", m.group(2), re.S)
        if c:
            out[m.group(1)] = json.loads(c.group(1)).get("chart_data")
    return out




def docling_probe() -> dict:
    """docling's enriched charts for Cascavilla pp. 7-9 (out/probe2.json), as
    pdfcancel-shaped tables, matched to pdfcancel's crop names top-down per page."""
    doc = json.load(open(f"{ROOT}/out/probe2.json"))["document"]
    doc = doc.get("json_content") or doc
    by_page: dict[int, list] = {}
    for p in doc["pictures"]:
        prov = p["prov"][0]
        by_page.setdefault(prov["page_no"], []).append((-prov["bbox"]["t"], p))
    out = {}
    for page, pics in by_page.items():
        for k, (_, p) in enumerate(sorted(pics, key=lambda x: x[0]), 1):
            cells = ((p.get("meta") or {}).get("tabular_chart") or {}).get("chart_data", {}).get("table_cells", [])
            grid: dict[int, dict[int, str]] = {}
            for c in cells:
                grid.setdefault(c["start_row_offset_idx"], {})[c["start_col_offset_idx"]] = c["text"]
            rows = [[r[c] for c in sorted(r)] for _, r in sorted(grid.items())][1:]
            out[f"p{page:03d}-f{k:02d}.png"] = {"table": {"rows": [dict(enumerate(r)) for r in rows]}}
    return out


def docling_run(run: str) -> dict[int, list]:
    """{page: [pdfcancel-shaped chart tables]} from a full docling run's
    cascavilla.json. A full run can find extra pictures on a page, so each
    truth chart is scored against the best picture on its page."""
    doc = json.load(open(f"{ROOT}/out/{run}/cascavilla.json"))["doc"]
    by_page: dict[int, list] = {}
    for p in doc["pictures"]:
        cells = ((p.get("meta") or {}).get("tabular_chart") or {}).get("chart_data", {}).get("table_cells", [])
        grid: dict[int, dict[int, str]] = {}
        for c in cells:
            grid.setdefault(c["start_row_offset_idx"], {})[c["start_col_offset_idx"]] = c["text"]
        rows = [[r[c] for c in sorted(r)] for _, r in sorted(grid.items())][1:]
        by_page.setdefault(p["prov"][0]["page_no"], []).append(
            {"table": {"rows": [dict(enumerate(r)) for r in rows]}})
    return by_page


if __name__ == "__main__":
    total = sum(len(v) for v in TRUTH.values())
    for run in sys.argv[1:]:
        if run.startswith("docling-full"):
            pages = docling_run(run)
            per = {img: max((score(img, t) for t in pages.get(int(img[1:4]), [])), default=0) for img in TRUTH}
            print(f"{run}: {sum(per.values())}/{total}  " + "  ".join(f"{k[:8]} {v}/{len(TRUTH[k])}" for k, v in per.items()))
            continue
        found = docling_probe() if run == "docling-enriched" else \
            charts(open(f"{ROOT}/out/{run}/cascavilla.md").read())
        per = {img: score(img, found.get(img)) for img in TRUTH}
        print(f"{run}: {sum(per.values())}/{total}  " + "  ".join(f"{k[:8]} {v}/{len(TRUTH[k])}" for k, v in per.items()))
