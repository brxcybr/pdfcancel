"""Score pdfcancel vs docling outputs against PDF text-layer ground truth."""
import difflib, glob, json, os, re, sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORD = re.compile(r"[a-z0-9]{2,}")


def words(text: str) -> list[str]:
    text = text.lower().replace("-\n", "")  # rejoin hyphenated line breaks
    return WORD.findall(text)


def strip_pdfcancel(md: str) -> str:
    """pdfcancel markdown minus generated figure descriptions/chart data and markup."""
    md = re.sub(r"<!--.*?-->", " ", md, flags=re.S)
    md = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", md)
    out, in_desc = [], False
    for line in md.splitlines():
        if line.startswith("> **Figure description:**"):
            in_desc = True
            continue
        if in_desc and line.startswith(">"):
            continue
        in_desc = False
        out.append(line)
    return "\n".join(out)


def docling_text(doc: dict) -> str:
    """Body/furniture text items plus table cells from docling JSON (no generated text)."""
    parts = [t.get("text", "") for t in doc.get("texts", [])]
    for tb in doc.get("tables", []):
        parts += [c.get("text", "") for c in tb.get("data", {}).get("table_cells", [])]
    return "\n".join(parts)


def strip_docling(md: str) -> str:
    """docling markdown minus placeholders/comments (reading order, like pdfcancel's)."""
    return re.sub(r"<!--.*?-->", " ", md, flags=re.S)


def md_tables(md: str) -> tuple[int, int]:
    tables = cells = 0
    in_t = False
    for line in md.splitlines():
        s = line.strip().lstrip("> ").strip()
        if s.startswith("|") and s.endswith("|"):
            if re.fullmatch(r"\|[\s:|-]+\|", s):
                if not in_t:
                    tables += 1
                in_t = True
                continue
            if in_t:
                cells += s.count("|") - 1
        else:
            in_t = False
    return tables, cells


def order_similarity(ref: list[str], out: list[str]) -> float:
    """Share of the reference word sequence matched in order (difflib, sampled)."""
    if not ref or not out:
        return 0.0
    sm = difflib.SequenceMatcher(None, ref, out, autojunk=False)
    return sum(b.size for b in sm.get_matching_blocks()) / len(ref)


def text_scores(truth: str, produced: str) -> dict:
    t, o = Counter(words(truth)), Counter(words(produced))
    hit = sum((t & o).values())
    return {"recall": hit / max(sum(t.values()), 1), "precision": hit / max(sum(o.values()), 1),
            "order": order_similarity(words(truth)[:6000], words(produced)[:8000])}


HEADING_TRUTH = json.load(open(f"{ROOT}/truth/headings.json"))


def heading_scores(stem: str, md: str) -> dict:
    """Recall of the numbered section headings (truth/headings.json) as markdown
    headings, and whether their levels follow the numbering: for consecutive
    found headings, a deeper number must get a deeper level, a shallower one a
    shallower level, and siblings the same level."""
    truth = HEADING_TRUTH.get(stem) or []
    if not truth:
        return {"h_recall": None, "h_hierarchy": None}
    heads = [(len(m.group(1)), words(m.group(2))) for m in re.finditer(r"^(#{1,6})\s+(.*)$", md, re.M)]
    found, start = [], 0
    for t in truth:
        key = words(t["text"])[:4]
        for j in range(start, len(heads)):
            if heads[j][1][:len(key)] == key:
                found.append((t["depth"], heads[j][0]))
                start = j + 1
                break
    sign = lambda x: (x > 0) - (x < 0)
    pairs = list(zip(found, found[1:]))
    ok = sum(sign(b[0] - a[0]) == sign(b[1] - a[1]) for a, b in pairs)
    return {"h_recall": len(found) / len(truth), "h_hierarchy": ok / len(pairs) if pairs else None}


def score_pdfcancel(stem: str, pages: int, run: str = "pdfcancel-full") -> dict:
    d = f"{ROOT}/out/{run}"
    md = open(f"{d}/{stem}.md").read()
    tim = json.load(open(f"{d}/{stem}.timing.json"))
    figs = len(re.findall(r"!\[[^\]]*\]\([^)]*_images/[^)]*\)", md))
    desc = re.findall(r"> \*\*Figure description:\*\* (.*)", md)
    tables, cells = md_tables(strip_pdfcancel(md))
    return {
        **text_scores(open(f"{ROOT}/truth/{stem}.txt").read(), strip_pdfcancel(md)),
        "s_per_page": tim["wall"] / pages,
        "ocr_s_per_page": tim.get("ocr_done", tim["wall"]) / pages,
        "headings": len(re.findall(r"^#{1,6} ", md, re.M)),
        "tables": tables, "table_cells": cells,
        "figures": figs, "described": len(desc),
        "chart_data": md.count("pdfcancel-chart-data"),
        "desc_words": sum(len(x.split()) for x in desc) / max(len(desc), 1),
        "latex": len(re.findall(r"\$[^$\n]+\$", strip_pdfcancel(md))),
        "exact": md.count("confidence=exact"), "estimated": md.count("confidence=estimated"),
        "leaked": _leaked(md),
        **heading_scores(stem, md),
    }


def _leaked(md: str) -> int:
    """Lines between a description's start and the next blank line that lost their '>'."""
    n, inside = 0, False
    for line in md.splitlines():
        if line.startswith("> **Figure description:**"):
            inside = True
        elif inside and not line.strip():
            inside = False
        elif inside and not line.startswith(">") and not line.startswith("<!--"):
            n += 1
    return n


def score_docling(config: str, stem: str, pages: int) -> dict:
    d = f"{ROOT}/out/{config}"
    meta = json.load(open(f"{d}/{stem}.json"))
    doc = meta.get("doc") or {}
    md = open(f"{d}/{stem}.md").read()
    pics = doc.get("pictures", [])
    desc = [p["meta"]["description"]["text"] for p in pics
            if (p.get("meta") or {}).get("description")]
    chart = [p for p in pics if any(k in (p.get("meta") or {}) for k in ("tabular_chart", "chart_data"))
             or any("chart" in str(a.get("kind", "")) and a.get("kind") != "classification"
                    for a in p.get("annotations", []))]
    headings = sum(1 for t in doc.get("texts", []) if t.get("label") in ("section_header", "title"))
    tcells = sum(len(tb.get("data", {}).get("table_cells", [])) for tb in doc.get("tables", []))
    return {
        **text_scores(open(f"{ROOT}/truth/{stem}.txt").read(), strip_docling(md)),
        "s_per_page": meta["wall"] / pages, "ocr_s_per_page": None,
        "headings": headings, "tables": len(doc.get("tables", [])), "table_cells": tcells,
        "figures": len(pics), "described": len(desc), "chart_data": len(chart),
        "desc_words": sum(len(x.split()) for x in desc) / max(len(desc), 1),
        "latex": sum(1 for t in doc.get("texts", []) if t.get("label") == "formula"),
        "status": meta.get("status"),
        **heading_scores(stem, md),
    }


def main():
    import pymupdf
    rows = []
    for pdf in sorted(glob.glob(f"{ROOT}/corpus/*.pdf")):
        stem = os.path.basename(pdf)[:-4]
        pages = pymupdf.open(pdf).page_count
        for tool, fn in (("pdfcancel-full", lambda: score_pdfcancel(stem, pages)),
                         ("pdfcancel-v3", lambda: score_pdfcancel(stem, pages, "pdfcancel-v3")),
                         ("pdfcancel-v4", lambda: score_pdfcancel(stem, pages, "pdfcancel-v4")),
                         ("pdfcancel-v5", lambda: score_pdfcancel(stem, pages, "pdfcancel-v5")),
                         ("pdfcancel-v6", lambda: score_pdfcancel(stem, pages, "pdfcancel-v6")),
                         ("docling-std", lambda: score_docling("docling-std", stem, pages)),
                         ("docling-std-v2", lambda: score_docling("docling-std-v2", stem, pages)),
                         ("docling-full", lambda: score_docling("docling-full", stem, pages)),
                         ("docling-full-v2", lambda: score_docling("docling-full-v2", stem, pages))):
            try:
                rows.append({"doc": stem, "pages": pages, "tool": tool, **fn()})
            except FileNotFoundError:
                pass
    json.dump(rows, open(f"{ROOT}/out/metrics.json", "w"), indent=1)
    cols = ["recall", "precision", "order", "s_per_page", "headings", "tables", "table_cells",
            "figures", "described", "chart_data", "desc_words", "latex", "exact", "leaked", "h_recall", "h_hierarchy"]
    print(f"{'doc':16} {'tool':15} " + " ".join(f"{c[:10]:>10}" for c in cols))
    for r in rows:
        vals = []
        for c in cols:
            v = r.get(c)
            vals.append(f"{v:10.2f}" if isinstance(v, float) else f"{v if v is not None else '-':>10}")
        print(f"{r['doc']:16} {r['tool']:15} " + " ".join(vals))


if __name__ == "__main__":
    main()
