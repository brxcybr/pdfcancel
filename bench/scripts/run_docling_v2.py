"""Convert a folder of PDFs with docling-serve into a named output folder.

    run_docling_v2.py <config> <input dir> <out name>

<config> is a key of run_docling.CONFIGS (docling-std, docling-full). Recurses
into the input dir; outputs are flat, named by stem, and skipped if present.
"""
import json, os, sys, time
from pathlib import Path
import httpx

# Same server and configs as run_docling.py (which runs on import, so they're copied).
BASE = os.environ["DOCLING_URL"]  # docling-serve, e.g. http://host:5001
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS = {
    "docling-std": {},
    "docling-full": {
        "do_picture_description": "true", "picture_description_preset": "granite_vision",
        "do_chart_extraction": "true", "do_picture_classification": "true",
        "do_formula_enrichment": "true", "do_code_enrichment": "true",
    },
}
COMMON = {"to_formats": ["md", "json"], "image_export_mode": "placeholder",
          "md_page_break_placeholder": "<!-- page-break -->"}

config, src, name = sys.argv[1], Path(os.path.expanduser(sys.argv[2])), sys.argv[3]
out = f"{ROOT}/out/{name}"; os.makedirs(out, exist_ok=True)
data = {**COMMON, **CONFIGS[config]}
with httpx.Client(timeout=httpx.Timeout(120, connect=15)) as c:
    for pdf in sorted(src.rglob("*.pdf")):
        stem = pdf.stem
        if os.path.exists(f"{out}/{stem}.json"):
            continue
        t0 = time.time()
        with open(pdf, "rb") as f:
            r = c.post(f"{BASE}/v1/convert/file/async", data=data, files={"files": (pdf.name, f, "application/pdf")})
        r.raise_for_status(); task = r.json()["task_id"]
        while True:
            s = c.get(f"{BASE}/v1/status/poll/{task}", params={"wait": 10}).json()
            if s["task_status"] in ("success", "failure", "partial_success"):
                break
        res = c.get(f"{BASE}/v1/result/{task}").json()
        wall = time.time() - t0
        doc = res.get("document") or {}
        open(f"{out}/{stem}.md", "w").write(doc.get("md_content") or "")
        json.dump({"wall": wall, "status": res.get("status"), "errors": res.get("errors"),
                   "processing_time": res.get("processing_time"), "timings": res.get("timings"),
                   "doc": doc.get("json_content"),
                   # keep the server's reply when it carries no document (task error, 404, ...)
                   **({} if doc else {"raw": res, "last_poll": s})}, open(f"{out}/{stem}.json", "w"))
        print(f"{name} {stem}: {res.get('status')} {wall:.1f}s", flush=True)
