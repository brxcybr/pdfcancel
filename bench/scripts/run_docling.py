"""Convert every corpus PDF with docling-serve (async API, one request at a time)."""
import json, os, sys, time
import httpx

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

def run(config: str) -> None:
    out = f"{ROOT}/out/{config}"; os.makedirs(out, exist_ok=True)
    data = {**COMMON, **CONFIGS[config]}
    with httpx.Client(timeout=httpx.Timeout(120, connect=15)) as c:
        for name in sorted(os.listdir(f"{ROOT}/corpus")):
            stem = name[:-4]
            if os.path.exists(f"{out}/{stem}.json"):
                continue
            t0 = time.time()
            with open(f"{ROOT}/corpus/{name}", "rb") as f:
                r = c.post(f"{BASE}/v1/convert/file/async", data=data, files={"files": (name, f, "application/pdf")})
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
                       "doc": doc.get("json_content")}, open(f"{out}/{stem}.json", "w"))
            print(f"{config} {stem}: {res.get('status')} {wall:.1f}s", flush=True)

for cfg in sys.argv[1:]:
    run(cfg)
