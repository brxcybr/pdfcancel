"""Convert every corpus PDF with the pdfcancel skill (vlm OCR + local figures + --full)."""
import json, os, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W = os.path.expanduser("~/.agents/skills/pdfcancel/scripts/pdfcancel-skill.sh")
out = f"{ROOT}/out/pdfcancel-v3"; os.makedirs(out, exist_ok=True)
for name in sorted(os.listdir(f"{ROOT}/corpus")):
    stem = name[:-4]
    if os.path.exists(f"{out}/{stem}.timing.json"):
        continue
    t0 = time.time(); marks = {}
    p = subprocess.Popen([W, "convert", f"{ROOT}/corpus/{name}", "-o", out, "--force"],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log = []
    for line in p.stdout:
        t = time.time() - t0; log.append(f"{t:7.1f} {line.rstrip()}")
        for key, needle in (("ocr_done", "Cropped"), ("describe_start", "Describing "), ("chunk_start", "Chunking")):
            if needle in line and key not in marks:
                marks[key] = t
    rc = p.wait(); wall = time.time() - t0
    open(f"{out}/{stem}.log", "w").write("\n".join(log))
    json.dump({"wall": wall, "rc": rc, **marks}, open(f"{out}/{stem}.timing.json", "w"))
    print(f"pdfcancel {stem}: rc={rc} {wall:.1f}s marks={marks}", flush=True)
