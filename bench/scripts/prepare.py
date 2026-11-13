"""Build the benchmark corpus (trimmed / synthetic-scan PDFs) and text-layer ground truth."""
import glob, os, sys
import pymupdf

SRC = os.path.expanduser(os.environ["BENCH_SOURCE_PDFS"])  # folder holding the source papers
OUT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = [  # (name, source glob, first page, last page) 1-indexed, inclusive
    ("cascavilla", "Cascavilla - 2021*.pdf", 1, 29),
    ("alaeifar", "Alaeifar - 2024*.pdf", 1, 20),
    ("chi-report", "Chi - 2022*.pdf", 1, 25),
    ("taxii-spec", "taxii-v2.1-os.pdf", 20, 39),
    ("barrios", "Barrios-Gonz*2026*.pdf", 1, 15),
    ("zibak", "Zibak et al. - 2022*.pdf", 1, 12),
    ("mavroeidis", "Mavroeidis and Bromander*.pdf", 1, 8),
]

def main():
    for name, pat, a, b in DOCS:
        src = pymupdf.open(glob.glob(os.path.join(SRC, pat))[0])
        doc = pymupdf.open()
        doc.insert_pdf(src, from_page=a - 1, to_page=min(b, src.page_count) - 1)
        doc.save(f"{OUT}/corpus/{name}.pdf", garbage=3, deflate=True)
        with open(f"{OUT}/truth/{name}.txt", "w") as f:
            f.write("\f".join(pg.get_text() for pg in doc))
        print(name, doc.page_count, "pages")
    # Synthetic scan: rasterize mavroeidis to grayscale images, no text layer.
    src = pymupdf.open(f"{OUT}/corpus/mavroeidis.pdf")
    scan = pymupdf.open()
    for pg in src:
        pix = pg.get_pixmap(dpi=200, colorspace=pymupdf.csGRAY)
        page = scan.new_page(width=pg.rect.width, height=pg.rect.height)
        page.insert_image(page.rect, stream=pix.tobytes("jpg", jpg_quality=85))
    scan.save(f"{OUT}/corpus/mavroeidis-scan.pdf", deflate=True)
    with open(f"{OUT}/truth/mavroeidis-scan.txt", "w") as f:
        f.write(open(f"{OUT}/truth/mavroeidis.txt").read())
    assert not "".join(pg.get_text() for pg in pymupdf.open(f"{OUT}/corpus/mavroeidis-scan.pdf")).strip()
    print("mavroeidis-scan", scan.page_count, "pages (image-only)")

main()
