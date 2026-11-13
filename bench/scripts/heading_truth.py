"""Numbered section headings from each PDF's text layer (hand-checked list).

Candidates are text-layer lines that start with a section number (1. / 2.3 /
II. / B.), or a number alone on a line followed by its title (Elsevier and ACM
layouts). Printed output was checked by hand against the PDFs; EXCLUDE drops
the few false candidates. chi-report has no numbered sections.
"""
import json, os, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NUM = re.compile(r"^(?:(\d{1,2}(?:\.\d{1,2}){0,3})\.?|([IVX]{1,5})\.|([A-H])\.)$")
LINE = re.compile(r"^(?:(\d{1,2}(?:\.\d{1,2}){0,3})\.?|([IVX]{1,5})\.|([A-H])\.)\s+([A-Z][^\n]{1,110})$")
EXCLUDE = {
    "alaeifar": {"9. Collaborative"},
    "cascavilla": {"2 Managed Security", "1. Practical experience", "2. Industrial case-study",
                   "3. Heterogeneity", "4. Implementation quantity"},
}
SKIP = {"chi-report"}
TRUTH_OF = {"mavroeidis-scan": "mavroeidis"}


def _depth(m) -> int:
    if m.group(1):
        return m.group(1).count(".") + 1
    return 1 if m.group(2) else 2


def headings(stem: str) -> list[dict]:
    if stem in SKIP:
        return []
    lines = [ln.strip() for ln in open(f"{ROOT}/truth/{TRUTH_OF.get(stem, stem)}.txt")]
    out, state = [], {}
    for i, ln in enumerate(lines):
        m = LINE.match(ln)
        text = ln
        if not m and NUM.match(ln) and i + 1 < len(lines):
            text = f"{ln} {lines[i + 1]}"
            m = LINE.match(text)
        if not m or len(text.split()) > 14 or text.endswith((".", ",", ";")):
            continue
        if re.search(r"\b(January|February|March|April|May|June|July|August|September|"
                     r"October|November|December) \d{4}", text):
            continue
        if text.startswith(tuple(EXCLUDE.get(stem, ()))) or any(text == h["text"] for h in out):
            continue
        num = _num(m)
        if not _follows(state, num):
            continue
        out.append({"text": text, "depth": _depth(m)})
    return out


def _num(m) -> tuple:
    if m.group(1):
        return ("d", tuple(int(p) for p in m.group(1).split(".")))
    if m.group(2):
        return ("r", "I II III IV V VI VII VIII IX X XI XII".split().index(m.group(2)) + 1)
    return ("l", ord(m.group(3)) - 64)


def _follows(state: dict, num: tuple) -> bool:
    """Strict section order: the next sibling or the first child. Body text,
    footnotes, list items, and page heads that start with a number fall out.
    Updates ``state`` when ``num`` is accepted."""
    kind, n = num
    if kind == "r":
        ok = n == state.get("r", 0) + 1
        if ok:
            state.update(r=n, l=0)
        return ok
    if kind == "l":
        ok = state.get("r", 0) > 0 and n == state.get("l", 0) + 1
        if ok:
            state["l"] = n
        return ok
    a = state.get("d")
    if a is None:
        ok = n in ((1,), (3, 3))  # the TAXII excerpt starts at 3.3
    elif len(n) == len(a) + 1:
        ok = n[:-1] == a and n[-1] == 1
    else:
        ok = len(n) <= len(a) and n[:-1] == a[:len(n) - 1] and n[-1] == a[len(n) - 1] + 1
    if ok:
        state["d"] = n
    return ok


if __name__ == "__main__":
    import glob
    truth = {}
    for pdf in sorted(glob.glob(f"{ROOT}/corpus/*.pdf")):
        stem = os.path.basename(pdf)[:-4]
        truth[stem] = headings(stem)
        print(f"== {stem} {len(truth[stem])}: " + " | ".join(h["text"][:28] for h in truth[stem]))
    json.dump(truth, open(f"{ROOT}/truth/headings.json", "w"), indent=1)
