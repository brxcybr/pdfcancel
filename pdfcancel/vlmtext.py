"""Repair common failure modes in vision-model page transcriptions.

- Repetition loops: a model occasionally repeats one phrase until it hits its
  token cap (one benchmark page repeated a 6-word phrase 624 times).
  A model pushed out of a loop can also degenerate into letter soup
  (``lI lI lL I I ee CC C ii``). :func:`find_loop` detects both, so the caller
  can retry; :func:`collapse_loops` cuts a loop to one copy and drops soup.
- LaTeX used for layout: models wrap ordinary words and spacing in math
  (``$\\text{Accuracy}\\quad 0.92$``). :func:`clean_latex` unwraps that and
  leaves real math alone; a lone symbol (``$\\checkmark$``) becomes its
  Unicode glyph so it survives into search text.
"""

from __future__ import annotations

import re

LOOP_MIN_REPEATS = 8     # consecutive copies of one unit ...
LOOP_MIN_WORDS = 40      # ... spanning at least this many words
LOOP_MAX_UNIT = 40       # longest repeated unit (words) looked for
SOUP_MIN_RUN = 30         # consecutive bare 1-2 letter tokens: never prose
LOOP_NOTE = "<!-- pdfcancel: repeated text removed -->"

_TOKEN = re.compile(r"\S+")
_WORDLIKE = re.compile(r"[A-Za-z]")
_SOUP = re.compile(r"[A-Za-z]{1,2}")


def _loops(text: str) -> list[tuple[int, int, int]]:
    """``[(first_token, end_token, keep)]`` for each loop or soup run.

    ``keep`` is how many leading tokens survive a collapse: one copy of a
    loop's unit, none of a soup run.
    """
    words = [m.group(0) for m in _TOKEN.finditer(text)]
    n, i, out = len(words), 0, []
    while i < n:
        found = None
        j = i
        while j < n and _SOUP.fullmatch(words[j]):
            j += 1
        if j - i >= SOUP_MIN_RUN:
            found = (i, j, 0)
        for p in range(1, LOOP_MAX_UNIT + 1):
            if found:
                break
            if i + 2 * p > n:
                break
            j = i
            while j + p < n and words[j] == words[j + p]:
                j += 1
            span = j - i + p  # tokens covered by the repeating run
            if span // p >= LOOP_MIN_REPEATS and span >= LOOP_MIN_WORDS:
                unit = words[i:i + p]
                # Numbers and table rows (a column of checkboxes) legitimately
                # repeat; a loop has words and no cell borders.
                if "|" not in unit and any(_WORDLIKE.search(w) for w in unit):
                    found = (i, i + span, p)
                    break
        if found:
            out.append(found)
            i = found[1]
        else:
            i += 1
    return out


def find_loop(text: str) -> bool:
    """True when ``text`` contains a repetition loop."""
    return bool(_loops(text))


def clean_prefix(text: str) -> str:
    """``text`` up to where its first loop or soup run starts."""
    loops = _loops(text)
    if not loops:
        return text
    start = [m.start() for m in _TOKEN.finditer(text)][loops[0][0]]
    return text[:start].rstrip()


def collapse_loops(text: str) -> str:
    """Keep one copy of each looping unit and note the cut."""
    loops = _loops(text)
    if not loops:
        return text
    spans = [m.span() for m in _TOKEN.finditer(text)]
    for first, end, keep in reversed(loops):
        keep_to = spans[first + keep - 1][1] if keep else spans[first][0]
        cut_to = spans[end - 1][1]
        text = text[:keep_to] + (" " if keep else "") + LOOP_NOTE + text[cut_to:]
    return text


# ── LaTeX used for layout ────────────────────────────────────────────────────

_SPACING = re.compile(r"\\(?:qquad|quad|enspace|thinspace|[,;:! ])")
_CODE = re.compile(r"```.*?(?:```|\Z)|`[^`\n]+`", re.DOTALL)
_TEXT_CMD = re.compile(r"\\(text|textrm|textit|textbf|mathrm|mathit|mathbf|operatorname)\s*\{([^{}]*)\}")
_MATH = re.compile(r"(?<![\\$])\$(?!\$)((?:\\\$|[^$\n])+?)(?<!\\)\$(?!\$)|\$\$([^$]+?)\$\$",
                   re.DOTALL)
_MATHY = re.compile(r"[\\^_{}=<>]")
# A span holding only one of these is a transcribed glyph, not math.
SYMBOLS = {
    "times": "×", "checkmark": "✓", "square": "☐", "blacksquare": "■", "bullet": "•",
    "cdot": "·", "ast": "*", "star": "★", "rightarrow": "→", "leftarrow": "←",
    "uparrow": "↑", "downarrow": "↓", "leftrightarrow": "↔", "Rightarrow": "⇒",
    "to": "→", "pm": "±", "approx": "≈", "leq": "≤", "geq": "≥", "le": "≤", "ge": "≥",
    "neq": "≠", "dagger": "†", "ddagger": "‡", "S": "§", "degree": "°", "circ": "°",
}
_SYMBOL = re.compile(r"\\(" + "|".join(sorted(SYMBOLS, key=len, reverse=True)) + r")(?![A-Za-z])")


def clean_latex(markdown: str) -> str:
    """Unwrap math spans that only hold words, numbers, and spacing commands.

    ``$\\text{Total}\\quad 42$`` becomes ``Total 42`` and a span of nothing but
    spacing disappears; spans with real math (``$x^2$``, ``$\\alpha$``) keep
    their math, minus layout-only spacing runs.
    """
    def fix(m: re.Match) -> str:
        display = m.group(2) is not None
        body = m.group(2) if display else m.group(1)
        glyph = _SYMBOL.fullmatch(body.strip())
        if glyph and not display:
            return SYMBOLS[glyph.group(1)]
        if not (_SPACING.search(body) or _TEXT_CMD.search(body)):
            return m.group(0)
        styled = {"textbf": "**", "mathbf": "**", "textit": "*", "mathit": "*"}

        def unwrap(t: re.Match) -> str:
            mark = styled.get(t.group(1), "")
            inner = t.group(2).strip()
            return f"{mark}{inner}{mark}" if inner and mark else inner

        plain = _SPACING.sub(" ", _TEXT_CMD.sub(unwrap, body))
        plain = re.sub(r"\\([%&#])", r"\1", plain)  # \$ stays escaped: a bare $ opens math
        plain = re.sub(r"[ \t]{2,}", " ", plain).strip()
        if not plain:
            return " "
        if not _MATHY.search(plain.replace("\\$", "").replace("*", "")):
            return plain
        # Real math: keep it, but drop runs of layout spacing inside.
        body = re.sub(r"(?:\s*\\(?:qquad|quad)\s*)+", " ", body).strip()
        return f"$${body}$$" if display else f"${body}$"

    # Code (shell variables, prices in literals) is left exactly as written.
    out, last = [], 0
    for c in _CODE.finditer(markdown):
        out += [_MATH.sub(fix, markdown[last:c.start()]), c.group(0)]
        last = c.end()
    out.append(_MATH.sub(fix, markdown[last:]))
    return "".join(out)
