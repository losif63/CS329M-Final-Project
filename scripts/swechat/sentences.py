"""Sentence splitting for comment text, with line mapping back to the source file.

Protected spans (backticks, URLs, paths, versions, dotted names) have their
sentence punctuation swapped for same-length private-use characters before
pysbd runs, so character offsets are preserved and restored afterwards.
"""

from __future__ import annotations

import re
from functools import lru_cache

import pysbd

_PROTECT_RE = re.compile(
    r"``[^`\n]+``|`[^`\n]+`"                                # ``rst`` and `md` code spans
    r"|\b(?:https?|ftp|file)://\S+|\bwww\.\S+"              # URLs
    r"|(?<![\w/])(?:~|\.{1,2})?/?(?:[\w.@+-]+/)+[\w.@+-]*(?:\?[\w=&%.-]*)?"  # paths a/b.py, ./x, /api/x?q=
    r"|\b[vV]?\d+(?:\.(?:\d+|[xX*]))+\w*"                   # versions 3.6, 5.x, v1.2.3rc1
    r"|\b\w+(?:\.\w+)+(?:\(\))?"                            # dotted names foo.bar(), a.b.c
    r"|(?<![\w.])\.\w+(?:\(\))?"                             # leading-dot names .strip(), .py
    r"|\b\w+\(\)"                                           # calls foo()
    r"|\b\w+(?:::\w+)+(?:\(\))?"                              # paths a::b::c() (Rust, C++, PHP)
    r"|\{@\w+[^}\n]*\}"                                     # JSDoc inline tags {@link Foo.bar}
)
_SWAP = str.maketrans({".": "\ue000", "?": "\ue001", "!": "\ue002", ";": "\ue003"})
_BULLET_RE = re.compile(r"^\s*([-*•+]|\d+[.)]|[a-z][.)])\s+")
_SECTION_RE = re.compile(
    r"^\s*(Args|Arguments|Parameters|Params|Returns?|Yields?|Raises?|Examples?|Notes?|Attributes|"
    r"See Also|Warnings?|Todo|References|Keyword Args|Other Parameters)\s*:?\s*$", re.I)
_HEADER_RE = re.compile(r"^\s*[\w*`'\"-]+(?:\s+[\w*`'\"-]+){0,2}\s*:\s*$")  # "Usage:", "Two halves:"
_PARAM_RE = re.compile(r"^\s*\*{0,2}\w+(\s*\([^)]*\))?\s*:\s")  # "name (type): desc"
_RST_UNDERLINE_RE = re.compile(r"^\s*[-=~^]{3,}\s*$")


@lru_cache(maxsize=1)
def _segmenter() -> pysbd.Segmenter:
    return pysbd.Segmenter(language="en", clean=False)


def protect(text: str) -> str:
    return _PROTECT_RE.sub(lambda m: m.group(0).translate(_SWAP), text)


def split_text(text: str) -> list[tuple[int, int]]:
    """Split one paragraph into sentences; returns (start, end) char spans."""
    if not text.strip():
        return []
    prot = protect(text)
    spans = []
    pos = 0
    for sent in _segmenter().segment(prot):
        s = sent.strip()
        if not s:
            continue
        i = prot.find(s, pos)
        if i < 0:  # should not happen with clean=False; fall back to whole text
            return [(len(text) - len(text.lstrip()), len(text.rstrip()))]
        spans.append((i, i + len(s)))
        pos = i + len(s)
    return spans


_CONTINUES_RE = re.compile(r"\b(?:the|a|an|of|to|in|on|for|with|and|or|but|by|from|as|at|that|which|is|are|was|"
                          r"be|not|than|into|via|per|this|its|their|our|your|when|if|so)$", re.I)


def _unpunctuated_break(prev: str, nxt: str) -> bool:
    """Consecutive comment lines that are separate statements: the previous line has no
    terminal punctuation, doesn't end mid-phrase, and the next line starts a new capitalized
    clause ("# Stateful Docker API mock" / "# Simulates container operations")."""
    p, n = prev.rstrip(), nxt.lstrip()
    if not p or not n or p[-1] in ".!?:;,-–—([{/\\&|+*=" or _CONTINUES_RE.search(p):
        return False
    return bool(re.match(r"[A-Z][a-z]", n))


def paragraphs(lines: list[tuple[int, str]]) -> list[list[tuple[int, str]]]:
    """Group comment lines into paragraphs: blank lines, bullets, docstring
    section headers and "name: desc" param lines start a new paragraph."""
    paras: list[list[tuple[int, str]]] = []
    cur: list[tuple[int, str]] = []
    for ln, t in lines:
        if not t.strip() or _RST_UNDERLINE_RE.match(t):
            if cur:
                paras.append(cur)
            cur = []
            continue
        if _SECTION_RE.match(t) or _HEADER_RE.match(t):
            if cur:
                paras.append(cur)
            cur = []
            continue  # bare "Args:" / "Usage:" headers carry no prose
        if cur and (_BULLET_RE.match(t) or _PARAM_RE.match(t) or _unpunctuated_break(cur[-1][1], t)):
            paras.append(cur)
            cur = []
        cur.append((ln, t))
    if cur:
        paras.append(cur)
    return paras


def split_comment(lines: list[tuple[int, str]]) -> list[tuple[str, int, int]]:
    """(sentence, line_start, line_end) for the prose lines of one comment."""
    out = []
    for para in paragraphs(lines):
        # join wrapped lines with a space; remember each line's char offset
        parts, offs, pos = [], [], 0
        for ln, t in para:
            s = t.strip()
            offs.append((pos, ln))
            parts.append(s)
            pos += len(s) + 1
        joined = " ".join(parts)

        def line_at(ch: int) -> int:
            ln_found = offs[0][1]
            for start, ln in offs:
                if start <= ch:
                    ln_found = ln
                else:
                    break
            return ln_found

        for a, b in split_text(joined):
            sent = joined[a:b].strip()
            if not re.search(r"[A-Za-z0-9]", sent):
                continue
            out.append((sent, line_at(a), line_at(b - 1)))
    return out


def norm_sentence(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().rstrip(".!?;:,").strip().lower()
