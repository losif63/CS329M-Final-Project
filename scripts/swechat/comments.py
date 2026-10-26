"""Extract comments (inline, block, docstring) from source code and filter non-prose.

Python uses tokenize + ast (docstrings); every other language uses tree-sitter.

Each comment carries `lines`: a list of (file_line_number, text) pairs with comment
markers / docstring quotes removed, so sentences can be mapped back to lines.
"""

from __future__ import annotations

import ast
import difflib
import io
import re
import textwrap
import tokenize
import warnings
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

from languages import CODE_KEYWORDS, DELIMS, DIRECTIVE_RE, EMBEDDED_SCRIPT, LINE_PREFIXES


@dataclass
class Comment:
    kind: str  # inline | block | docstring
    lines: list[tuple[int, str]]  # (1-based line in source, text)
    col: int = 0
    near_error: bool = False  # within a few lines of a parse error: boundaries may be wrong

    @property
    def line_start(self) -> int:
        return self.lines[0][0]

    @property
    def line_end(self) -> int:
        return self.lines[-1][0]

    @property
    def text(self) -> str:
        return "\n".join(t for _, t in self.lines).strip("\n")


def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


# ---------------------------------------------------------------- extraction

def _strip_hash(tok: str) -> str:
    s = tok[1:]
    return s[1:] if s.startswith(" ") else s


def _hash_comments_tokenize(src: str) -> list[tuple[int, int, str, bool]]:
    """(line, col, text, full_line) for every COMMENT token. Raises on tokenize error."""
    out = []
    src_lines = src.splitlines()
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.COMMENT:
            ln, col = tok.start
            line = src_lines[ln - 1] if ln - 1 < len(src_lines) else ""
            full = line[:col].strip() == ""
            out.append((ln, col, _strip_hash(tok.string), full))
    return out


def _hash_comments_linescan(src: str) -> list[tuple[int, int, str, bool]]:
    """Fallback: approximate '#' comment detection ignoring quoted strings."""
    out = []
    in_triple: str | None = None
    for ln, line in enumerate(src.splitlines(), start=1):
        i, quote = 0, None
        while i < len(line):
            if in_triple:
                j = line.find(in_triple, i)
                if j < 0:
                    break
                i, in_triple = j + 3, None
                continue
            ch = line[i]
            if quote:
                if ch == "\\":
                    i += 2
                    continue
                if ch == quote:
                    quote = None
            elif line.startswith(('"""', "'''"), i):
                in_triple = line[i : i + 3]
                i += 3
                continue
            elif ch in "\"'":
                quote = ch
            elif ch == "#":
                out.append((ln, i, _strip_hash(line[i:]), line[:i].strip() == ""))
                break
            i += 1
    return out


def _group_hash(raw: list[tuple[int, int, str, bool]]) -> list[Comment]:
    """Consecutive full-line comments at the same column form one block."""
    comments: list[Comment] = []
    cur: Comment | None = None
    for ln, col, text, full in raw:
        if not full:
            cur = None
            comments.append(Comment("inline", [(ln, text)], col))
            continue
        if cur is not None and cur.col == col and cur.line_end == ln - 1:
            cur.lines.append((ln, text))
        else:
            cur = Comment("block", [(ln, text)], col)
            comments.append(cur)
    return comments


_DOC_OPEN_RE = re.compile(r"""^(\s*)([rRuU]?)(\"\"\"|'''|\"|')""")


def _docstring_lines(src_lines: list[str], start: int, end: int) -> list[tuple[int, str]] | None:
    """Map a docstring literal spanning [start, end] (1-based) to (line, text) pairs."""
    first = src_lines[start - 1]
    m = _DOC_OPEN_RE.match(first)
    if not m:
        return None
    q = m.group(3)
    raw = [(start, first[m.end():])] + [(ln, src_lines[ln - 1]) for ln in range(start + 1, end + 1)]
    ln_last, last = raw[-1]
    k = last.rfind(q)
    if k < 0:
        return None
    raw[-1] = (ln_last, last[:k])
    # cleandoc-style dedent of lines after the first
    indents = [len(t) - len(t.lstrip()) for _, t in raw[1:] if t.strip()]
    ind = min(indents) if indents else 0
    lines = [(raw[0][0], raw[0][1].strip())] + [(ln, t[ind:].rstrip()) for ln, t in raw[1:]]
    while lines and not lines[0][1].strip():
        lines.pop(0)
    while lines and not lines[-1][1].strip():
        lines.pop()
    return lines or None


def _docstrings_ast(tree: ast.AST, src_lines: list[str]) -> list[Comment]:
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = getattr(node, "body", [])
        if not body:
            continue
        first = body[0]
        if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)):
            lines = _docstring_lines(src_lines, first.lineno, first.end_lineno or first.lineno)
            if lines:
                out.append(Comment("docstring", lines))
    return out


_DEF_RE = re.compile(r"^\s*(async\s+def|def|class)\s.*:\s*(#.*)?$")


def _docstrings_linescan(src_lines: list[str]) -> list[Comment]:
    """Fallback: a triple-quoted string right after a def/class line or at file top."""
    out = []
    i = 0
    prev_code = None
    while i < len(src_lines):
        line = src_lines[i]
        s = line.strip()
        if not s:
            i += 1
            continue
        m = re.match(r"""^\s*[rRuU]?(\"\"\"|''')""", line)
        if m and (prev_code is None or _DEF_RE.match(prev_code)):
            q = m.group(1)
            rest = line[m.end():]
            j = i if q in rest else next((k for k in range(i + 1, len(src_lines)) if q in src_lines[k]), None)
            if j is None:
                break
            lines = _docstring_lines(src_lines, i + 1, j + 1)
            if lines:
                out.append(Comment("docstring", lines))
            prev_code, i = src_lines[j], j + 1
            continue
        prev_code = line
        i += 1
    return out


def _extract_python(src: str) -> tuple[list[Comment], bool]:
    """Python: (comments sorted by line, used_fallback)."""
    src_lines = src.splitlines()
    fallback = False
    try:
        raw = _hash_comments_tokenize(src)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        try:  # snippets are often indented fragments; line numbers survive dedent
            raw = _hash_comments_tokenize(textwrap.dedent(src))
        except (tokenize.TokenError, IndentationError, SyntaxError):
            raw, fallback = _hash_comments_linescan(src), True
    comments = _group_hash(raw)
    tree = None
    for text in (src, textwrap.dedent(src)):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tree = ast.parse(text)
            comments += _docstrings_ast(tree, text.splitlines())
            break
        except (SyntaxError, ValueError, RecursionError, MemoryError):
            continue
    if tree is None:
        fallback = True
        comments += _docstrings_linescan(src_lines)
    comments.sort(key=lambda c: (c.line_start, c.kind != "docstring"))
    return comments, fallback


# ---------------------------------------------------------------- tree-sitter (non-Python)

@lru_cache(maxsize=None)
def _parser(lang: str):
    import tree_sitter_language_pack as tslp
    return tslp.get_parser(lang)


_GO_DECLS = {"function_declaration", "method_declaration", "type_declaration", "const_declaration",
             "var_declaration", "package_clause"}
_SCRIPT_RE = re.compile(r"<script\b([^>]*)>(.*?)</script>", re.S | re.I)


def _strip_delims(raw: str, lang: str, node_type: str) -> tuple[list[str], bool, bool, str] | None:
    """Comment text -> (lines without markers, is_doc, is_line_comment, marker)."""
    raw = raw.rstrip("\r\n")
    for open_, close, doc in DELIMS:
        if not raw.startswith(open_):
            continue
        if open_ in LINE_PREFIXES and lang not in LINE_PREFIXES[open_]:
            continue
        if close is None:
            text = raw[len(open_):]
            return [text[1:] if text.startswith(" ") else text], doc or node_type == "documentation_comment", True, open_
        body = raw.rstrip()
        if not body.endswith(close) or len(body) < len(open_) + len(close):
            continue
        if open_ == "/**" and body == "/**/":
            return [""], False, False, open_
        inner = body[len(open_):len(body) - len(close)]
        lines = inner.split("\n")
        if open_.startswith("/*"):
            # " * text" continuation lines
            lines = [lines[0]] + [re.sub(r"^\s*\* ?", "", ln) if re.match(r"^\s*\*", ln) else ln for ln in lines[1:]]
        rest = [ln for ln in lines[1:] if ln.strip()]
        ind = min((len(ln) - len(ln.lstrip()) for ln in rest), default=0)
        lines = [lines[0].strip()] + [ln[ind:].rstrip() for ln in lines[1:]]
        return lines, doc, False, open_
    return None


_ERROR_MARGIN = 2  # comments this close (in lines) to a parse error are not trusted
_PERVASIVE_ERRORS = 0.2  # error regions covering this share of the file make it unreliable as a whole


def _error_rows(root, row_off: int) -> set[int]:
    rows: set[int] = set()
    stack = [root]
    while stack:
        n = stack.pop()
        if n.type == "ERROR" or n.is_missing:
            rows.update(range(n.start_point[0] + row_off, n.end_point[0] + row_off + 1))
            continue
        if n.has_error:
            stack.extend(n.children)
    return rows


def _ts_raw_comments(src: bytes, lang: str, row_off: int = 0) -> tuple[list[dict], set[int]]:
    tree = _parser(lang).parse(src)
    lines = src.split(b"\n")
    out = []
    stack = [tree.root_node]
    while stack:
        n = stack.pop()
        if "comment" in n.type:
            raw = src[n.start_byte:n.end_byte].decode("utf-8", errors="replace")
            stripped = _strip_delims(raw, lang, n.type)
            if stripped is not None:
                (r0, c0), (r1, _) = n.start_point, n.end_point
                text_lines, is_doc, is_line, marker = stripped
                if is_line:
                    r1 = r0  # line comments end at the newline
                prefix = lines[r0][:c0] if r0 < len(lines) else b""
                nxt = n.next_named_sibling
                out.append({"r0": r0 + row_off, "r1": r1 + row_off, "col": c0, "lines": text_lines,
                            "doc": is_doc, "line": is_line, "full": prefix.strip() == b"", "mk": marker,
                            "next_type": nxt.type if nxt is not None else None,
                            "next_row": (nxt.start_point[0] + row_off) if nxt is not None else None})
            continue
        stack.extend(reversed(n.children))
    return out, (_error_rows(tree.root_node, row_off) if tree.root_node.has_error else set())


def _extract_treesitter(src: str, lang: str) -> tuple[list[Comment], bool]:
    data = src.encode("utf-8")
    raw, err_rows = _ts_raw_comments(data, lang)
    if lang in EMBEDDED_SCRIPT:
        for m in _SCRIPT_RE.finditer(src):
            sub_lang = "typescript" if re.search(r"lang\s*=\s*[\"']?ts", m.group(1)) else "javascript"
            row_off = src[:m.start(2)].count("\n")
            sub, sub_err = _ts_raw_comments(m.group(2).encode("utf-8"), sub_lang, row_off)
            raw += sub
            err_rows |= sub_err
    raw.sort(key=lambda r: (r["r0"], r["col"]))
    comments: list[Comment] = []
    cur: Comment | None = None
    cur_raw: dict | None = None
    for r in raw:
        lines = [(r["r0"] + 1 + i, t) for i, t in enumerate(r["lines"])]
        mergeable = (r["line"] and r["full"] and cur is not None and cur_raw is not None and cur_raw["line"]
                     and cur_raw["full"] and cur.col == r["col"] and cur.line_end == r["r0"]
                     and cur_raw["mk"] == r["mk"])
        if mergeable:
            cur.lines += lines
            cur_raw = r
            continue
        kind = "docstring" if r["doc"] else ("block" if r["full"] else "inline")
        cur = Comment(kind, lines, r["col"])
        cur_raw = r
        comments.append(cur)
        if lang == "go":
            cur_raw["_c"] = cur
    if lang == "go":
        # Go doc comments: a // block immediately followed by a declaration
        for c, r in _go_doc_candidates(comments, raw):
            if r["next_type"] in _GO_DECLS and r["next_row"] == c.line_end and c.kind == "block":
                c.kind = "docstring"
    for c in comments:
        while len(c.lines) > 1 and not c.lines[-1][1].strip():
            c.lines.pop()
        while len(c.lines) > 1 and not c.lines[0][1].strip():
            c.lines.pop(0)
        lo, hi = c.line_start - 1 - _ERROR_MARGIN, c.line_end - 1 + _ERROR_MARGIN
        c.near_error = any(lo <= r <= hi for r in err_rows)
    pervasive = len(err_rows) > _PERVASIVE_ERRORS * max(1, src.count("\n") + 1)
    return comments, pervasive


def _go_doc_candidates(comments: list[Comment], raw: list[dict]):
    """Pair each comment with the raw record of its last line (for next-sibling info)."""
    by_row = {r["r0"] + 1: r for r in raw}
    for c in comments:
        r = by_row.get(c.line_end)
        if r is not None:
            yield c, r


def extract_comments(src: str, lang: str = "python") -> tuple[list[Comment], bool]:
    """Return (comments sorted by line, unreliable). `unreliable` means the whole file can't be
    trusted (Python: tokenize/ast failed; others: pervasive parse errors). Comments next to a
    local tree-sitter error carry `near_error`."""
    if lang == "python":
        return _extract_python(src)
    return _extract_treesitter(src, lang)


# ---------------------------------------------------------------- filtering

_DIRECTIVE_RE = DIRECTIVE_RE  # all languages (Python's directives included)
# separators between a directive and the author's rationale, e.g. "noqa: BLE001 - reason"
_RATIONALE_SEP_RE = re.compile(r"\s+(?:-{1,2}|—|–|#|;|because\b)\s*|\s+\(|:\s+(?=[A-Za-z]+\s+[A-Za-z]+)")
_ENCODING_RE = re.compile(r"^-\*-.*coding[:=].*-\*-$|^(vim?:.*)?coding[:=]\s*[-\w.]+", re.I)
_LICENSE_RE = re.compile(
    r"copyright\b|\(c\)\s*\d{4}|spdx-license-identifier|licensed under|apache license|mit license"
    r"|all rights reserved|gnu (lesser |affero )?general public license|permission is hereby granted",
    re.I,
)
_SEP_RE = re.compile(r"^[\W_]*$")
_CODE_HINT_RE = re.compile(r"[()\[\]{}=]|^\s*(import|from|return|raise|del|assert|yield|await|print|pass|break|continue)\b"
                           r"|^\s*(if|elif|else|for|while|with|try|except|finally|def|class|async)\b.*:\s*$"
                           r"|^\s*@\w")
_ENGLISH_START_RE = re.compile(r"^[A-Z][a-z]+ [a-z]")  # "Set in auto_context()"
_CODE_PUNCT = set("{}[]()\"'=:,|<>")

_DECOR_CH = r"-=#*~_+─━═╌┄┈▬"
_DECOR_RE = re.compile(rf"^\s*([{_DECOR_CH}]{{3,}}|[{_DECOR_CH}]{{2,}}\s)|(\s[{_DECOR_CH}]{{2,}}|[{_DECOR_CH}]{{3,}})\s*$")
# docstring headers whose indented body is code/commands, not prose
_CODE_SECTION_RE = re.compile(r"^\s*(usage|examples?|cli|run|commands?|invocation|install(ation)?|setup|output|input|"
                              r"format|schema|sample|e\.g\.)\b[^:]*:\s*$", re.I)


_CODE_END = (";", "{", "}", "};", "});", ");", "),", "*/", "=>", "->")
_ASSIGN_RE = re.compile(r"^[\w$.\[\]'\"]+\s*(?::=|\+=|-=|\*=|/=|\|\|=|&&=|\?\?=|=(?!=))\s*\S")
_CALL_RE = re.compile(r"^(?:await\s+|return\s+)?[\w$]+(?:(?:\.|\?\.|::|->)[\w$]+)*(?:<[^>]*>)?\(.*\)\s*;?$")
_KEYWORD_CODE_RE = re.compile(rf"^(?:{CODE_KEYWORDS})\b.*(?:[=({{;\[]|:=|=>|->)")
_PREPROC_RE = re.compile(r"^(?:#include\b|#define\b|#if(?:n?def)?\b|#endif\b|#pragma\b|import\s+[\w{*\"'.]|"
                         r"from\s+\S+\s+import\b|package\s+[\w.]+;?$|use\s+[\w:{}, ]+;$)")


def _is_code_line_generic(s: str) -> bool:
    """Non-Python commented-out code: structural cues, guarded against prose."""
    no_str = re.sub(r"\"[^\"]*\"|'[^']*'|`[^`]*`", " ", s)
    ends_code = s.endswith(_CODE_END) or s in ("{", "}", "};", ")", "]")
    if not ends_code and (re.search(r"\b[A-Za-z]{2,} [A-Za-z]{2,} [A-Za-z]{2,}\b", no_str) or _ENGLISH_START_RE.match(s)):
        return False  # reads as prose
    if ends_code and re.search(r"[(){}=;\[\]]", s):
        return True
    if _ASSIGN_RE.match(s) or _PREPROC_RE.match(s) or _KEYWORD_CODE_RE.match(s):
        return True
    if _CALL_RE.match(s) and not re.search(r"\w \(", s):  # "Compute it (cached)" is prose
        return True
    nonspace = [ch for ch in no_str if not ch.isspace()]
    return len(nonspace) > 8 and sum(ch in _CODE_PUNCT for ch in nonspace) / len(nonspace) > 0.25


def is_code_line(text: str, lang: str = "python") -> bool:
    """Heuristic: does this comment line look like commented-out code (or data)?"""
    s = _DECOR_RE.sub(" ", text).strip()  # "=== Section ===" -> "Section"
    if lang != "python":
        return bool(s) and _is_code_line_generic(s)
    if not s or not _CODE_HINT_RE.search(s):
        return False  # bare words like "TODO" or "Example" parse but are not code
    if re.match(r"^(else|try|finally)\s*:$", s):
        return True
    candidates = [s, s + " pass"] if s.endswith(":") else [s]  # "if x > 0:" needs a body
    for c in candidates:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                body = ast.parse(c).body
        except (SyntaxError, ValueError, RecursionError, MemoryError):
            continue
        if not body:
            return False
        if len(body) == 1:
            st = body[0]
            # "Note: something" parses as an annotation; treat as prose
            if isinstance(st, ast.AnnAssign) and st.value is None:
                return False
            if isinstance(st, ast.Expr):
                if isinstance(st.value, (ast.Name, ast.Constant)):
                    return False
                if _ENGLISH_START_RE.match(s):
                    return False
        return True
    # unparseable but symbol-dense, e.g. a JSON/dict schema line inside a docstring,
    # unless it reads as prose (two adjacent plain words outside string literals)
    if re.search(r"\b[A-Za-z]{2,} [A-Za-z]{2,}\b", re.sub(r"\"[^\"]*\"|'[^']*'", " ", s)):
        return False
    nonspace = [ch for ch in s if not ch.isspace()]
    return len(nonspace) > 8 and sum(ch in _CODE_PUNCT for ch in nonspace) / len(nonspace) > 0.25


def directive_rationale(t: str) -> str | None:
    """For a directive line, the prose rationale after it (or None if there is none)."""
    m = _DIRECTIVE_RE.match(t)
    if not m:
        return None
    rest = t[m.end():]
    sep = _RATIONALE_SEP_RE.search(rest)
    if not sep:
        return None
    why = rest[sep.end():].strip(" )")
    return why if len(re.findall(r"[A-Za-z]{2,}", why)) >= 2 else None


def is_shebang_or_encoding(ln: int, t: str) -> bool:
    s = t.strip()
    return ln <= 2 and (s.startswith("!") or bool(_ENCODING_RE.search(s)))


def drop_reason(c: Comment, lang: str = "python") -> str | None:
    """Reason to drop the whole comment as non-prose, or None to keep it."""
    text = c.text
    if not text.strip():
        return "empty"
    if c.kind != "docstring":
        if all(is_shebang_or_encoding(ln, t) or not t.strip() for ln, t in c.lines):
            return "shebang_encoding"
        if all((_DIRECTIVE_RE.match(t) and directive_rationale(t) is None) or not t.strip()
               or is_shebang_or_encoding(ln, t) for ln, t in c.lines):
            return "directive"
        if all(_SEP_RE.match(t) for _, t in c.lines):
            return "separator"
    if _LICENSE_RE.search(text) and (c.line_start <= 30 or "spdx" in text.lower()):
        return "license"
    if c.kind != "docstring":
        nonempty = [t for _, t in c.lines if t.strip() and not _SEP_RE.match(t) and not _DIRECTIVE_RE.match(t)]
        if nonempty:
            code = sum(is_code_line(t, lang) for t in nonempty)
            if code * 2 >= len(nonempty) and code > 0:
                return "commented_out_code"
            if code:
                return "mixed_code_and_prose"  # sentence boundaries around code are unreliable
    return None


_DOC_TAG_RE = re.compile(r"^\s*@(\w+)")
_NAMED_TAGS = {"param", "arg", "argument", "property", "prop", "typedef", "template", "typeparam", "field",
               "throws", "exception", "callback", "member", "event"}
_XML_DOC_LANGS = {"csharp", "vb", "fsharp"}
_MD_HEADING_RE = re.compile(r"^\s*#{1,6}\s+\S")
_MARK_RE = re.compile(r"^\s*MARK:\s*-?\s*(\S.*)$")


def _doc_tag_prose(s: str) -> str | None:
    """'@param {T} name - desc' -> desc (None when the tag carries no prose)."""
    m = _DOC_TAG_RE.match(s)
    rest = re.sub(r"^\s*\{[^}]*\}", "", s[m.end():])
    if m.group(1).lower() in _NAMED_TAGS:
        rest = re.sub(r"^\s*\[?[\w$.\-]+(?:=[^\]]*)?\]?", "", rest)
    rest = re.sub(r"^\s*[-–:]\s*", "", rest)
    return rest if len(re.findall(r"[A-Za-z]{2,}", rest)) >= 2 else None


def _xml_doc(t: str) -> tuple[str, bool]:
    """Strip C#-style XML doc tags; also report whether the line starts a new section."""
    new_section = bool(re.match(r"^\s*<(param|typeparam|returns|remarks|exception|example|value)\b", t))
    t = re.sub(r'<(?:see|seealso|paramref|typeparamref)\s+(?:cref|name|langword|href)="([^"]+)"\s*/>', r"\1", t)
    t = re.sub(r'<(?:param|typeparam)\s+name="([^"]+)">', r"\1: ", t)
    t = re.sub(r"</?\w+[^>]*>", "", t)
    return t, new_section


def prose_lines(c: Comment, lang: str = "python") -> tuple[list[tuple[int, str]], Counter]:
    """Lines of a kept comment that carry prose, plus counts of removed lines by reason."""
    removed: Counter = Counter()
    out: list[tuple[int, str]] = []
    if c.kind == "docstring":
        in_doctest = in_fence = in_literal = in_example = False
        literal_indent = 0
        for ln, t in c.lines:
            if lang in _XML_DOC_LANGS and "<" in t:
                t, new_section = _xml_doc(t)
                if new_section:
                    out.append((ln, ""))
                if not t.strip():
                    continue
            s = t.strip()
            indent = len(t) - len(t.lstrip())
            if lang != "python" and not in_fence and _DOC_TAG_RE.match(s):
                # JSDoc / Javadoc / KDoc tags start a new paragraph
                in_example = s.lower().startswith("@example")
                why = None if in_example else _doc_tag_prose(s)
                out.append((ln, ""))
                if why is None:
                    removed["doc_tag_without_prose"] += 1
                else:
                    out.append((ln, re.sub(r"\{[^}]*\}\s*", "", t)))
                continue
            if in_example and not s.startswith("```"):
                removed["docstring_code_block"] += 1
                continue
            if lang != "python" and not in_fence and _MD_HEADING_RE.match(s):
                removed["doc_heading"] += 1
                out.append((ln, ""))
                continue
            if s.startswith("```"):
                in_fence = not in_fence
                removed["docstring_code_block"] += 1
            elif in_fence:
                removed["docstring_code_block"] += 1
            elif s.startswith(">>>") or (in_doctest and s):
                in_doctest = True
                removed["doctest"] += 1
            elif in_literal and (not s or indent > literal_indent):
                if s:
                    removed["docstring_code_block"] += 1
            else:
                in_doctest = in_literal = False
                if s.endswith("::"):
                    in_literal, literal_indent = True, indent
                    t = t.rstrip()[:-1]  # "Example::" -> "Example:"
                elif _CODE_SECTION_RE.match(s):
                    in_literal, literal_indent = True, indent  # "Usage:" + indented commands
                if indent >= 4 and s and is_code_line(s, lang):
                    removed["docstring_code_line"] += 1
                    out.append((ln, ""))
                else:
                    out.append((ln, t))
            if not s and in_doctest:
                in_doctest = False
    else:
        for ln, t in c.lines:
            if is_shebang_or_encoding(ln, t):
                removed["shebang_encoding_line"] += 1
            elif _SEP_RE.match(t):
                if t.strip():
                    removed["separator_line"] += 1
                out.append((ln, ""))  # keep as paragraph break
            elif _DIRECTIVE_RE.match(t):
                why = directive_rationale(t)
                if why:
                    removed["directive_kept_rationale"] += 1
                    out.append((ln, why))
                else:
                    removed["directive_line"] += 1
            elif is_code_line(t, lang):
                removed["code_line"] += 1
                out.append((ln, ""))
            elif _MARK_RE.match(t):
                # Swift "MARK: - Section": a heading
                out += [(ln, ""), (ln, _MARK_RE.match(t).group(1)), (ln, "")]
            elif _DECOR_RE.search(t):
                # "--- Section ---" is a heading: its own paragraph
                out += [(ln, ""), (ln, _DECOR_RE.sub(" ", t).strip()), (ln, "")]
            else:
                out.append((ln, t))
    return out, removed


# ---------------------------------------------------------------- diffing

def _changed_lines(before: str, after: str) -> set[int]:
    a, b = before.splitlines(), after.splitlines()
    changed: set[int] = set()
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    for op, _i1, _i2, j1, j2 in sm.get_opcodes():
        if op != "equal":
            changed.update(range(j1 + 1, j2 + 1))
    return changed


def new_comments(after: list[Comment], before: list[Comment] | None,
                 before_src: str | None = None, after_src: str | None = None
                 ) -> tuple[list[Comment], int]:
    """Comments of `after` whose normalized text is not in `before` (multiset).

    With a before-state, a new comment must also overlap lines the write changed;
    candidates that don't (e.g. a moved duplicate) are not attributable and are
    counted in the second return value instead.
    """
    if before is None:
        return list(after), 0
    remaining = Counter(norm_text(c.text) for c in before)
    groups: dict[str, list[Comment]] = {}
    for c in after:
        groups.setdefault(norm_text(c.text), []).append(c)
    changed = _changed_lines(before_src or "", after_src or "")
    picked, unsure = [], 0
    for key, cs in groups.items():
        n_new = len(cs) - remaining.get(key, 0)
        if n_new <= 0:
            continue
        touching = [c for c in cs if any(ln in changed for ln, _ in c.lines)]
        if len(touching) == n_new:
            picked.extend(touching)
        else:
            unsure += n_new  # can't tell which occurrence is the new one
    return sorted(picked, key=lambda c: c.line_start), unsure
