"""File extension -> language, and per-language comment conventions.

Data/prose formats (Markdown, JSON, YAML, TOML, LaTeX, XML, plain text) are out of scope.
"""

from __future__ import annotations

import re

EXT_LANG = {
    "py": "python", "pyi": "python",
    "go": "go",
    "ts": "typescript", "mts": "typescript", "cts": "typescript", "tsx": "tsx",
    "js": "javascript", "mjs": "javascript", "cjs": "javascript", "jsx": "javascript",
    "rs": "rust", "swift": "swift", "kt": "kotlin", "kts": "kotlin", "java": "java",
    "rb": "ruby", "cs": "csharp", "php": "php",
    "c": "c", "h": "cpp", "cc": "cpp", "cpp": "cpp", "cxx": "cpp", "hpp": "cpp", "hh": "cpp", "hxx": "cpp",
    "m": "objc", "mm": "objc",
    "scala": "scala", "sc": "scala", "lua": "lua", "sh": "bash", "bash": "bash", "zsh": "bash",
    "dart": "dart", "ex": "elixir", "exs": "elixir", "erl": "erlang", "hrl": "erlang", "hs": "haskell",
    "ml": "ocaml", "mli": "ocaml", "clj": "clojure", "cljs": "clojure", "r": "r", "jl": "julia",
    "sql": "sql", "vue": "vue", "svelte": "svelte", "zig": "zig", "nim": "nim", "pl": "perl", "pm": "perl",
    "ps1": "powershell", "gd": "gdscript", "sol": "solidity", "proto": "proto", "tf": "hcl", "hcl": "hcl",
    "groovy": "groovy", "gradle": "groovy", "fs": "fsharp", "fsx": "fsharp", "vb": "vb",
    "html": "html", "htm": "html", "css": "css", "scss": "scss",
}

# grammars that host embedded <script> blocks
EMBEDDED_SCRIPT = {"vue", "svelte", "html"}

_EXT_RE = re.compile(r"\.([A-Za-z0-9]+)$")


def language_of(path: str | None) -> str | None:
    m = _EXT_RE.search(path or "")
    return EXT_LANG.get(m.group(1).lower()) if m else None


def is_code_path(path: str | None) -> bool:
    return language_of(path) is not None


# comment delimiters, tried longest first: (open, close or None for line comments, is_doc)
DELIMS = [
    ("/**", "*/", True), ("/*!", "*/", True), ("/*", "*/", False),
    ("///", None, True), ("//!", None, True), ("//", None, False),
    ("<!--", "-->", False), ("--[[", "]]", False), ("{-", "-}", False), ("(*", "*)", False),
    ("<#", "#>", False), ("=begin", "=end", False), ("##", None, False), ("#", None, False),
    ("---", None, False), ("--", None, False), (";;", None, False), (";", None, False),
    ("%%", None, False), ("%", None, False), ("'", None, False), ("REM ", None, False),
]
# languages whose "#"/"--"/";"/"%"/"'" prefixes are comments (others never use them)
LINE_PREFIXES = {
    "#": {"python", "ruby", "bash", "perl", "r", "elixir", "php", "nim", "powershell", "gdscript", "hcl", "julia"},
    "##": {"nim", "gdscript"},
    "--": {"lua", "sql", "haskell"}, "---": {"lua", "haskell"},
    ";": {"clojure"}, ";;": {"clojure"}, "%": {"erlang"}, "%%": {"erlang"}, "'": {"vb"}, "REM ": {"vb"},
}

# directive / pragma comments (tool instructions, not prose), across languages
DIRECTIVE_RE = re.compile(
    r"^\s*(?:"
    # python
    r"noqa\b|type:\s*ignore\b|pragma\b|pylint\s*:|fmt\s*:\s*(?:on|off|skip)\b|isort\s*:|mypy\s*:|pyright\s*:"
    r"|ruff\s*:|nosec\b|flake8\s*:|yapf\s*:|autopep8\s*:|pyre-|pytype\s*:|type:\s*\S+$"
    # js/ts
    r"|eslint(?:-disable|-enable|-env)?\b|@ts-(?:ignore|expect-error|nocheck|check)\b|prettier-ignore\b"
    r"|istanbul\s+ignore\b|c8\s+ignore\b|v8\s+ignore\b|jshint\b|jscs:|tslint:|biome-ignore\b|deno-lint-ignore"
    r"|deno-fmt-ignore|oxlint-|@jsx\b|@jsxImportSource\b|@flow\b|@refresh\b"
    r"|webpackChunkName|@vite-ignore|<reference\s|<amd-module|#?(?:end)?region\b(?:\s+\S+)?\s*$"
    # go
    r"|go:\w+|\+build\b|nolint\b|lint:ignore\b|export\s+\w+$"
    # rust / c / c++ / objc
    r"|rustfmt::|clippy::|NOLINT(?:NEXTLINE|BEGIN|END)?\b|clang-format\s+(?:on|off)\b|clang-tidy\b|cppcheck-suppress"
    r"|MARK:\s*-?\s*$|swiftlint:|swift-format-ignore|sourcery:"
    # jvm / c#
    r"|ktlint\b|noinspection\b|CHECKSTYLE:|NOSONAR\b|@formatter:|ReSharper\s+(?:disable|restore)\b|language=\w+$"
    # ruby / php / shell / other
    r"|rubocop:|frozen_string_literal:|warn_indent:|shareable_constant_value:|typed:\s*\w+$"
    r"|phpcs:|@phpstan-|@psalm-|phpstan:|@codingStandardsIgnore|shellcheck\s|@formatter|vim:|-\*-"
    r")",
    re.I,
)

# keywords that start a code line in most C-like / scripting languages
CODE_KEYWORDS = (
    r"return|if|else|for|foreach|while|switch|case|break|continue|throw|try|catch|finally|import|export|from|"
    r"package|const|let|var|val|func|fn|function|def|class|struct|enum|impl|trait|interface|type|pub|public|"
    r"private|protected|static|async|await|yield|defer|go|match|use|mod|namespace|using|include|require|new|"
    r"delete|echo|print|println|printf|fmt|console|self|this|super|end|local|elif|fi|then|do|done|esac"
)


# throwaway scripts outside any repository (system temp directories)
_SCRATCH_RE = re.compile(r"^(?:/tmp/|/private/tmp/|/var/tmp/|/private/var/tmp/|/var/folders/|/private/var/folders/|/dev/shm/)"
                         r"|^[A-Za-z]:[\\/](?:Windows[\\/])?Temp[\\/]|[\\/]AppData[\\/]Local[\\/]Temp[\\/]", re.I)


def is_scratch_path(path: str | None) -> bool:
    return bool(_SCRATCH_RE.search(path or ""))
