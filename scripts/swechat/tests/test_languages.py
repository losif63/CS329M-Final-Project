import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from comments import drop_reason, extract_comments, is_code_line, prose_lines  # noqa: E402
from languages import is_code_path, language_of  # noqa: E402
from sentences import split_comment  # noqa: E402


def kinds(src, lang):
    return [(c.kind, c.line_start, c.line_end, c.text) for c in extract_comments(src, lang)[0]]


def sents(c, lang):
    return [s for s, _, _ in split_comment(prose_lines(c, lang)[0])]


def test_language_of_and_exclusions():
    assert language_of("/a/b/x.tsx") == "tsx" and language_of("x.GO") == "go"
    for p in ["README.md", "a.json", "b.yaml", "c.yml", "d.toml", "e.tex", "f.xml", "g.txt"]:
        assert not is_code_path(p)


def test_go_doc_block_inline_and_strings():
    src = ('// F does z.\n// More detail.\nfunc F() int { return 1 } // trailing note\n\n'
           '/*\n * Block comment\n * second line.\n */\nvar s = "// not a comment"\n')
    assert kinds(src, "go") == [
        ("docstring", 1, 2, "F does z.\nMore detail."),
        ("inline", 3, 3, "trailing note"),
        ("block", 6, 7, "Block comment\nsecond line."),
    ]


def test_typescript_jsdoc_tags():
    src = ("/**\n * Fetch the user.\n * @param id - the user id to look up\n * @returns {Promise<User>}\n */\n"
           "export async function getUser(id: string) {}\n")
    (c,) = extract_comments(src, "typescript")[0]
    assert c.kind == "docstring" and (c.line_start, c.line_end) == (2, 4)
    assert sents(c, "typescript") == ["Fetch the user.", "@param id - the user id to look up"]


def test_rust_inner_and_outer_doc_not_merged():
    src = "//! Crate docs.\n/// Adds one.\n/// Second line.\nfn f(x: i32) -> i32 { x + 1 } // tail\n"
    assert [k[:3] for k in kinds(src, "rust")] == [("docstring", 1, 1), ("docstring", 2, 3), ("inline", 4, 4)]


def test_rust_doc_code_fence_and_heading_removed():
    src = "/// Adds one.\n///\n/// # Examples\n///\n/// ```\n/// let y = f(1);\n/// ```\nfn f() {}\n"
    (c,) = extract_comments(src, "rust")[0]
    assert sents(c, "rust") == ["Adds one."]


def test_vue_script_and_template_comments():
    src = ('<template>\n  <!-- header slot -->\n  <div/>\n</template>\n<script lang="ts">\n'
           "// Load the store lazily.\nconst x = 1 // tail\n</script>\n")
    assert [k[:3] for k in kinds(src, "vue")] == [("block", 2, 2), ("block", 6, 6), ("inline", 7, 7)]


def test_csharp_xml_doc():
    src = '/// <summary>Loads the config.</summary>\n/// <param name="path">Where to read it from.</param>\nvoid Load(string path) {}\n'
    (c,) = extract_comments(src, "csharp")[0]
    assert sents(c, "csharp") == ["Loads the config.", "path: Where to read it from."]


def test_generic_commented_out_code():
    for code in ["const x = useState(0);", "fetchUser(id)", "if (err) return;", "}", "x := compute(a, b)",
                 "#include <stdio.h>", "import { foo } from './foo'"]:
        assert is_code_line(code, "typescript"), code
    for prose in ["Retry because the API is flaky.", "TODO: fix this later", "Compute it (cached)",
                  "return the value to the caller", "See foo.bar() for details"]:
        assert not is_code_line(prose, "typescript"), prose


def test_js_directives_and_shebang():
    one = lambda src, lang="javascript": extract_comments(src, lang)[0][0]  # noqa: E731
    assert drop_reason(one("// eslint-disable-next-line no-console\n"), "javascript") == "directive"
    assert drop_reason(one("x(); // @ts-ignore\n", "typescript"), "typescript") == "directive"
    assert drop_reason(one("//go:build linux\n", "go"), "go") == "directive"
    assert drop_reason(one("#!/usr/bin/env bash\n", "bash"), "bash") == "shebang_encoding"
    assert drop_reason(one("// Retry because the API is flaky.\n"), "javascript") is None


def test_swift_mark_heading():
    (c,) = extract_comments("// MARK: - Networking\nfunc f() {}\n", "swift")[0]
    assert sents(c, "swift") == ["Networking"]


def test_rust_path_protected_in_split():
    (c,) = extract_comments("// Uses std::fs::read() here. Then exits.\nfn f() {}\n", "rust")[0]
    assert sents(c, "rust") == ["Uses std::fs::read() here.", "Then exits."]


def test_local_parse_error_flags_only_nearby_comments():
    src = ("// Far away comment about caching.\nconst a = 1;\n\n\n\n\n"
           "// Near the error.\nconst actual = await importOriginal<typeof import('../x')>();\nconst b = 2;\n")
    comments, unreliable = extract_comments(src, "tsx")
    assert not unreliable
    assert [(c.text, c.near_error) for c in comments] == [("Far away comment about caching.", False), ("Near the error.", True)]


def test_unpunctuated_lines_split_but_wrapped_sentences_join():
    lines = [(3, "Stateful Docker API mock for testing"), (4, "Simulates Docker operations without real Docker")]
    assert [s for s, _, _ in split_comment(lines)] == ["Stateful Docker API mock for testing",
                                                      "Simulates Docker operations without real Docker"]
    wrapped = [(1, "Create the composite index using the"), (2, "PostgreSQL btree_gist extension.")]
    assert [s for s, _, _ in split_comment(wrapped)] == ["Create the composite index using the PostgreSQL btree_gist extension."]


def test_scratch_paths():
    from languages import is_scratch_path
    for p in ["/tmp/verify-css.mjs", "/private/tmp/claude-501/x/a.py", "/var/folders/ab/T/x.ts",
              r"C:\Users\me\AppData\Local\Temp\x.js"]:
        assert is_scratch_path(p), p
    for p in ["/Users/me/proj/src/a.go", "/home/u/.claude/worktrees/x/a.ts", "/workspace/tmp/a.py"]:
        assert not is_scratch_path(p), p
