# SWE-chat slop-comment tables

Built by `scripts/swechat/extract.py` from `data/swechat/raw/` (SALT-NLP/SWE-chat v2 at revision `202b071f`:
conversations, sessions, repositories, checkpoints, commits). Re-running overwrites every file here; this README is
copied from `scripts/swechat/SCHEMA.md`. Output is deterministic given `--limit-sessions` and `--seed`.

See `scripts/swechat/README.md` for setup and the end-to-end commands (`scripts/swechat/run_all.sh`).

## Session selection
`agent == "Claude Code"`, `kind == ""` (this excludes `imported`, `agent_review` and `agent_investigate`),
and `files_touched` contains at least one **code file** (`stats.json → session_filters`).

**Languages** (`scripts/swechat/languages.py`): Python, Go, TypeScript/TSX, JavaScript, Rust, Swift, Kotlin, Java,
Ruby, C#, PHP, C/C++, Objective-C, Scala, Lua, shell, Dart, Elixir, Erlang, Haskell, OCaml, Clojure, R, Julia, SQL,
Vue, Svelte, Zig, Nim, Perl, PowerShell, GDScript, Solidity, Protobuf, HCL, Groovy, F#, VB, HTML, CSS, SCSS.
Out of scope: prose and data formats (Markdown, JSON, YAML, TOML, LaTeX, XML, plain text).

## Tables

### turn.parquet
A trimmed copy of `conversations.parquet` for the selected sessions, minus `progress` and `file_snapshot` rows,
sorted by `(session_id, turn_number)`. `content` is not truncated. For `tool_use` rows, `content` is the tool input JSON.

| column | type | notes |
|---|---|---|
| session_id | str | |
| turn_number | int | unique within a session |
| turn_type | str | user_prompt, assistant_response, assistant_thinking, tool_use, tool_result, system_injected, summary, attachment, queue_operation, system_event, ... |
| tool_name | str | set on both tool_use and tool_result rows |
| tool_call_id | str | **not unique within a session** (some models reuse ids such as `functions.Read:12`) |
| file_path | str | |
| command | str | Bash command |
| content | str | |

### source_file.parquet
One row per code-file version whose **full text is known with certainty**, right after a successful
Write/Edit/MultiEdit. When in doubt, a version is left out instead of being approximated. Throwaway scripts in
system temp directories (`/tmp`, `/private/tmp`, `/var/folders`, Windows `Temp`) aren't repository files and are
dropped (`versions_dropped_scratch_file`).

| column | type | notes |
|---|---|---|
| id | int | primary key |
| session_id | str | |
| turn_number | int | turn of the tool_use row; the conversation for this version = turn rows with a smaller turn_number |
| repo_id | str | from sessions.parquet |
| file_path | str | absolute path as the agent used it |
| language | str | from the file extension |
| tool_name | str | Write / Edit / MultiEdit |
| content | str | full file text after the write |
| content_flag | str | `write`, `replayed`, `from_read` (session route) or `commit` (commit route only) |
| certified_by | str | `session`, `commit` or `both` (see below) |

Two independent routes certify versions.

**Session route** (`reconstruct.py`): it tracks each file's text through the session from Writes and complete Reads,
replaying Edits on top.
- `write`: content comes from the Write payload. An Edit with empty `old_string` that creates a file also counts as a write.
- `replayed`: an Edit applied to a state built only from Writes and Edits.
- `from_read`: an Edit applied to a state that started from a complete Read.

A tool_use is paired with the tool_result that has the same `tool_call_id` and is closest in turn order. The dataset
sometimes records a result just before its call, so both directions are allowed. Success is decided by whitelist:
`^(The file .+ has been (updated|overwritten)|File created successfully at|Applied N edits to)`.
- A **known failure** (`<tool_use_error>`, user rejection, `PreToolUse` hook error, permission prompt, `EACCES`, …) wrote nothing, so tracking continues.
- A missing result or an **unrecognised** message means the outcome is unknown, so the file's tracked text is discarded.

A Read counts as **complete** when:
- it has no `limit`, and its `offset` is absent, 0 or 1;
- every line is numbered consecutively from 1 (`  12→` or `12\t`, with a trailing `<system-reminder>` or `[result-id]` stripped);
- no line contains `[truncated]`;
- it returns fewer than 2000 lines.

An Edit yields no version when:
- there is no tracked state;
- `old_string` is missing from the state, or occurs more than once without `replace_all` (Claude Code would have refused that on disk);
- the result says the file "had been modified on disk".

The tracked state is **discarded** (until the next Write or complete Read) when:
- a subagent that can edit runs (any type except `Explore`, `Plan`, `claude-code-guide`);
- a Bash command could modify the file. Commands are parsed with `shlex`. Known read-only tools are exempt: `cat`, `grep`, test runners, compilers, `git diff/log/add/commit`, and linters without fix flags. Formatters, `--fix`/`--write` modes, `fmt` subcommands and repo-wide git operations discard every file unless they target explicit files. Unknown tools that name the file are assumed to write it;
- a partial Read disagrees with the state.

If drift is detected later (a Read disagrees, a replay fails, or the disk-modified note appears), every Edit version of
that file since the state was last confirmed is **retracted**.

**Commit route** (`commit_route.py`): it uses the session's linked commits (`checkpoints` → `commits`).
1. For each file in a commit, `file_attribution[path].committed_version` gives the committed text. Reverse-applying the commit's `patch` gives the pre-commit text.
2. The agent's changes from `agent_changes` (Write content, or Edit `old_string`/`new_string`) are undone newest-first from the committed text.
3. If that lands **exactly** on the pre-commit text, or on a Write's content, the commit's change to that file is fully explained by the agent's visible edits, so every intermediate version is known.
4. Each change is mapped to its tool call by content.

The commit route also refuses:
- commits whose author and commit dates differ by more than 2 minutes (rebased, amended or cherry-picked);
- commits made more than 10 minutes after the session's last turn;
- commits preceded by a base-moving git command (`pull`, `merge`, `rebase`, `cherry-pick`, `stash pop/apply`, `reset --hard`, or checking out another ref) after the first explained change. In those cases the agent edited a different base than the commit's parent.

**Cross-check.** When both routes certify a version, the texts must match; otherwise the version is dropped
(`routes_disagree`). The comparison tolerates the dataset's `REDACTED` placeholders. The session route reads the
PII-redacted conversation, while commits hold the real file, so the session text is kept to stay consistent with the
conversation. No extracted sentence contains `REDACTED`. On a 400-session sample, the cross-check exposed two blind
spots that are now fixed: rebased commits, and results recorded before their call. After the fixes, 0 of 1,148
overlapping versions disagree.

### comment.parquet
One row per newly added, agent-written comment sentence.

| column | type | notes |
|---|---|---|
| id | int | primary key |
| source_id | int | FK → source_file.id |
| sentence | str | |
| full_comment | str | the whole comment with comment markers removed (merged line comments, `/* */` body, doc comment, or dedented Python docstring) |
| comment_kind | str | inline / block / docstring |
| line_start, line_end | int | 1-based lines of the sentence in `source_file.content` |
| label | str | `slop`, `not_slop` or `unsure` after `labeling/finalize.py` (null straight out of `extract.py`) |

**Extraction.**
- Python uses `tokenize` plus `ast` docstrings.
- Every other language uses tree-sitter comment nodes, with the script blocks of Vue/Svelte/HTML parsed as TS/JS.
- Consecutive full-line comments with the same marker, in the same column, form one `block`. A trailing comment on a code line is `inline`.
- These count as `docstring`: doc comments (`/** */`, `///`, `//!`, Dart docs, Go `//` blocks directly above a declaration) and Python docstrings.

**Comments are skipped when the boundaries can't be trusted:**
- Python files that fail `tokenize`/`ast` even after `dedent`;
- files where tree-sitter parse errors cover more than 20% of the lines;
- individual comments within 2 lines of a local tree-sitter error (`near_parse_error`). Grammars have gaps, such as `importOriginal<typeof import(...)>()` in TSX;
- Writes over an existing file whose earlier content was never seen.

**Newly added.**
- A comment is new when its normalised text appears more often after the write than before, **and** it overlaps lines the write changed. Ambiguous repeated copies are skipped.
- Sentences already present in a before-state comment are removed.

**Dropped as non-prose** (`comments_dropped_by_reason`; about 200 random examples in `dropped_sample.parquet`):
- shebang and encoding lines;
- directives for many tools (`noqa`, `type: ignore`, `eslint-disable`, `@ts-ignore`, `//go:build`, `nolint`, `NOLINT`, `rubocop:`, `swiftlint:`, `#region`, …). A directive followed by a rationale keeps the rationale;
- separators and license headers;
- commented-out code. Python uses `ast`; other languages use structural cues (trailing `;{}`, assignments, calls, keyword-led statements, preprocessor lines), guarded against prose;
- mixed code-and-prose blocks;
- empty comments.

**Doc comments.**
- Removed: doctests, code fences, RST `::` blocks, the bodies of `Usage:`/`Example:` sections, JSDoc `@example` bodies, Markdown headings, and tags without prose (`@returns {T}`).
- C# XML doc tags are stripped, with `<param name="x">` becoming `x:`.
- Swift `MARK: -` lines and decorated headings (`--- Foo ---`) become their own sentence.

**Sentences** are split with pysbd.
- Protected from splitting: code spans, URLs, paths, versions, dotted names, `a::b::c()` and JSDoc `{@link …}`.
- New paragraphs start at blank lines, bullets, `name: desc` and `@tag` lines, and between consecutive lines that read as separate unpunctuated statements. Short `Header:` lines are skipped.
- Within a session, only the first `(file_path, normalised sentence)` is kept.

### label_<tag>.parquet
Written by `labeling/finalize.py`. One row per sentence: `comment_id`, `label`, `category`, `rationale`,
`evidence_turns`, `source` (`judge`, or `screen_negative` = not flagged by the screen, so counted as not slop), `model`,
`rulebook`, `pilot_item`/`pilot_stratum` (the 400 pilot sentences, to hold out of training), and
`hand_label`/`hand_category` (the 100 hand-labeled sentences). See `scripts/swechat/labeling/README.md`.

## Model inputs
`scripts/swechat/example.py: build_example(comment_id, max_total_chars=300_000, max_turn_chars=None)` returns
`(sentence, source_file_content, conversation_text)`. The conversation is the **whole session** before the write
(`turn_number < source_file.turn_number`), rendered by `conversation.py`, the same view the HTML viewer shows:
- user prompts, assistant responses and compaction summaries in full (`[user] …`, `[assistant] …`, `[summary] …`);
- each run of agent work between them collapsed into one stub, with tool results never shown:
  `[agent: thinking ×2 · 3 tool calls: Read …/cli/lifecycle.go · Grep "SessionStart" in cmd · Bash go test ./…]`.

Text turns carry the session context, because the agent reports what it found and the user replies. The stubs keep
provenance, such as whether the agent read a repository file or ran a command. Text is about 8% of a session's
characters: the median session is about 11k characters, and fewer than 1% exceed the default 300k-character cap.
Over the cap, the oldest items are dropped.
