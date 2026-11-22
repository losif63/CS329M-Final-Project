# Slop comment rulebook (v1.2)

v1.2 clarifies when change history is slop: mentioning the code's earlier state is slop unless it is a test describing
the bug it guards against, or the history comes with a pointer a reader can follow (the session_history rule and
decision rule 2). The scope is unchanged from v1. (v1.1 phrased this as "slop only when the change is the point",
which let most history through.)

You label one **target sentence** from a code comment that an AI coding agent wrote during a session. You are given:
- the target sentence and the full comment it belongs to;
- an excerpt of the source file right after the agent wrote it, with the target lines marked `>>`;
- the conversation before that write: user and assistant messages in full, agent thinking and tool calls as one-line
  stubs (`[agent: 3 tool calls: Read …/plan.md · Bash pytest …]`), with tool outputs omitted.

## Definition

A sentence is **slop** when its meaning or truth depends on context from the authoring session (the conversation or
the agent's local machine) that a future reader of the repository does not have.

Test: imagine a competent developer who has the whole repository but never saw this session. Could they understand
the sentence and check that it is true? If they could not, because it relies on something said, done, or observed
only in the session, the sentence is slop.

Label only the target sentence. Use the rest of the comment and the code to understand it, but a slop sentence
next to it does not make the target slop.

## Slop categories

Pick the single best category for a slop sentence.

- **conversation**: refers to the chat itself or to what the session's user said, asked, wanted, or decided.
  "…we built/found it ourselves, per the conversation." "…produces the 혼용 the user reported." "As requested, …"
- **session_history**: mentions how the code used to be or how it changed ("now", "no longer", "previously", "used
  to", "the old …", "replaces …", "was removed"). A reader of the repository sees only the current code, so a
  description of a version they cannot see does not help them, even when the rest of the sentence is a good
  explanation. This holds unless one of the two exceptions under "Not slop" applies. "The fix is in transcript.go,
  which now uses MainRepoRoot() instead of WorktreeRoot()." "The old patterns used `\s*`; the replacements use
  `[ \t]*`." "Building the gallery used to spawn 124 processes." "The timing breakdown is always printed now."
  "Test 1: WITH autolog (the old broken behavior)"
- **local_environment**: states facts about the author's machine, accounts, paths, or one particular run.
  "On this machine that's Blender 5.x at /Applications/Blender.app." "1372 .co files at /home/runner/aiter/…"
  "Scott's /Users/sholodak/… against my own /Users/stsoucas/…"
- **unverifiable_result**: reports a measurement or outcome from runs in the session that the repository gives no way
  to reproduce. "measured at 3.26 MB raw, 0.46 MB brotli" "bs=4: 26 µs vs ~150 µs (5.7x faster)"
- **plan_reference**: points to a plan, phase, step, task, or story that exists only in the session or in a document
  outside the repository. "Scheduling constants (slice 2.4 expands these…)" "Story 7.9: …" "Phase 5: …"
- **other_session**: depends on session context in another way.

## Not slop

These are **not_slop**, even when they look similar:

- Explanations of what the code does, why it is designed this way, invariants, constraints, and edge cases, when they
  are self-contained. Comparing with an alternative is fine when it is stated as a design reason, not as history:
  "Uses os.path.normpath, not pathlib, because pathlib has no lexical equivalent."
- References a reader can resolve: files, functions, or docs in the repository; public issue, PR, or ticket IDs
  (#1268, ABA-232); standards, URLs, and library documentation.
- Plan or step references when the plan is a file in the repository. Look for the agent reading or writing it in the
  tool stubs (for example `Read docs/plan.md`). Numbered steps inside the same comment or function ("Step 2: …") are
  not slop either.
- "The user" meaning the user of the software being built, not the person in the chat:
  "Returns a RewindRequest if the user wants to rewind."
- Paths, values, or names that are clearly placeholders or documented examples (`/home/user/…`, `~/path/to/file`).
- Redundant, obvious, verbose, or AI-sounding comments. Those may be poor comments, but they are not slop.
- Text the agent copied from a file it read in the session. The sentence was already in the repository.
- Two kinds of history are allowed. (a) A test that describes the bug it guards against: "Regression for …: nothing
  used to swap the `<audio>` element, so it kept playing the old chapter." That is what the test is for. (b) History
  with a pointer a reader can follow: "Since the Phase C merge (docs/architecture/move-forward-to-cursor.md),
  move_forward no longer implements …", or a public issue or PR number. Outside these two cases, history is slop.

## Decision rules

1. Decide from the sentence's meaning, not from keywords. "Now" or "previously" can describe runtime behaviour
   ("the stream is now backgrounded") rather than code history.
2. For change history, first check that the sentence really talks about the code's earlier state (not runtime
   behaviour such as "the stream is now backgrounded"). If it does, it is slop unless it is a test describing the bug
   it guards against, or it carries a pointer to a document in the repository or a public issue or PR. A good
   explanation elsewhere in the same sentence does not rescue it.
3. If the session shows where the fact came from (a user message, a run's result, a file outside the repo), cite those
   turns. If the sentence is self-contained, cite nothing.
4. Use **unsure** only when the provided context cannot decide the case, for example when the conversation is cut off
   before the evidence, or when you cannot tell whether a referenced plan is in the repository. Prefer a decision.

## Output

Return JSON with:
- `label`: `slop`, `not_slop`, or `unsure`;
- `category`: one of the slop categories above, or `none` when the label is not `slop`;
- `evidence_turns`: turn numbers in the conversation that the decision rests on (may be empty);
- `rationale`: one or two sentences explaining the decision.
