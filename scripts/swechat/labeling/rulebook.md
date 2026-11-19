# Slop comment rulebook (v1)

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
- **session_history**: narrates how the code changed during the session, so it only makes sense as a changelog.
  "The fix is in transcript.go, which now uses MainRepoRoot() instead of WorktreeRoot()." "The old patterns used
  `\s*`; the replacements use `[ \t]*`." "Building the gallery used to spawn 124 processes."
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

## Decision rules

1. Decide from the sentence's meaning, not from keywords. "Now" or "previously" can describe runtime behaviour
   ("the stream is now backgrounded") rather than code history.
2. If the session shows where the fact came from (a user message, a run's result, a file outside the repo), cite those
   turns. If the sentence is self-contained, cite nothing.
3. Use **unsure** only when the provided context cannot decide the case, for example when the conversation is cut off
   before the evidence, or when you cannot tell whether a referenced plan is in the repository. Prefer a decision.

## Output

Return JSON with:
- `label`: `slop`, `not_slop`, or `unsure`;
- `category`: one of the slop categories above, or `none` when the label is not `slop`;
- `evidence_turns`: turn numbers in the conversation that the decision rests on (may be empty);
- `rationale`: one or two sentences explaining the decision.
