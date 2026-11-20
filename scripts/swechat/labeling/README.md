# Slop-labeling pilot

Measures how much slop the SWE-chat dataset holds, how good the keyword proxy is, and whether an LLM judge agrees with
human labels, before labeling at scale.

- `rulebook.md`: the labeling standard. The LLM receives it verbatim as its instructions, and humans should label by it too.
- `rulebook_v2.md`: draft v2. It adds the agent's own working process to "session", plus a `deliberation` category
  for thinking-aloud comments. It isn't used yet: v1 is the reference until enough v1 misses justify a comparison.
- `screen.py`: cheap high-recall screen (about 40 sentences per call, comment text only) that shrinks the pool the
  full judge has to read. On the pilot it kept 27 of 28 Codex positives and all 7 hand positives while flagging about
  7% of sentences.
- `finalize.py`: combines the screen and judge results into a final label for every sentence
  (`data/swechat/db/label_<tag>.parquet`, and the `label` column of `comment.parquet`).
- `pilot.py`: sampling, prompt building, LLM labeling with the Claude or Codex CLI, a blind hand-labeling sheet, and the report.

```bash
python scripts/swechat/labeling/pilot.py sample                 # 200 random + 200 keyword-marker sentences (seed 0)
python scripts/swechat/labeling/pilot.py prompts                # data/swechat/labels/pilot_v1/prompts/<item>.md
python scripts/swechat/labeling/pilot.py label --backend codex  # or --backend claude [--model sonnet]; resumable
python scripts/swechat/labeling/pilot.py handsheet              # blind 100-item sheet for hand labels
python scripts/swechat/labeling/pilot.py report                 # prevalence, proxy precision, agreement
```

Each prompt holds the target sentence, its full comment, the code within 60 lines of the target and the whole
conversation before the write (capped at 120k characters, keeping the most recent context). Each conversation line
carries its turn number so the judge can cite evidence. The rulebook goes as the system prompt (Claude) or ahead of
the prompt (Codex). Both CLIs run with no tools, from an empty directory, with a JSON schema enforcing
`{label, category, evidence_turns, rationale}`.

**Hand labeling.** Run `python scripts/swechat/labeling/pilot.py serve` and open http://localhost:8000. For each
item it shows the target sentence inside its full comment, the full source file with the sentence highlighted, the
other files the agent wrote earlier in the session (latest version before this write), and the whole conversation
before the write. Click an agent line to list its tool calls, and click a call to see its input and output (file reads,
greps, command output). Keys: `s` slop, `n` not slop, `u` unsure, `1`–`6` category (optional), `←`/`→` move. The
"Unsure only" filter revisits open cases. Every change is written to `handsheet.csv` straight away, which you can also
edit by hand. The UI never shows the LLM labels, the sampling stratum or `handsheet_key.json`. Then run `report`.

**Estimates.** The `random` stratum is a uniform sample, so it gives unbiased prevalence. The `marker` stratum (at
most 2 per session) measures the proxy's precision. Prevalence is estimated by stratifying on the marker:
P(slop) = P(slop | marker)·P(marker) + P(slop | no marker)·P(no marker). Intervals are 95% Wilson intervals.
Agreement between you and each LLM judge is Cohen's kappa on slop vs. not_slop, excluding `unsure`.

**Cost (smoke test, 6 items).** Claude Opus: about $0.50 per item at list price, so about $200 for 400 items. Codex
(ChatGPT plan): no per-call charge, about 8 s per item.

## Full run (rulebook v1.2, Claude Opus 5.5)

```bash
python scripts/swechat/labeling/screen.py all --backend claude --workers 8                     # 162,584 -> 14,728 flagged
python scripts/swechat/labeling/pilot.py from-screen --run judge_all --screen screen_all
python scripts/swechat/labeling/pilot.py prompts --run judge_all
python scripts/swechat/labeling/pilot.py label --run judge_all --backend claude --tag v1.2 --workers 8
python scripts/swechat/labeling/finalize.py --screen screen_all --judge judge_all --tag v1.2
```

Result: 3,320 slop sentences (2.0%), 159,253 not slop, 11 unsure, across 818 sessions and 197 repositories. By
category: session_history 1,435, plan_reference 768, unverifiable_result 583, other_session 226, conversation 170,
local_environment 138. Checks against the pilot:
- on the pilot's 200 random sentences, the full pipeline gives 2.5% slop (2.0% from the pilot's direct judge);
- the same judge and rulebook, run twice on 80 pilot items, agree on 74 (κ 0.85), so borderline items can flip;
- the screen kept 34 of 36 judged positives on the pilot, so about 1 in 20 slop sentences is labeled not slop.

Slop is concentrated: the top repository holds 19% and the top 10 hold 54%. Split train/test by repository.
