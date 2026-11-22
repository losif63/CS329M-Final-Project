"""Surface markers of session context: a regex proxy for browsing and sampling, not a label."""

import re

MARKER_RE = re.compile("|".join([
    r"\b(now|no longer|previously|used to|instead of|originally|anymore|changed (to|from)|updated to|switched (to|from)"
    r"|the old|the new (approach|logic|version|behavior|implementation)|unlike before|this (fix|change)"
    r"|(was|were) (removed|replaced|changed))\b",
    r"\b(as requested|per (the )?(user|your|request)|requested by|the user (asked|wants|requested)|you asked"
    r"|as discussed|as mentioned|we (discussed|agreed|decided)|per our|as (you|we) )\b",
    r"(/Users/|/home/\w|on (my|this) machine|\blocally\b|in this (environment|session|run)|the (test )?run (above|earlier)"
    r"|output above|tested (it|this|locally)|verified (it|this|that))",
    r"\b(fix(es|ed)? (the|a|an) (bug|issue|error|crash|problem)|bug fix|workaround for the|to avoid the (bug|issue|error))\b",
]), re.I)
