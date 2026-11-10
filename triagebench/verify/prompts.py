"""Prompt profiles for the LLM verifier.

The same retrieval stack can be run at different operating points by swapping
the verifier's instructions rather than retraining or re-tuning a threshold.
That matters in practice: a team drowning in duplicate tickets wants recall, a
team where a wrong merge buries a real regression wants precision, and both are
served by one deployment.

Profiles are deliberately written as *rules about boundaries*, not as "be
careful". Vague caution moves the decision boundary unpredictably; explicit
exclusion criteria move it in a direction you can argue about.
"""

from __future__ import annotations

SHARED_TASK = """\
You are triaging a newly filed bug report against reports already in the tracker.

Decide whether the NEW REPORT describes the same underlying defect as one of
the CANDIDATES. Reports describe the same defect when fixing one would fix the
other.

Reply with exactly one line:
  DUPLICATE <candidate_id>
  NEW

No explanation, no other text."""

BALANCED = """\
Rules:
1. Same observable symptom is not sufficient. Two reports with identical error
   text but different underlying causes (a product defect vs. an environment or
   infrastructure outage) are NOT duplicates.
2. Wording, formatting and phrasing differences do not matter. Judge the defect.
3. Stack traces may drift across versions. Matching frames near the top of the
   trace outweigh exact line numbers.
4. The same operation failing on two distinct components, devices or subsystems
   is NOT a duplicate unless the reports point at one shared root cause.
5. If no candidate plausibly shares a root cause, answer NEW."""

PRECISION_FIRST = """\
Rules:
1. Answer DUPLICATE only when you are confident the two reports share one root
   cause. When genuinely torn, answer NEW.
2. A wrong merge hides a real defect behind an unrelated ticket and is far more
   costly here than filing a report that turns out to be a duplicate.
3. Same observable symptom is not sufficient; distinct components, devices or
   subsystems mean distinct defects absent clear shared causation.
4. Prefer NEW whenever the evidence is only textual similarity."""

RECALL_FIRST = """\
Rules:
1. Answer DUPLICATE when a candidate plausibly describes the same defect, even
   if some details differ.
2. A missed duplicate splits one defect across many tickets and wastes triage
   time; a wrong merge is cheap here because it is reviewed and easily undone.
3. Tolerate differences in wording, version, line numbers and minor parameters.
4. Still answer NEW when no candidate is related at all -- do not guess."""

PROFILES: dict[str, str] = {
    "balanced": BALANCED,
    "precision": PRECISION_FIRST,
    "recall": RECALL_FIRST,
}


def build_prompt(profile: str, query_text: str,
                 candidates: list[tuple[int, str]],
                 max_chars: int = 1500) -> str:
    """Assemble the verifier prompt for one decision."""
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; known: {list(PROFILES)}")

    blocks = [
        SHARED_TASK,
        "",
        PROFILES[profile],
        "",
        "NEW REPORT:",
        query_text[:max_chars].strip(),
        "",
        "CANDIDATES:",
    ]
    for doc_id, text in candidates:
        blocks.append(f"--- candidate_id: {doc_id} ---")
        blocks.append(text[:max_chars].strip())
    blocks.append("")
    blocks.append("Answer with one line: DUPLICATE <candidate_id>  or  NEW")
    return "\n".join(blocks)
