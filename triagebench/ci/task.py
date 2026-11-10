"""Adapting the CI failure corpus to the Layer-1 harness.

The point of this module is that it is thin. A CI failure corpus is loaded into
exactly the `Project` shape that `triagebench.data` produces for bug reports,
so `build_task`, every retriever, the verifiers and all the metrics run over it
unchanged. The interesting question then becomes a measurement rather than an
argument: does a stack tuned on prose bug reports still work when the input is
a raw stack trace?

The duplicate label is different in kind, and stronger. For bug reports, two
reports are duplicates because a human noticed and linked them -- which misses
every pair nobody linked. Here, two failure records are duplicates when they
are failures of the same `file.py::Class::test`. Nothing is missed, and the
label does not depend on anyone's diligence.

The cost of that strength is a narrower notion of "same problem": one root
cause breaking three different tests shows up as three distinct groups. So
recall here is measured against a *stricter* target than Layer 1's, and the two
numbers should not be read as directly comparable.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import pandas as pd

from ..data import Project

DEFAULT_CI_ROOT = Path("data/ci")

_MASK = "<TEST>"


def mask_identity(text: str, test_id: str) -> str:
    """Remove the grouping label from the text a retriever is allowed to see.

    Two failures are labelled duplicates because they share a `test_id`. If
    that same `test_id` is in the retrieval text, the task collapses into exact
    string matching and every retriever scores a perfect Recall@1 -- a
    measurement of the label, not of the system.

    So the identity is masked: the test id line, the class and test names, and
    the `in <test_name>` frames inside the traceback. The file path and line
    numbers are *kept*, because they are evidence a real triage system legibly
    has, rather than the label itself.
    """
    if not test_id:
        return text

    path, _, rest = test_id.partition("::")
    names = [n for n in rest.split("::") if n]
    out = text.replace(test_id, _MASK)

    for n in sorted(names, key=len, reverse=True):
        base = n.split("[")[0]            # drop pytest parametrisation
        for token in (n, base):
            if len(token) > 3:
                out = re.sub(rf"\b{re.escape(token)}\b", _MASK, out)
    return out


def load_ci_project(path: Path | str, name: str | None = None,
                    *, max_pairs_per_test: int = 50,
                    mask_test_identity: bool = False) -> Project:
    """Load `failures.jsonl` as a `Project` the Layer-1 code can consume.

    Args:
        path: the corpus directory or the jsonl file itself.
        name: project name; defaults to the corpus directory name.
        max_pairs_per_test: cap on pairs generated per test. A test failing k
            times yields k*(k-1)/2 pairs, so one pathologically flaky test can
            otherwise dominate the entire evaluation set.
        mask_test_identity: hide the test id from the retrieval text. False
            reproduces what a deployed system sees (and is close to trivial);
            True measures whether the failure evidence alone is enough.
    """
    path = Path(path)
    jsonl = path if path.suffix == ".jsonl" else path / "failures.jsonl"
    if not jsonl.exists():
        raise FileNotFoundError(f"no CI corpus at {jsonl}")

    records = [json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()]
    if not records:
        raise ValueError(f"{jsonl} is empty")

    rows = []
    for i, r in enumerate(records):
        test_id = r["test_id"]
        exc, msg = r.get("exception") or "", r.get("message") or ""
        excerpt = r.get("excerpt") or ""
        summary = f"{test_id} {exc} {msg}"
        if mask_test_identity:
            summary = mask_identity(f"{exc} {msg}", test_id)
            excerpt = mask_identity(excerpt, test_id)
        rows.append({
            "issue_id": i,
            "summary": summary.strip(),
            "description": excerpt,
            "created": r.get("timestamp"),
            "test_id": r["test_id"],
            "exception": r.get("exception"),
            "flagged_flaky": bool(r.get("flagged_flaky")),
            "job_id": r.get("job_id"),
            "run_id": r.get("run_id"),
        })

    df = pd.DataFrame(rows)
    df["created"] = pd.to_datetime(df["created"], errors="coerce", format="mixed", utc=True)
    df["text"] = (df["summary"].str.strip() + "\n\n"
                  + df["description"].str.strip().str.slice(0, 4000)).str.strip()
    df = df.dropna(subset=["created"]).set_index("issue_id", drop=False).sort_index()

    by_test: dict[str, list[int]] = defaultdict(list)
    for rid, test_id in zip(df.index, df["test_id"]):
        by_test[test_id].append(int(rid))

    pairs: list[tuple[int, int]] = []
    for ids in by_test.values():
        if len(ids) < 2:
            continue
        combos = list(combinations(sorted(ids), 2))[:max_pairs_per_test]
        pairs.extend(combos)

    return Project(
        name=name or (path.name if path.is_dir() else path.parent.name),
        reports=df,
        pairs=sorted(set(pairs)),
        raw_pair_rows=len(pairs),
        dropped_unknown_id=0,
    )


def corpus_stats(project: Project) -> dict:
    """Shape of a CI corpus: what the benchmark is actually running on."""
    df = project.reports
    by_test = df.groupby("test_id").size()
    return {
        "records": len(df),
        "distinct_tests": int(by_test.size),
        "tests_with_repeats": int((by_test > 1).sum()),
        "duplicate_pairs": len(project.pairs),
        "flagged_flaky": int(df["flagged_flaky"].sum()),
        "not_flagged": int((~df["flagged_flaky"]).sum()),
        "distinct_jobs": int(df["job_id"].nunique()),
        "top_exceptions": df["exception"].value_counts().head(5).to_dict(),
        "span": [str(df["created"].min()), str(df["created"].max())],
    }
