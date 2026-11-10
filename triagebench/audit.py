"""Audit of the GitBugs duplicate-detection ground truth and its splits.

This module makes no claim about the *data* being wrong. The reports and the
duplicate links are real. What it checks is whether the evaluation protocol
built on top of them measures what it appears to measure.

Three checks:

`symmetry`
    pairs.csv lists duplicate links in both directions (`a,b` and `b,a`). A row
    is therefore not an independent example. Any procedure that splits, samples
    or counts *rows* double-counts every symmetric link.

`split_leak`
    The upstream `-train` / `-test` pair files appear to be split on rows. If
    `a,b` lands in train and `b,a` in test, the test item is already in train:
    the same unordered link, same two texts. Reported as the share of test
    links recoverable from train.

`temporal_validity`
    The reference protocol indexes the whole corpus and retrieves the top-k
    most similar reports, excluding self. When the gold duplicate was filed
    *after* the query, a real triage system could not have retrieved it: the
    report did not exist yet. Measured per directed row, as shipped.

Together these say the headline Recall@10 on this benchmark does not describe
any realizable system -- but *not*, as one might assume, because it is simply
optimistic. `scripts/compare_protocols.py` measures the two errors separately
and they point in opposite directions:

* Scoring against a corpus that contains the future adds distractors, which
  pushes the reference number *down* by 5-10 points relative to the same
  retriever restricted to the past.
* Roughly half the reference queries are unretrievable in production, which
  means half the measurement is of a task nobody performs.

The errors partially cancel. That is worse than a clean bias, because the
resulting number looks plausible while being a blend of a measurable task and
an impossible one.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .data import DEFAULT_ROOT, Project, _normalise_pairs, load_raw_pairs, parse_directed_rows


@dataclass
class SymmetryReport:
    raw_rows: int
    directed_links: int
    unordered_pairs: int
    symmetric_links: int          # links present in both directions
    symmetric_share: float        # of unordered pairs
    dropped_unknown_id: int


@dataclass
class SplitLeakReport:
    train_rows: int
    test_rows: int
    train_links: int
    test_links: int
    leaked_links: int             # test links also present in train
    leak_share: float             # of test links


@dataclass
class TemporalReport:
    evaluable_queries: int        # directed rows with both timestamps known
    skipped_missing_timestamp: int
    future_gold_queries: int      # gold filed after the query
    future_gold_share: float
    median_gap_days: float | None


def directed_links(name: str, root: Path | str = DEFAULT_ROOT) -> set[tuple[int, int]]:
    """Every `(query, gold)` link exactly as pairs.csv orders it."""
    raw = pd.read_csv(Path(root) / name / "pairs.csv", low_memory=False)
    return parse_directed_rows(raw)


def audit_symmetry(project: Project, directed: set[tuple[int, int]],
                   root: Path | str = DEFAULT_ROOT) -> SymmetryReport:
    raw_rows = len(pd.read_csv(Path(root) / project.name / "pairs.csv", low_memory=False))
    unordered = {(min(a, b), max(a, b)) for a, b in directed}
    symmetric = sum(1 for a, b in unordered
                    if (a, b) in directed and (b, a) in directed)

    return SymmetryReport(
        raw_rows=raw_rows,
        directed_links=len(directed),
        unordered_pairs=len(unordered),
        symmetric_links=symmetric,
        symmetric_share=symmetric / len(unordered) if unordered else 0.0,
        dropped_unknown_id=project.dropped_unknown_id,
    )


def audit_split_leak(project: Project,
                     root: Path | str = DEFAULT_ROOT) -> SplitLeakReport | None:
    train_df = load_raw_pairs(project.name, "train", root)
    test_df = load_raw_pairs(project.name, "test", root)
    if train_df is None or test_df is None:
        return None

    known = set(project.reports.index)
    train_links, train_rows, _ = _normalise_pairs(train_df, known)
    test_links, test_rows, _ = _normalise_pairs(test_df, known)

    train_set, test_set = set(train_links), set(test_links)
    leaked = len(test_set & train_set)

    return SplitLeakReport(
        train_rows=train_rows,
        test_rows=test_rows,
        train_links=len(train_set),
        test_links=len(test_set),
        leaked_links=leaked,
        leak_share=leaked / len(test_set) if test_set else 0.0,
    )


def audit_temporal(project: Project, directed: set[tuple[int, int]]) -> TemporalReport:
    """Count directed rows whose gold answer did not exist at query time."""
    created = project.reports["created"]
    known = set(project.reports.index)

    evaluable, skipped, future = 0, 0, 0
    gaps: list[float] = []

    for q, gold in directed:
        if q not in known or gold not in known:
            continue
        tq, tg = created.get(q), created.get(gold)
        if pd.isna(tq) or pd.isna(tg):
            skipped += 1
            continue
        evaluable += 1
        if tg > tq:
            future += 1
        gaps.append(abs((tg - tq).total_seconds()) / 86400.0)

    return TemporalReport(
        evaluable_queries=evaluable,
        skipped_missing_timestamp=skipped,
        future_gold_queries=future,
        future_gold_share=future / evaluable if evaluable else 0.0,
        median_gap_days=float(pd.Series(gaps).median()) if gaps else None,
    )


def audit_project(project: Project, root: Path | str = DEFAULT_ROOT) -> dict[str, Any]:
    directed = directed_links(project.name, root)
    leak = audit_split_leak(project, root)
    return {
        "project": project.name,
        "tracker": project.tracker,
        "reports": len(project.reports),
        "symmetry": asdict(audit_symmetry(project, directed, root)),
        "split_leak": asdict(leak) if leak else None,
        "temporal": asdict(audit_temporal(project, directed)),
    }
