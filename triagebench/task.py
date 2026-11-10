"""Building a time-aware triage task from a project.

The evaluation question TriageBench asks is the one a triage system actually
faces: *a report has just been filed; is it a duplicate of something already in
the tracker, and if so, which?*

That framing fixes the three problems the audit finds:

* **Direction** is no longer arbitrary. A duplicate link {a, b} yields exactly
  one query -- the later-filed report -- with the earlier one as gold. A link
  is one example, not two, so symmetric rows stop double-counting.
* **The candidate pool is the past.** Only reports filed strictly before the
  query are retrievable, so no system is credited for finding the future.
* **NEW is representable.** Real triage sees mostly non-duplicates. Sampling
  reports that were never linked as duplicates gives the negative class the
  upstream protocol has no way to express.

Splitting, when wanted, is by time rather than at random: `split_by_time`
holds out the most recent queries, which is the only split that cannot leak a
symmetric partner across the boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .data import Project


@dataclass(frozen=True)
class Query:
    """One triage decision."""

    query_id: int
    gold_ids: frozenset[int]   # earlier reports this one duplicates; empty => NEW
    cutoff: pd.Timestamp       # only reports filed before this are retrievable

    @property
    def is_duplicate(self) -> bool:
        return bool(self.gold_ids)


@dataclass
class TriageTask:
    project: str
    reports: pd.DataFrame
    queries: list[Query]
    n_duplicate: int
    n_new: int
    dropped_no_earlier: int    # links whose two reports share a timestamp

    @property
    def duplicate_rate(self) -> float:
        return self.n_duplicate / len(self.queries) if self.queries else 0.0

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"TriageTask({self.project!r}, queries={len(self.queries):,}, "
                f"dup={self.n_duplicate:,}, new={self.n_new:,}, "
                f"dup_rate={self.duplicate_rate:.1%})")


def build_task(
    project: Project,
    *,
    new_ratio: float = 1.0,
    seed: int = 0,
    min_pool: int = 100,
) -> TriageTask:
    """Turn a project's duplicate links into time-ordered triage queries.

    Args:
        project: loaded project.
        new_ratio: how many NEW queries to sample per duplicate query. 0 gives
            a retrieval-only task; 1.0 gives a balanced decision task. Real
            trackers run far lower duplicate rates, so sweep this rather than
            trusting one value.
        seed: sampling seed for the NEW queries.
        min_pool: skip queries with fewer than this many reports before them,
            where retrieval is not yet meaningful.
    """
    created = project.reports["created"]

    gold_by_query: dict[int, set[int]] = {}
    dropped = 0

    for a, b in project.pairs:
        ta, tb = created.get(a), created.get(b)
        if pd.isna(ta) or pd.isna(tb) or ta == tb:
            dropped += 1
            continue
        query, gold = (a, b) if ta > tb else (b, a)
        gold_by_query.setdefault(query, set()).add(gold)

    order = project.reports.index.to_series()
    pool_size = _pool_sizes(created)

    dup_queries = [
        Query(query_id=q, gold_ids=frozenset(g), cutoff=created[q])
        for q, g in gold_by_query.items()
        if pool_size.get(q, 0) >= min_pool
    ]
    dup_queries.sort(key=lambda x: x.cutoff)

    linked = {i for a, b in project.pairs for i in (a, b)}
    new_queries = _sample_new(
        project, linked, dup_queries, pool_size,
        n=int(round(len(dup_queries) * new_ratio)), seed=seed, min_pool=min_pool,
    )

    queries = sorted([*dup_queries, *new_queries], key=lambda x: x.cutoff)
    del order

    return TriageTask(
        project=project.name,
        reports=project.reports,
        queries=queries,
        n_duplicate=len(dup_queries),
        n_new=len(new_queries),
        dropped_no_earlier=dropped,
    )


def _pool_sizes(created: pd.Series) -> dict[int, int]:
    """How many reports precede each report, by creation time."""
    valid = created.dropna().sort_values()
    return {int(idx): rank for rank, idx in enumerate(valid.index)}


def _sample_new(
    project: Project,
    linked: set[int],
    dup_queries: list[Query],
    pool_size: dict[int, int],
    *,
    n: int,
    seed: int,
    min_pool: int,
) -> list[Query]:
    """Sample never-linked reports as NEW queries, matched on filing time.

    Matching on time matters: duplicates cluster in busy periods, so sampling
    NEW uniformly would let a classifier separate the classes on date alone.
    """
    if n <= 0 or not dup_queries:
        return []

    created = project.reports["created"]
    eligible = [
        int(i) for i in project.reports.index
        if i not in linked
        and not pd.isna(created[i])
        and pool_size.get(int(i), 0) >= min_pool
    ]
    if not eligible:
        return []

    rng = np.random.default_rng(seed)
    elig_times = created.loc[eligible].astype("int64").to_numpy()
    sort_idx = np.argsort(elig_times)
    elig_sorted = np.asarray(eligible)[sort_idx]
    times_sorted = elig_times[sort_idx]

    # For each duplicate query, take the nearest-in-time unused eligible report.
    # Anchors may repeat when more NEW queries are wanted than there are
    # duplicates (new_ratio > 1); `used` still keeps the sampled reports
    # distinct, so a repeated anchor just yields its next-nearest neighbour.
    used: set[int] = set()
    picked: list[int] = []
    n_draw = min(n, len(eligible))
    targets = rng.choice(len(dup_queries), size=n_draw,
                         replace=n_draw > len(dup_queries))

    for t in targets:
        target_time = dup_queries[t].cutoff.value
        pos = int(np.searchsorted(times_sorted, target_time))
        for cand in _spiral(pos, len(elig_sorted)):
            rid = int(elig_sorted[cand])
            if rid not in used:
                used.add(rid)
                picked.append(rid)
                break

    return [
        Query(query_id=r, gold_ids=frozenset(), cutoff=created[r])
        for r in picked
    ]


def _spiral(start: int, size: int) -> Iterable[int]:
    """Indices near `start`, alternating outward, clipped to [0, size)."""
    for offset in range(size):
        for cand in ({start + offset, start - offset} if offset else {start}):
            if 0 <= cand < size:
                yield cand


def split_by_time(task: TriageTask, test_fraction: float = 0.2) -> tuple[TriageTask, TriageTask]:
    """Hold out the most recent queries as test.

    A time split is the only split that cannot leak a symmetric partner: both
    reports of a link are older than any later query, so a held-out link's
    counterpart cannot appear in train as the same decision.
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be in (0, 1)")

    cut = int(len(task.queries) * (1 - test_fraction))
    parts = []
    for qs in (task.queries[:cut], task.queries[cut:]):
        n_dup = sum(1 for q in qs if q.is_duplicate)
        parts.append(TriageTask(
            project=task.project,
            reports=task.reports,
            queries=qs,
            n_duplicate=n_dup,
            n_new=len(qs) - n_dup,
            dropped_no_earlier=0,
        ))
    return parts[0], parts[1]
