"""Metrics for the two decisions a triage system makes.

**Retrieval** -- given a duplicate report, is the master in the top k?
Recall@k counts a query as a hit if *any* of its gold masters is retrieved,
which is the convention in the duplicate-report literature; MRR uses the rank
of the first gold hit.

**Decision** -- is this report a duplicate at all? Retrieval metrics alone
cannot answer it, because a retriever always returns something. A system is
only useful if it can also say NEW, so decision quality is scored over both
duplicate and NEW queries as precision / recall / F1 on the DUPLICATE class.

Retrieval runs once per (project, retriever) and the candidates are cached, so
that verifiers are compared over an identical candidate set rather than each
paying for its own retrieval pass.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .task import TriageTask

DEFAULT_KS = (1, 5, 10, 25)


@dataclass
class Candidates:
    """Cached retrieval output for one (task, retriever) pair."""

    project: str
    retriever: str
    top_k: int
    hits: dict[int, list[tuple[int, float]]] = field(default_factory=dict)
    seconds: float = 0.0

    def __len__(self) -> int:
        return len(self.hits)


def retrieve_all(task: TriageTask, retriever, top_k: int = 25,
                 progress_every: int = 0) -> Candidates:
    """Run `retriever` over every query in `task`, newest-last."""
    retriever.index(task.reports)

    out = Candidates(project=task.project, retriever=retriever.name, top_k=top_k)
    start = time.perf_counter()
    for i, q in enumerate(task.queries, 1):
        text = task.reports.at[q.query_id, "text"]
        out.hits[q.query_id] = retriever.search(text, q.cutoff, top_k=top_k)
        if progress_every and i % progress_every == 0:
            rate = i / (time.perf_counter() - start)
            print(f"      {i:,}/{len(task.queries):,} queries ({rate:.1f}/s)", flush=True)
    out.seconds = time.perf_counter() - start
    return out


@dataclass
class RetrievalMetrics:
    project: str
    retriever: str
    n_duplicate_queries: int
    recall_at: dict[int, float]
    mrr: float
    seconds_per_query: float

    def row(self, ks=DEFAULT_KS) -> str:
        cells = "".join(f"{self.recall_at.get(k, float('nan')):>10.3f}" for k in ks)
        return (f"{self.retriever:<22}{self.n_duplicate_queries:>8,}{cells}"
                f"{self.mrr:>10.3f}{self.seconds_per_query * 1000:>12.1f}")


def score_retrieval(task: TriageTask, cand: Candidates,
                    ks=DEFAULT_KS) -> RetrievalMetrics:
    dup_queries = [q for q in task.queries if q.is_duplicate]
    if not dup_queries:
        raise ValueError("task has no duplicate queries to score")

    hits_at = {k: 0 for k in ks}
    rr_total = 0.0

    for q in dup_queries:
        ranked = [doc for doc, _ in cand.hits.get(q.query_id, [])]
        first = next((i for i, doc in enumerate(ranked, 1) if doc in q.gold_ids), None)
        if first is not None:
            rr_total += 1.0 / first
            for k in ks:
                if first <= k:
                    hits_at[k] += 1

    n = len(dup_queries)
    return RetrievalMetrics(
        project=task.project,
        retriever=cand.retriever,
        n_duplicate_queries=n,
        recall_at={k: hits_at[k] / n for k in ks},
        mrr=rr_total / n,
        seconds_per_query=cand.seconds / max(len(cand), 1),
    )


# --------------------------------------------------------------------------
# NEW-vs-DUPLICATE decision
# --------------------------------------------------------------------------

@dataclass
class DecisionMetrics:
    project: str
    system: str
    n_queries: int
    n_duplicate: int
    precision: float
    recall: float
    f1: float
    accuracy: float
    correct_master: float   # of predicted duplicates, share naming a gold master
    threshold: float | None = None

    def row(self) -> str:
        thr = "-" if self.threshold is None else f"{self.threshold:.4f}"
        return (f"{self.system:<22}{self.n_queries:>8,}{self.precision:>11.3f}"
                f"{self.recall:>9.3f}{self.f1:>9.3f}{self.accuracy:>10.3f}"
                f"{self.correct_master:>14.3f}{thr:>11}")


def score_decisions(task: TriageTask, predictions: dict[int, int | None],
                    system: str, threshold: float | None = None) -> DecisionMetrics:
    """Score NEW-vs-DUPLICATE calls.

    `predictions` maps query id to the predicted master id, or None for NEW.
    """
    tp = fp = fn = tn = 0
    master_right = 0

    for q in task.queries:
        pred = predictions.get(q.query_id)
        if pred is not None and q.is_duplicate:
            tp += 1
            if pred in q.gold_ids:
                master_right += 1
        elif pred is not None and not q.is_duplicate:
            fp += 1
        elif pred is None and q.is_duplicate:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return DecisionMetrics(
        project=task.project,
        system=system,
        n_queries=len(task.queries),
        n_duplicate=sum(1 for q in task.queries if q.is_duplicate),
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy=(tp + tn) / len(task.queries) if task.queries else 0.0,
        correct_master=master_right / tp if tp else 0.0,
        threshold=threshold,
    )


def threshold_predictions(cand: Candidates, threshold: float) -> dict[int, int | None]:
    """Call DUPLICATE when the top-1 score clears `threshold`."""
    out: dict[int, int | None] = {}
    for qid, hits in cand.hits.items():
        out[qid] = hits[0][0] if hits and hits[0][1] >= threshold else None
    return out


def tune_threshold(task: TriageTask, cand: Candidates,
                   n_steps: int = 200) -> tuple[float, float]:
    """Pick the top-1 score threshold maximising F1 on `task`.

    Must be run on a *train* split and applied to test; tuning and reporting on
    the same queries is exactly the kind of optimism this benchmark exists to
    remove.
    """
    scores = [h[0][1] for h in cand.hits.values() if h]
    if not scores:
        return 0.0, 0.0

    lo, hi = float(np.min(scores)), float(np.max(scores))
    best_thr, best_f1 = lo, -1.0
    for thr in np.linspace(lo, hi, n_steps):
        m = score_decisions(task, threshold_predictions(cand, float(thr)), "tune")
        if m.f1 > best_f1:
            best_thr, best_f1 = float(thr), m.f1
    return best_thr, best_f1
