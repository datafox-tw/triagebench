"""Invariants the benchmark's claims rest on.

These run on synthetic fixtures, not on the downloaded corpus, so they are
fast and do not depend on an upstream dataset that may change.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from triagebench.data import _normalise_pairs, _normalise_reports, parse_directed_rows
from triagebench.evaluate import (
    Candidates, score_decisions, score_retrieval, threshold_predictions,
)
from triagebench.retrieval import RRFusion, TfidfRetriever
from triagebench.task import Query, TriageTask, build_task, split_by_time
from triagebench.verify.base import Candidate, TopOneThreshold
from triagebench.verify.llm import LLMVerifier
from triagebench.verify.prompts import build_prompt


# --------------------------------------------------------------------------
# pair parsing
# --------------------------------------------------------------------------

def test_parses_jira_and_github_id_formats():
    jira = pd.DataFrame({"Issue id": [1, 2], "Duplicate id": ["10", "20, 30"]})
    assert parse_directed_rows(jira) == {(1, 10), (2, 20), (2, 30)}

    github = pd.DataFrame({"Issue id": [" 47816", 100335],
                           "Duplicates": ["['113635']", "['138833', '149170']"]})
    assert parse_directed_rows(github) == {
        (47816, 113635), (100335, 138833), (100335, 149170),
    }


def test_self_links_are_dropped():
    df = pd.DataFrame({"Issue id": [7], "Duplicate id": ["7"]})
    assert parse_directed_rows(df) == set()


def test_missing_gold_column_is_an_error():
    with pytest.raises(ValueError, match="pairs table needs"):
        parse_directed_rows(pd.DataFrame({"Issue id": [1], "Nope": [2]}))


def test_normalise_pairs_canonicalises_direction_and_filters_corpus():
    df = pd.DataFrame({"Issue id": [1, 2, 3], "Duplicate id": ["2", "1", "999"]})
    pairs, raw_rows, dropped = _normalise_pairs(df, known={1, 2, 3})
    assert pairs == [(1, 2)]        # both directions collapse to one link
    assert raw_rows == 3
    assert dropped == 1             # (3, 999) references an unknown id


# --------------------------------------------------------------------------
# task construction
# --------------------------------------------------------------------------

def _project(n: int = 300, pairs=((5, 200),)):
    reports = pd.DataFrame({
        "Issue id": range(n),
        "Summary": [f"report {i} crash in module {i % 7}" for i in range(n)],
        "Description": [f"stack trace frame {i % 13}" for i in range(n)],
        "Created": pd.date_range("2024-01-01", periods=n, freq="D").astype(str),
    })

    class P:
        name = "synthetic"
        dropped_unknown_id = 0
    p = P()
    p.reports = _normalise_reports(reports)
    p.pairs = list(pairs)
    return p


def test_query_is_the_later_report_and_gold_is_the_earlier():
    task = build_task(_project(pairs=[(5, 200)]), new_ratio=0.0, min_pool=0)
    assert len(task.queries) == 1
    q = task.queries[0]
    assert q.query_id == 200            # later-filed report asks the question
    assert q.gold_ids == frozenset({5})


def test_each_link_yields_exactly_one_query():
    task = build_task(_project(pairs=[(5, 200), (200, 5)]), new_ratio=0.0, min_pool=0)
    assert len(task.queries) == 1


def test_new_queries_are_never_linked_reports():
    proj = _project(pairs=[(5, 200), (10, 220)])
    task = build_task(proj, new_ratio=2.0, min_pool=0, seed=1)
    linked = {5, 200, 10, 220}
    for q in task.queries:
        if not q.is_duplicate:
            assert q.query_id not in linked


def test_time_split_puts_newest_queries_in_test():
    task = build_task(_project(pairs=[(5, 100), (6, 150), (7, 250)]),
                      new_ratio=0.0, min_pool=0)
    tune, test = split_by_time(task, test_fraction=0.34)
    assert max(q.cutoff for q in tune.queries) <= min(q.cutoff for q in test.queries)


# --------------------------------------------------------------------------
# time-aware retrieval
# --------------------------------------------------------------------------

def test_search_excludes_self_and_the_future():
    proj = _project(n=200)
    r = TfidfRetriever().index(proj.reports)
    cutoff = proj.reports.at[100, "created"]

    hits = r.search(proj.reports.at[100, "text"], cutoff, top_k=50)
    ids = [d for d, _ in hits]

    assert 100 not in ids, "the query itself must not be retrievable"
    assert all(proj.reports.at[d, "created"] < cutoff for d in ids)


def test_untimed_search_does_see_the_future():
    proj = _project(n=200)
    r = TfidfRetriever().index(proj.reports)
    ids = [d for d, _ in r.search_untimed(proj.reports.at[100, "text"],
                                          exclude_id=100, top_k=50)]
    assert 100 not in ids
    assert any(d > 100 for d in ids), "reference protocol retrieves later reports"


def test_earliest_report_has_no_candidates():
    proj = _project(n=50)
    r = TfidfRetriever().index(proj.reports)
    first = proj.reports["created"].idxmin()
    assert r.search(proj.reports.at[first, "text"],
                    proj.reports.at[first, "created"], top_k=10) == []


# --------------------------------------------------------------------------
# fusion
# --------------------------------------------------------------------------

class _Fake:
    """Returns a fixed ranking, ignoring the query."""

    def __init__(self, name, ids):
        self.name, self._ids = name, ids

    def index(self, reports):
        return self

    def search(self, text, cutoff, top_k=10):
        return [(i, 1.0 / r) for r, i in enumerate(self._ids[:top_k], start=1)]


def test_rrf_rewards_being_found_by_both_retrievers():
    # 1 is surfaced by both; 2 and 9 by only one each.
    a, b = _Fake("a", [1, 2]), _Fake("b", [1, 9])
    fused = RRFusion([a, b], depth=2).search("q", pd.Timestamp("2024-01-01"), top_k=3)
    assert fused[0][0] == 1
    assert fused[0][1] > fused[1][1]


def test_rrf_is_convex_so_spread_ranks_edge_out_consistent_ones():
    # Documented behaviour, not a bug: 1/61 + 1/63 > 2/62. Pinned so that a
    # change to rrf_k or the weighting scheme cannot silently invert it.
    a, b = _Fake("a", [1, 2, 3]), _Fake("b", [3, 2, 1])
    fused = dict(RRFusion([a, b], depth=3).search("q", pd.Timestamp("2024-01-01"), top_k=3))
    assert fused[1] == pytest.approx(fused[3])
    assert fused[1] > fused[2]
    assert fused[1] - fused[2] < 1e-4   # the margin is negligible in practice


def test_rrf_weights_shift_the_outcome():
    a, b = _Fake("a", [1, 9]), _Fake("b", [9, 1])
    heavy_a = RRFusion([a, b], weights=[0.99, 0.01], depth=2)
    heavy_b = RRFusion([a, b], weights=[0.01, 0.99], depth=2)
    ts = pd.Timestamp("2024-01-01")
    assert heavy_a.search("q", ts, top_k=1)[0][0] == 1
    assert heavy_b.search("q", ts, top_k=1)[0][0] == 9


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def _task_with(queries):
    return TriageTask(project="t", reports=pd.DataFrame(), queries=queries,
                      n_duplicate=sum(1 for q in queries if q.is_duplicate),
                      n_new=sum(1 for q in queries if not q.is_duplicate),
                      dropped_no_earlier=0)


def test_recall_at_k_and_mrr_use_rank_of_first_gold():
    ts = pd.Timestamp("2024-01-01")
    task = _task_with([Query(1, frozenset({50}), ts)])
    cand = Candidates("t", "fake", top_k=5)
    cand.hits = {1: [(9, 0.9), (8, 0.8), (50, 0.7)]}   # gold at rank 3

    m = score_retrieval(task, cand, ks=(1, 2, 3))
    assert m.recall_at[1] == 0.0
    assert m.recall_at[2] == 0.0
    assert m.recall_at[3] == 1.0
    assert m.mrr == pytest.approx(1 / 3)


def test_any_gold_counts_as_a_hit():
    ts = pd.Timestamp("2024-01-01")
    task = _task_with([Query(1, frozenset({50, 60}), ts)])
    cand = Candidates("t", "fake", top_k=5)
    cand.hits = {1: [(60, 0.9)]}
    assert score_retrieval(task, cand, ks=(1,)).recall_at[1] == 1.0


def test_decision_metrics_separate_calling_duplicate_from_naming_the_master():
    ts = pd.Timestamp("2024-01-01")
    task = _task_with([
        Query(1, frozenset({50}), ts),      # duplicate, master named right
        Query(2, frozenset({60}), ts),      # duplicate, wrong master
        Query(3, frozenset(), ts),          # NEW, correctly left alone
        Query(4, frozenset(), ts),          # NEW, wrongly merged
    ])
    m = score_decisions(task, {1: 50, 2: 99, 3: None, 4: 77}, system="s")

    assert m.precision == pytest.approx(2 / 3)   # 3 called dup, 2 really are
    assert m.recall == 1.0
    assert m.correct_master == pytest.approx(1 / 2)
    assert m.accuracy == pytest.approx(3 / 4)


def test_threshold_predictions_respect_the_cutoff():
    cand = Candidates("t", "fake", top_k=5)
    cand.hits = {1: [(50, 0.9)], 2: [(60, 0.1)], 3: []}
    preds = threshold_predictions(cand, threshold=0.5)
    assert preds == {1: 50, 2: None, 3: None}


# --------------------------------------------------------------------------
# verifier
# --------------------------------------------------------------------------

def test_top_one_threshold_baseline():
    v = TopOneThreshold(0.5)
    assert v.verify(1, "q", [Candidate(7, 0.9, "t")]).master_id == 7
    assert v.verify(1, "q", [Candidate(7, 0.1, "t")]).master_id is None
    assert v.verify(1, "q", []).master_id is None


@pytest.mark.parametrize("reply,allowed,expected", [
    ("DUPLICATE 123", {123}, 123),
    ("duplicate: 123", {123}, 123),
    ("NEW", {123}, None),
    ("DUPLICATE 999", {123}, None),      # hallucinated id must not become a merge
    ("", {123}, None),
    ("I think maybe", {123}, None),
])
def test_verifier_parsing_rejects_ids_never_offered(reply, allowed, expected):
    assert LLMVerifier._parse(reply, allowed)[0] == expected


def test_prompt_contains_query_candidates_and_profile_rules():
    prompt = build_prompt("precision", "the query text", [(11, "cand A"), (22, "cand B")])
    assert "the query text" in prompt
    assert "candidate_id: 11" in prompt and "cand B" in prompt
    assert "answer new" in prompt.lower()   # the precision profile's fallback rule


def test_unknown_profile_is_rejected():
    with pytest.raises(ValueError, match="unknown profile"):
        build_prompt("nope", "q", [(1, "c")])
