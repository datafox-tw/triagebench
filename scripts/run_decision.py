"""Benchmark the NEW-vs-DUPLICATE decision: retrieval, then verification.

Protocol:
  1. Build a time-aware task mixing duplicate and NEW queries.
  2. Split by time -- the oldest queries tune, the newest score.
  3. Retrieve candidates once; every verifier sees the identical shortlist.
  4. Tune the top-1 score threshold on the tune split, apply it to the test
     split, and compare against each LLM prompt profile on the same split.

The threshold baseline is tuned and the LLM profiles are not, which favours the
baseline. That is deliberate: the question is whether a verifier earns its
latency and cost against a well-configured cheap system, not against a strawman.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from triagebench.data import load_project                          # noqa: E402
from triagebench.evaluate import (                                 # noqa: E402
    Candidates, retrieve_all, score_decisions, threshold_predictions, tune_threshold,
)
from triagebench.retrieval import (                                # noqa: E402
    BM25Retriever, DenseRetriever, KeywordOverlapRetriever, RRFusion,
)
from triagebench.task import build_task, split_by_time             # noqa: E402
from triagebench.verify import Candidate, LLMVerifier              # noqa: E402


def load_env(paths: list[Path]) -> None:
    """Minimal .env loader; existing environment variables win."""
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def subset(cand: Candidates, query_ids: set[int]) -> Candidates:
    sub = Candidates(project=cand.project, retriever=cand.retriever,
                     top_k=cand.top_k, seconds=cand.seconds)
    sub.hits = {q: h for q, h in cand.hits.items() if q in query_ids}
    return sub


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("project")
    ap.add_argument("--root", type=Path, default=Path("data/raw"))
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--profiles", nargs="*", default=["balanced", "precision", "recall"])
    ap.add_argument("--top-n", type=int, default=5, help="candidates shown to the verifier")
    ap.add_argument("--new-ratio", type=float, default=1.0)
    ap.add_argument("--test-fraction", type=float, default=0.3)
    ap.add_argument("--limit", type=int, default=0, help="cap test queries (smoke tests)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    load_env([Path(".env"), Path("../.env"),
              Path(__file__).resolve().parents[2] / ".env"])

    project = load_project(args.project, args.root)
    task = build_task(project, new_ratio=args.new_ratio, seed=0)
    tune_task, test_task = split_by_time(task, args.test_fraction)

    if args.limit:
        test_task.queries = test_task.queries[-args.limit:]
        test_task.n_duplicate = sum(1 for q in test_task.queries if q.is_duplicate)
        test_task.n_new = len(test_task.queries) - test_task.n_duplicate

    print(f"\n{'=' * 98}")
    print(f"{args.project}  |  {len(project.reports):,} reports")
    print(f"  tune: {len(tune_task.queries):,} queries "
          f"({tune_task.n_duplicate:,} dup / {tune_task.n_new:,} new)")
    print(f"  test: {len(test_task.queries):,} queries "
          f"({test_task.n_duplicate:,} dup / {test_task.n_new:,} new)")
    print("=" * 98)

    retriever = RRFusion(
        [DenseRetriever(), BM25Retriever(), KeywordOverlapRetriever()],
        weights=[0.6, 0.3, 0.1], name="rrf-weighted",
    )
    print(f"\nretrieving candidates with {retriever.name} ...")
    cand = retrieve_all(task, retriever, top_k=max(25, args.top_n))

    tune_ids = {q.query_id for q in tune_task.queries}
    test_ids = {q.query_id for q in test_task.queries}
    cand_tune, cand_test = subset(cand, tune_ids), subset(cand, test_ids)

    rows = []
    print(f"\n{'system':<22}{'queries':>8}{'precision':>11}{'recall':>9}"
          f"{'F1':>9}{'accuracy':>10}{'master ok':>14}{'threshold':>11}")
    print("-" * 98)

    thr, tune_f1 = tune_threshold(tune_task, cand_tune)
    m = score_decisions(test_task, threshold_predictions(cand_test, thr),
                        system="top1-threshold", threshold=thr)
    print(m.row())
    rows.append({**asdict(m), "tune_f1": tune_f1, "verifier": "top1-threshold"})

    texts = project.reports["text"]
    items = [
        (q.query_id, texts[q.query_id],
         [Candidate(d, s, texts[d]) for d, s in cand_test.hits.get(q.query_id, [])[:args.top_n]])
        for q in test_task.queries
    ]

    for profile in args.profiles:
        verifier = LLMVerifier(model=args.model, profile=profile, top_n=args.top_n,
                               base_url=args.base_url, api_key_env=args.api_key_env)
        decisions = verifier.verify_many(items, workers=args.workers,
                                         progress_every=50)
        preds = {d.query_id: d.master_id for d in decisions}
        m = score_decisions(test_task, preds, system=f"llm-{profile}")
        print(m.row())

        n = max(len(decisions), 1)
        rows.append({
            **asdict(m),
            "verifier": f"llm-{profile}",
            "model": args.model,
            "top_n": args.top_n,
            "api_calls": verifier.n_calls,
            "cache_hits": verifier.n_cache_hits,
            "failures": verifier.n_failures,
            "failure_rate": verifier.n_failures / n,
            "mean_latency_s": sum(d.latency_s for d in decisions) / n,
            "prompt_tokens": sum(d.prompt_tokens for d in decisions),
            "completion_tokens": sum(d.completion_tokens for d in decisions),
        })

    out = args.out or Path(f"results/decision_{args.project}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print("-" * 98)
    print("master ok = of the reports called DUPLICATE, the share naming a true master")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
