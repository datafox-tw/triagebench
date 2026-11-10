"""Run the Layer-1 stack over the Layer-2 CI failure corpus.

Nothing in `triagebench.retrieval`, `triagebench.task`, `triagebench.evaluate`
or `triagebench.verify` is specialised for CI logs. This script only swaps the
corpus, which is the experiment: does a retrieval-plus-verifier stack developed
on prose bug reports still work when the input is a raw stack trace?
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from triagebench.ci.task import corpus_stats, load_ci_project      # noqa: E402
from triagebench.evaluate import (                                 # noqa: E402
    DEFAULT_KS, Candidates, retrieve_all, score_decisions, score_retrieval,
    threshold_predictions, tune_threshold,
)
from triagebench.retrieval import (                                # noqa: E402
    BM25Retriever, DenseRetriever, KeywordOverlapRetriever, RRFusion, TfidfRetriever,
)
from triagebench.task import build_task, split_by_time             # noqa: E402
from triagebench.verify import Candidate, LLMVerifier              # noqa: E402


def load_env(paths: list[Path]) -> None:
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def subset(cand: Candidates, ids: set[int]) -> Candidates:
    s = Candidates(project=cand.project, retriever=cand.retriever,
                   top_k=cand.top_k, seconds=cand.seconds)
    s.hits = {q: h for q, h in cand.hits.items() if q in ids}
    return s


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, default=Path("data/ci/pytorch__pytorch"))
    ap.add_argument("--min-pool", type=int, default=5)
    ap.add_argument("--new-ratio", type=float, default=1.0)
    ap.add_argument("--test-fraction", type=float, default=0.4)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--profiles", nargs="*", default=["balanced", "precision"])
    ap.add_argument("--top-n", type=int, default=5)
    ap.add_argument("--skip-llm", action="store_true")
    ap.add_argument("--mask-identity", action="store_true",
                    help="hide the test id from retrieval text (the label must "
                         "not be a feature); without it the task is near-trivial")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if args.out is None:
        suffix = "_masked" if args.mask_identity else ""
        args.out = Path(f"results/ci_benchmark{suffix}.json")

    load_env([Path(".env"), Path("../.env"),
              Path(__file__).resolve().parents[2] / ".env"])

    project = load_ci_project(args.corpus, mask_test_identity=args.mask_identity)
    stats = corpus_stats(project)
    stats["test_identity"] = "masked" if args.mask_identity else "visible"

    print("=" * 94)
    print(f"CI corpus: {project.name}  [test identity: {stats['test_identity']}]")
    print("=" * 94)
    for k, v in stats.items():
        print(f"  {k:<22} {v}")

    task = build_task(project, new_ratio=0.0, min_pool=args.min_pool)
    print(f"\nretrieval task: {len(task.queries)} duplicate queries "
          f"(from {len(project.pairs)} duplicate pairs)")
    if not task.queries:
        print("  corpus too small for any time-aware query; collect more first")
        return 1

    rows: list[dict] = []
    dense = DenseRetriever()
    systems = {
        "tfidf": TfidfRetriever(),
        "bm25": BM25Retriever(),
        "dense": dense,
        "rrf-weighted": RRFusion([dense, BM25Retriever(), KeywordOverlapRetriever()],
                                 weights=[0.6, 0.3, 0.1], name="rrf-weighted"),
    }

    header = "".join(f"{'R@' + str(k):>10}" for k in DEFAULT_KS)
    print(f"\n{'retriever':<20}{'queries':>8}{header}{'MRR':>10}{'ms/query':>12}")
    print("-" * 94)
    for name, retriever in systems.items():
        cand = retrieve_all(task, retriever, top_k=25)
        m = score_retrieval(task, cand, ks=DEFAULT_KS)
        print(m.row(DEFAULT_KS))
        row = asdict(m)
        row["recall_at"] = {str(k): v for k, v in m.recall_at.items()}
        row["system"] = name
        row["stage"] = "retrieval"
        rows.append(row)

    # -- decision -----------------------------------------------------------
    dec_task = build_task(project, new_ratio=args.new_ratio,
                          min_pool=args.min_pool, seed=0)
    tune_task, test_task = split_by_time(dec_task, args.test_fraction)
    print(f"\ndecision task: tune {len(tune_task.queries)} / "
          f"test {len(test_task.queries)} "
          f"({test_task.n_duplicate} dup / {test_task.n_new} new)")

    out_payload = {"corpus": stats, "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)

    if not test_task.queries or test_task.n_duplicate == 0:
        print("  not enough held-out duplicates to score a decision yet")
        args.out.write_text(json.dumps(out_payload, indent=2))
        return 0

    retriever = RRFusion([dense, BM25Retriever(), KeywordOverlapRetriever()],
                         weights=[0.6, 0.3, 0.1], name="rrf-weighted")
    cand = retrieve_all(dec_task, retriever, top_k=25)
    cand_tune = subset(cand, {q.query_id for q in tune_task.queries})
    cand_test = subset(cand, {q.query_id for q in test_task.queries})

    print(f"\n{'system':<22}{'queries':>8}{'precision':>11}{'recall':>9}"
          f"{'F1':>9}{'accuracy':>10}{'master ok':>14}")
    print("-" * 94)

    thr, _ = tune_threshold(tune_task, cand_tune)
    m = score_decisions(test_task, threshold_predictions(cand_test, thr),
                        system="top1-threshold", threshold=thr)
    print(m.row())
    rows.append({**asdict(m), "verifier": "top1-threshold", "stage": "decision"})

    if not args.skip_llm:
        texts = project.reports["text"]
        items = [
            (q.query_id, texts[q.query_id],
             [Candidate(d, s, texts[d])
              for d, s in cand_test.hits.get(q.query_id, [])[:args.top_n]])
            for q in test_task.queries
        ]
        for profile in args.profiles:
            v = LLMVerifier(model=args.model, profile=profile, top_n=args.top_n)
            decisions = v.verify_many(items, workers=6)
            m = score_decisions(test_task,
                                {d.query_id: d.master_id for d in decisions},
                                system=f"llm-{profile}")
            print(m.row())
            rows.append({**asdict(m), "verifier": f"llm-{profile}",
                         "stage": "decision", "model": args.model,
                         "failures": v.n_failures})

    args.out.write_text(json.dumps(out_payload, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
