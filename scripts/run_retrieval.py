"""Benchmark candidate retrieval under the time-aware protocol."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from triagebench.data import available_projects, load_project          # noqa: E402
from triagebench.evaluate import DEFAULT_KS, retrieve_all, score_retrieval  # noqa: E402
from triagebench.retrieval import (                                     # noqa: E402
    BM25Retriever, DenseRetriever, KeywordOverlapRetriever, RRFusion, TfidfRetriever,
)
from triagebench.task import build_task                                 # noqa: E402


def build_systems(names: list[str] | None):
    """Shared retriever instances, reused inside fusions to index only once."""
    tfidf = TfidfRetriever()
    bm25 = BM25Retriever()
    keyword = KeywordOverlapRetriever()
    dense = DenseRetriever()

    systems = {
        "tfidf": tfidf,
        "bm25": bm25,
        "keyword": keyword,
        "dense": dense,
        # Weights mirroring a deployed triage stack: dense leads, lexical
        # corrects it, keyword overlap breaks ties on shared identifiers.
        "rrf-weighted": RRFusion([dense, bm25, keyword], weights=[0.6, 0.3, 0.1],
                                 name="rrf-weighted"),
        "rrf-equal": RRFusion([dense, bm25, keyword], name="rrf-equal"),
        "rrf-lexical": RRFusion([bm25, keyword], weights=[0.7, 0.3],
                                name="rrf-lexical"),
    }
    if names:
        unknown = [n for n in names if n not in systems]
        if unknown:
            raise SystemExit(f"unknown retriever(s): {unknown}. known: {list(systems)}")
        systems = {n: systems[n] for n in names}
    return systems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("projects", nargs="*")
    ap.add_argument("--retrievers", nargs="*", default=None)
    ap.add_argument("--root", type=Path, default=Path("data/raw"))
    ap.add_argument("--top-k", type=int, default=25)
    ap.add_argument("--min-pool", type=int, default=100)
    ap.add_argument("--out", type=Path, default=Path("results/retrieval.json"))
    args = ap.parse_args()

    targets = args.projects or available_projects(args.root)
    if not targets:
        print(f"no projects under {args.root}; run scripts/download.py first")
        return 1

    all_rows = []
    for name in targets:
        project = load_project(name, args.root)
        # new_ratio=0: retrieval metrics are defined on duplicate queries only,
        # so NEW queries would cost a retrieval pass and change nothing.
        task = build_task(project, new_ratio=0.0, min_pool=args.min_pool)
        print(f"\n{'=' * 96}")
        print(f"{name}  |  {len(project.reports):,} reports  |  "
              f"{len(task.queries):,} duplicate queries  "
              f"(dropped {task.dropped_no_earlier} tie-timestamp links)")
        print("=" * 96)
        if not task.queries:
            print("  no evaluable queries, skipping")
            continue

        header = ("".join(f"{'R@' + str(k):>10}" for k in DEFAULT_KS))
        print(f"{'retriever':<22}{'queries':>8}{header}{'MRR':>10}{'ms/query':>12}")
        print("-" * 96)

        for sys_name, retriever in build_systems(args.retrievers).items():
            cand = retrieve_all(task, retriever, top_k=args.top_k)
            m = score_retrieval(task, cand, ks=DEFAULT_KS)
            print(m.row(DEFAULT_KS), flush=True)
            row = asdict(m)
            row["recall_at"] = {str(k): v for k, v in m.recall_at.items()}
            row["system"] = sys_name
            row["n_reports"] = len(project.reports)
            all_rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(all_rows, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
