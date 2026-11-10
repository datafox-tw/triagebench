"""Same retriever, same data, two evaluation protocols.

Protocol A -- reference (GitBugs duplicate_detection.ipynb):
    index the whole corpus, treat every directed row of pairs.csv as a query,
    retrieve top-k excluding self, count a hit if the gold id is present.

Protocol B -- TriageBench:
    one query per duplicate link (the later-filed report), retrieve top-k from
    reports filed strictly before it, count a hit if any gold master is present.

The gap between them is not a modelling result. It is the share of the
reference number that comes from retrieving reports that did not exist when
the query was filed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from triagebench.audit import directed_links                      # noqa: E402
from triagebench.data import available_projects, load_project     # noqa: E402
from triagebench.evaluate import retrieve_all, score_retrieval    # noqa: E402
from triagebench.retrieval import TfidfRetriever                  # noqa: E402
from triagebench.task import build_task                           # noqa: E402


def protocol_a(project, retriever, top_k: int) -> dict:
    """Reference protocol: whole-corpus retrieval over every directed row.

    Split by direction, because the two halves measure different things:

    `forward`   gold was filed before the query -- a decision a real system
                could have made, but scored against a corpus that also
                contains the future.
    `backward`  gold was filed after the query -- the system is credited for
                retrieving a report that did not exist yet.
    """
    created = project.reports["created"]
    known = set(project.reports.index)

    stats = {"forward": [0, 0], "backward": [0, 0]}  # [hits, total]

    for q, gold in sorted(directed_links(project.name)):
        if q not in known or gold not in known:
            continue
        tq, tg = created.get(q), created.get(gold)
        if pd.isna(tq) or pd.isna(tg) or tq == tg:
            continue

        direction = "backward" if tg > tq else "forward"
        ranked = [d for d, _ in retriever.search_untimed(
            project.reports.at[q, "text"], exclude_id=q, top_k=top_k)]
        stats[direction][1] += 1
        if gold in ranked:
            stats[direction][0] += 1

    hits = stats["forward"][0] + stats["backward"][0]
    total = stats["forward"][1] + stats["backward"][1]
    return {
        "recall": hits / total if total else 0.0,
        "queries": total,
        "forward_recall": stats["forward"][0] / stats["forward"][1] if stats["forward"][1] else 0.0,
        "forward_queries": stats["forward"][1],
        "backward_recall": stats["backward"][0] / stats["backward"][1] if stats["backward"][1] else 0.0,
        "backward_queries": stats["backward"][1],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("projects", nargs="*")
    ap.add_argument("--root", type=Path, default=Path("data/raw"))
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--min-pool", type=int, default=100)
    ap.add_argument("--out", type=Path, default=Path("results/protocols.json"))
    args = ap.parse_args()

    targets = args.projects or available_projects(args.root)
    rows = []

    print("=" * 100)
    print(f"PROTOCOL COMPARISON  (TF-IDF, Recall@{args.top_k}, identical index)")
    print("=" * 100)
    print(f"{'project':<13}{'A: reference':>14}{'A fwd':>9}{'A bwd':>9}"
          f"{'B: time-aware':>16}{'B-A':>9}{'A queries':>12}{'B queries':>12}")
    print("-" * 100)

    for name in targets:
        project = load_project(name, args.root)
        retriever = TfidfRetriever().index(project.reports)

        a = protocol_a(project, retriever, args.top_k)

        task = build_task(project, new_ratio=0.0, min_pool=args.min_pool)
        if not task.queries:
            continue
        cand = retrieve_all(task, retriever, top_k=args.top_k)
        b = score_retrieval(task, cand, ks=(args.top_k,))
        b_recall = b.recall_at[args.top_k]

        print(f"{name:<13}{a['recall']:>14.3f}{a['forward_recall']:>9.3f}"
              f"{a['backward_recall']:>9.3f}{b_recall:>16.3f}"
              f"{b_recall - a['recall']:>+9.3f}{a['queries']:>12,}"
              f"{b.n_duplicate_queries:>12,}")

        rows.append({
            "project": name,
            "top_k": args.top_k,
            "reference": a,
            "timeaware_recall": b_recall,
            "timeaware_queries": b.n_duplicate_queries,
            "gap_b_minus_a": b_recall - a["recall"],
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2))
    print("-" * 100)
    print("A fwd = reference queries whose gold is older (a decision a real system could make)")
    print("A bwd = reference queries whose gold is NEWER than the query (unretrievable in production)")
    print("B     = one query per link, candidates restricted to reports filed before it")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
