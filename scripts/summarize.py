"""Regenerate the README's result tables from results/*.json.

Hand-copied numbers drift from the runs that produced them. This prints the
markdown straight from the saved results so a table in the README can always
be rebuilt and diffed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def md_row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def f(x, nd=3) -> str:
    return "—" if x is None else f"{x:.{nd}f}"


def audit_tables(results: Path) -> None:
    path = results / "audit.json"
    if not path.exists():
        return
    data = json.loads(path.read_text())

    print("\n### Ground-truth symmetry\n")
    print(md_row(["project", "rows", "unordered links", "symmetric", "share"]))
    print(md_row(["---", "---:", "---:", "---:", "---:"]))
    for r in data:
        s = r["symmetry"]
        print(md_row([r["project"], f"{s['raw_rows']:,}", f"{s['unordered_pairs']:,}",
                      f"{s['symmetric_links']:,}", f"{s['symmetric_share']:.1%}"]))

    print("\n### Train/test split leak\n")
    print(md_row(["project", "train links", "test links", "leaked", "leak share"]))
    print(md_row(["---", "---:", "---:", "---:", "---:"]))
    for r in data:
        s = r["split_leak"]
        if not s:
            continue
        print(md_row([r["project"], f"{s['train_links']:,}", f"{s['test_links']:,}",
                      f"{s['leaked_links']:,}", f"**{s['leak_share']:.1%}**"]))

    print("\n### Temporal validity\n")
    print(md_row(["project", "queries", "gold filed after query", "share", "median gap"]))
    print(md_row(["---", "---:", "---:", "---:", "---:"]))
    for r in data:
        t = r["temporal"]
        gap = "—" if t["median_gap_days"] is None else f"{t['median_gap_days']:.1f} d"
        print(md_row([r["project"], f"{t['evaluable_queries']:,}",
                      f"{t['future_gold_queries']:,}",
                      f"{t['future_gold_share']:.1%}", gap]))


def protocol_table(results: Path) -> None:
    path = results / "protocols.json"
    if not path.exists():
        return
    data = json.loads(path.read_text())
    print("\n### Protocol comparison (TF-IDF, Recall@10)\n")
    print(md_row(["project", "reference", "...gold older", "...gold newer",
                  "time-aware", "Δ"]))
    print(md_row(["---", "---:", "---:", "---:", "---:", "---:"]))
    for r in data:
        a = r["reference"]
        print(md_row([r["project"], f(a["recall"]), f(a["forward_recall"]),
                      f(a["backward_recall"]), f"**{f(r['timeaware_recall'])}**",
                      f"{r['gap_b_minus_a']:+.3f}"]))


def retrieval_table(results: Path) -> None:
    rows = []
    for path in sorted(results.glob("retrieval*.json")):
        rows.extend(json.loads(path.read_text()))
    if not rows:
        return
    print("\n### Candidate retrieval (time-aware)\n")
    print(md_row(["project", "retriever", "R@1", "R@5", "R@10", "R@25", "MRR", "ms/query"]))
    print(md_row(["---", "---", "---:", "---:", "---:", "---:", "---:", "---:"]))
    for r in rows:
        ra = r["recall_at"]
        print(md_row([r["project"], r["system"], f(ra.get("1")), f(ra.get("5")),
                      f(ra.get("10")), f(ra.get("25")), f(r["mrr"]),
                      f"{r['seconds_per_query'] * 1000:.1f}"]))


def decision_tables(results: Path) -> None:
    for path in sorted(results.glob("decision_*.json")):
        rows = json.loads(path.read_text())
        if not rows:
            continue
        project = rows[0]["project"]
        n, n_dup = rows[0]["n_queries"], rows[0]["n_duplicate"]
        print(f"\n### Decision — {project} "
              f"({n} test queries, {n_dup} duplicate / {n - n_dup} new)\n")
        print(md_row(["system", "precision", "recall", "F1", "accuracy",
                      "master correct", "failures"]))
        print(md_row(["---", "---:", "---:", "---:", "---:", "---:", "---:"]))
        for r in rows:
            fails = r.get("failure_rate")
            print(md_row([r["verifier"], f(r["precision"]), f(r["recall"]),
                          f(r["f1"]), f(r["accuracy"]), f(r["correct_master"]),
                          "—" if fails is None else f"{fails:.1%}"]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=Path("results"))
    args = ap.parse_args()

    if not args.results.exists():
        print(f"no results directory at {args.results}")
        return 1

    audit_tables(args.results)
    protocol_table(args.results)
    retrieval_table(args.results)
    decision_tables(args.results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
