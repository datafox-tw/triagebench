"""Run the ground-truth audit over every downloaded project."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from triagebench.audit import audit_project          # noqa: E402
from triagebench.data import available_projects, load_project  # noqa: E402


def pct(x: float) -> str:
    return f"{100 * x:5.1f}%"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("projects", nargs="*")
    ap.add_argument("--root", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("results/audit.json"))
    args = ap.parse_args()

    targets = args.projects or available_projects(args.root)
    if not targets:
        print(f"no projects under {args.root}; run scripts/download.py first")
        return 1

    reports = []
    for name in targets:
        proj = load_project(name, args.root)
        reports.append(audit_project(proj, args.root))
        print(f"  audited {proj!r}")

    print("\n" + "=" * 94)
    print("GROUND-TRUTH SYMMETRY   (a duplicate link listed in both directions)")
    print("=" * 94)
    print(f"{'project':<13}{'rows':>9}{'directed':>11}{'unordered':>11}"
          f"{'symmetric':>11}{'share':>9}{'unknown id':>12}")
    for r in reports:
        s = r["symmetry"]
        print(f"{r['project']:<13}{s['raw_rows']:>9,}{s['directed_links']:>11,}"
              f"{s['unordered_pairs']:>11,}{s['symmetric_links']:>11,}"
              f"{pct(s['symmetric_share']):>9}{s['dropped_unknown_id']:>12,}")

    print("\n" + "=" * 94)
    print("UPSTREAM TRAIN/TEST SPLIT LEAK   (test links already present in train)")
    print("=" * 94)
    any_split = False
    print(f"{'project':<13}{'train rows':>12}{'test rows':>11}{'train links':>13}"
          f"{'test links':>12}{'leaked':>9}{'leak share':>12}")
    for r in reports:
        s = r["split_leak"]
        if s is None:
            continue
        any_split = True
        print(f"{r['project']:<13}{s['train_rows']:>12,}{s['test_rows']:>11,}"
              f"{s['train_links']:>13,}{s['test_links']:>12,}"
              f"{s['leaked_links']:>9,}{pct(s['leak_share']):>12}")
    if not any_split:
        print("  (no upstream splits shipped for these projects)")

    print("\n" + "=" * 94)
    print("TEMPORAL VALIDITY   (gold duplicate filed AFTER the query = unretrievable)")
    print("=" * 94)
    print(f"{'project':<13}{'queries':>10}{'future gold':>14}{'share':>9}"
          f"{'median gap (d)':>17}{'no timestamp':>14}")
    for r in reports:
        t = r["temporal"]
        gap = "n/a" if t["median_gap_days"] is None else f"{t['median_gap_days']:.1f}"
        print(f"{r['project']:<13}{t['evaluable_queries']:>10,}"
              f"{t['future_gold_queries']:>14,}{pct(t['future_gold_share']):>9}"
              f"{gap:>17}{t['skipped_missing_timestamp']:>14,}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(reports, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
