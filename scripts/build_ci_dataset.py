"""Build a corpus of real CI test failures from a repository's flaky-test issues.

Pipeline:
  1. list bot-filed `DISABLED <test>` issues (each cites one CI job);
  2. download each cited job log once, even when several issues cite it;
  3. split every log into per-test failure records;
  4. label each record by whether its test is one the bot has flagged flaky.

A job log holds many failing tests, not just the one its issue is about, so the
corpus contains both flagged and unflagged failures from the same logs. That is
what makes it usable as a benchmark rather than a list of known-flaky tests:
the unflagged failures are the contrast class, drawn from identical sources and
formats.

Two records of the same `file.py::Class::test` are failures of one underlying
problem. That is the duplicate label -- derived from test identity, not from a
human remembering to link two tickets.

Output: JSONL at data/ci/<repo>/failures.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from triagebench.ci.extract import extract_failures     # noqa: E402
from triagebench.ci.github import GitHub                # noqa: E402


#: A job with one of these outcomes will never yield more than it already has,
#: so re-downloading it is pure waste. Anything else (network error, rate
#: limit) may succeed on a later run and is left eligible.
SETTLED_STATUSES = frozenset({"ok", "no_failures", "expired"})


def jobs_to_skip(ledger: dict[str, str]) -> set[int]:
    """Job ids whose outcome is settled and should not be fetched again."""
    return {int(j) for j, status in ledger.items() if status in SETTLED_STATUSES}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default="pytorch/pytorch")
    ap.add_argument("--issues", type=int, default=300,
                    help="how many DISABLED issues to pull")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-jobs", type=int, default=120,
                    help="cap log downloads per run; keeps a scheduled run "
                         "bounded in time and bandwidth")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    out_dir = args.out or Path("data/ci") / args.repo.replace("/", "__")
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = out_dir / "failures.jsonl"
    issues_path = out_dir / "disabled_issues.json"

    gh = GitHub()
    if not gh.token:
        print("no GitHub token found (set GITHUB_TOKEN or run `gh auth login`); "
              "unauthenticated access is 60 req/hour", file=sys.stderr)
        return 1

    print(f"listing DISABLED issues in {args.repo} ...")
    issues = gh.disabled_tests(args.repo, limit=args.issues)
    with_job = [d for d in issues if d.has_job]
    print(f"  {len(issues)} issues parsed, {len(with_job)} cite a CI job")
    if not with_job:
        print("  nothing to fetch")
        return 1

    issues_path.write_text(json.dumps([d.__dict__ for d in issues], indent=2))

    # Actions log retention is short and configurable per repository, and in
    # practice only the last few weeks of pytorch logs are retrievable. A
    # one-shot build is therefore permanently small; the corpus has to
    # accumulate across runs, so previously collected records are kept and
    # already-downloaded jobs are not fetched again.
    existing: list[dict] = []
    if corpus_path.exists():
        existing = [json.loads(ln) for ln in corpus_path.read_text().splitlines()
                    if ln.strip()]
    # Jobs that produced records are implicit in the corpus, but jobs that
    # downloaded and yielded nothing are not -- and without a record of them
    # they are re-downloaded on every scheduled run, forever, crowding out new
    # work under --max-jobs. The ledger tracks every job already attempted.
    ledger_path = out_dir / "collected_jobs.json"
    ledger: dict[str, str] = {}
    if ledger_path.exists():
        try:
            ledger = json.loads(ledger_path.read_text())
        except json.JSONDecodeError:
            ledger = {}

    productive = {r["job_id"] for r in existing if r.get("job_id")}
    for jid in productive:
        ledger.setdefault(str(jid), "ok")

    # Transient failures are worth another try; a settled outcome is not.
    done_jobs = jobs_to_skip(ledger)

    if existing:
        print(f"  existing corpus: {len(existing):,} records "
              f"from {len(productive):,} jobs (will be kept)")
    if done_jobs - productive:
        print(f"  {len(done_jobs - productive):,} jobs previously tried and "
              f"settled (expired or no parseable failures) -- skipping")

    # Several issues cite the same job; download each log once.
    job_to_issues: dict[int, list] = {}
    for d in with_job:
        if d.job_id in done_jobs:
            continue
        job_to_issues.setdefault(d.job_id, []).append(d)
    # Newest first: logs expire, so the most recent jobs are the ones that
    # will not be collectable on the next run.
    ordered = sorted(job_to_issues, key=lambda j: max(d.created_at for d in job_to_issues[j]),
                     reverse=True)
    capped = ordered[:args.max_jobs]
    if len(ordered) > len(capped):
        print(f"  {len(ordered)} new jobs available, capping at {args.max_jobs}")
    job_to_issues = {j: job_to_issues[j] for j in capped}
    print(f"  {len(job_to_issues)} new jobs to download "
          f"({len(done_jobs)} already collected)")

    # Tests the bot has flagged, by short name -- the form a log banner uses.
    flagged: dict[str, dict] = {}
    for d in issues:
        prev = flagged.get(d.test_name)
        if prev is None or d.created_at < prev["created_at"]:
            flagged[d.test_name] = {"issue": d.issue_number,
                                    "created_at": d.created_at,
                                    "platforms": d.platforms}

    records: list[dict] = []
    stats = Counter()
    lock = threading.Lock()

    def fetch(job_id: int) -> None:
        ds = job_to_issues[job_id]
        log, reason = gh.job_log(args.repo, job_id)
        if log is None:
            with lock:
                stats[f"log_failed:{reason}"] += 1
                # "expired" is permanent; anything else may be transient.
                ledger[str(job_id)] = "expired" if reason == "expired" else "error"
            return

        failures = extract_failures(log, repo=args.repo,
                                    run_id=ds[0].run_id, job_id=job_id)
        with lock:
            stats["logs_ok"] += 1
            stats["log_bytes"] += len(log)
            ledger[str(job_id)] = "ok" if failures else "no_failures"
            if not failures:
                stats["logs_no_failures_parsed"] += 1
            for f in failures:
                rec = f.to_dict()
                hit = flagged.get(f.test_name)
                rec["flagged_flaky"] = hit is not None
                rec["issue_number"] = hit["issue"] if hit else None
                rec["issue_created_at"] = hit["created_at"] if hit else None
                rec["cited_by"] = [d.issue_number for d in ds
                                   if d.test_name == f.test_name]
                records.append(rec)
                stats["failures"] += 1
            if stats["logs_ok"] % 25 == 0:
                print(f"    {stats['logs_ok']}/{len(job_to_issues)} logs, "
                      f"{stats['failures']} failures", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(fetch, list(job_to_issues)))

    # A scheduled collector that silently stops collecting is worse than one
    # that fails. Treat "every download was rejected" as an error, because it
    # means the token is wrong or lacks scope -- not that the logs aged out.
    auth_failed = sum(v for k, v in stats.items()
                      if k.startswith("log_failed:http-40")
                      and k.split("http-")[1] in ("401", "403"))
    if job_to_issues and stats["logs_ok"] == 0 and auth_failed:
        print(f"\nERROR: all {auth_failed} log downloads were rejected (401/403). "
              "The token is missing or lacks access to the Actions API; "
              "this is not log expiry.", file=sys.stderr)
        return 2

    # Merge, keyed by (job, test): one failure of one test in one job is the
    # unit, so re-running the collector can only add records, never duplicate.
    merged: dict[tuple, dict] = {(r.get("job_id"), r["test_id"]): r
                                 for r in existing}
    added = 0
    for r in records:
        key = (r.get("job_id"), r["test_id"])
        if key not in merged:
            merged[key] = r
            added += 1
    records = sorted(merged.values(), key=lambda r: (r.get("timestamp") or "", r["test_id"]))

    with corpus_path.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    ledger_path.write_text(json.dumps(dict(sorted(ledger.items())), indent=0))

    by_test = Counter(r["test_id"] for r in records)
    repeated = {t: n for t, n in by_test.items() if n > 1}

    print(f"\n{'=' * 78}")
    print(f"corpus: {corpus_path}")
    print(f"{'=' * 78}")
    print(f"  logs downloaded      {stats['logs_ok']:,} "
          f"({stats['log_bytes'] / 1e6:.0f} MB fetched, logs discarded)")
    for key in sorted(k for k in stats if k.startswith("log_failed:")):
        print(f"  {key[11:]:<20} {stats[key]:,}")
    print(f"  logs w/ 0 parsed     {stats['logs_no_failures_parsed']:,}")
    print(f"  failures parsed      {stats['failures']:,} this run "
          f"({added:,} new after merge)")
    print(f"  corpus total         {len(records):,} records")
    print(f"  distinct tests       {len(by_test):,}")
    print(f"  tests seen >1 time   {len(repeated):,}  "
          f"(these yield the duplicate pairs)")
    print(f"  flagged flaky        {sum(1 for r in records if r['flagged_flaky']):,}")
    print(f"  not flagged          {sum(1 for r in records if not r['flagged_flaky']):,}")
    print(f"  API requests         {gh.n_requests:,}")

    if repeated:
        print("\n  most repeated tests:")
        for t, n in Counter(repeated).most_common(5):
            print(f"    {n:>3}x  {t[:88]}")

    _write_summary(args.repo, records, added, stats, by_test, repeated, gh)
    return 0


def _write_summary(repo, records, added, stats, by_test, repeated, gh) -> None:
    """Emit a run summary where a scheduled runner will surface it.

    GitHub Actions renders $GITHUB_STEP_SUMMARY on the run page, so a daily
    collection shows its yield without anyone opening the logs.
    """
    expired = stats.get("log_failed:expired", 0)
    lines = [
        f"### CI corpus collection — `{repo}`",
        "",
        f"| metric | value |",
        f"| --- | ---: |",
        f"| new records this run | {added:,} |",
        f"| corpus total | {len(records):,} |",
        f"| distinct tests | {len(by_test):,} |",
        f"| tests seen >1 time | {len(repeated):,} |",
        f"| logs downloaded | {stats['logs_ok']:,} |",
        f"| logs expired | {expired:,} |",
        f"| API requests | {gh.n_requests:,} |",
    ]
    other = {k[11:]: v for k, v in stats.items()
             if k.startswith("log_failed:") and k != "log_failed:expired"}
    if other:
        lines.append(f"| other download failures | {other} |")

    text = "\n".join(lines) + "\n"
    target = os.environ.get("GITHUB_STEP_SUMMARY")
    if target:
        with open(target, "a") as fh:
            fh.write(text)


if __name__ == "__main__":
    raise SystemExit(main())
