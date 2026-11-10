"""Loading and normalising GitBugs projects.

Upstream ships, per project:
  bugs.csv   - the report corpus (Summary, Issue id, Created, Description, ...)
  pairs.csv  - duplicate ground truth as rows of `Issue id, Duplicate id`

Two properties of pairs.csv drive everything in this module:

1. `Duplicate id` may hold several comma-separated ids in one quoted field.
2. The file is (largely) *symmetric*: if `a,b` appears, `b,a` usually does too.

So a row is not a unit of truth; an unordered {a, b} pair is. Treating rows as
units is what makes the upstream train/test split leak -- see audit.py.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DEFAULT_ROOT = Path("data/raw")

#: Projects keyed by the tracker they came from. Column layout is identical
#: across all three, but duplicate conventions and text style are not.
TRACKER = {
    "cassandra": "jira",
    "hadoop": "jira",
    "hbase": "jira",
    "spark": "jira",
    "firefox": "bugzilla",
    "mozillacore": "bugzilla",
    "seamonkey": "bugzilla",
    "thunderbird": "bugzilla",
    "vscode": "github",
}

#: Id fields are spelled inconsistently across projects. Jira/Bugzilla ship
#: `Duplicate id` holding `123` or `"123, 456"`; vscode ships `Duplicates`
#: holding a Python list literal, `"['123', '456']"`. Pulling out integer runs
#: handles every observed form without branching on the tracker.
_IDS = re.compile(r"\d+")

#: Column in pairs.csv naming the query report, and the possible names of the
#: column holding its duplicate(s).
_QUERY_COL = "Issue id"
_GOLD_COLS = ("Duplicate id", "Duplicates")


@dataclass(frozen=True)
class Project:
    """A loaded project: the report corpus plus normalised duplicate pairs."""

    name: str
    reports: pd.DataFrame          # indexed by issue id
    pairs: list[tuple[int, int]]   # unordered, deduplicated, corpus-resident
    raw_pair_rows: int             # rows in pairs.csv, before normalisation
    dropped_unknown_id: int        # pairs referencing ids absent from corpus

    @property
    def tracker(self) -> str:
        return TRACKER.get(self.name, "unknown")

    def text(self, issue_id: int) -> str:
        return self.reports.at[issue_id, "text"]

    def created(self, issue_id: int) -> pd.Timestamp:
        return self.reports.at[issue_id, "created"]

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"Project({self.name!r}, reports={len(self.reports):,}, "
                f"pairs={len(self.pairs):,}, tracker={self.tracker})")


def _normalise_reports(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={
        "Issue id": "issue_id",
        "Summary": "summary",
        "Description": "description",
        "Created": "created",
        "Resolved": "resolved",
        "Resolution": "resolution",
        "Status": "status",
        "Priority": "priority",
    })
    keep = [c for c in ("issue_id", "summary", "description", "created",
                        "resolved", "resolution", "status", "priority")
            if c in df.columns]
    df = df[keep].copy()

    df["issue_id"] = pd.to_numeric(df["issue_id"], errors="coerce")
    df = df.dropna(subset=["issue_id"])
    df["issue_id"] = df["issue_id"].astype("int64")

    for col in ("summary", "description"):
        df[col] = df.get(col, "").fillna("").astype(str)

    # Retrieval text. Summary carries most of the duplicate signal; description
    # is appended but truncated so one essay-length report cannot dominate the
    # term statistics of a BM25 index.
    df["text"] = (df["summary"].str.strip() + "\n\n"
                  + df["description"].str.strip().str.slice(0, 4000)).str.strip()

    df["created"] = pd.to_datetime(df["created"], errors="coerce", format="mixed", utc=True)

    df = df.drop_duplicates(subset=["issue_id"], keep="first")
    return df.set_index("issue_id", drop=False).sort_index()


def parse_directed_rows(df: pd.DataFrame) -> set[tuple[int, int]]:
    """Explode a pairs table into `(query, gold)` links, as the file orders them.

    Direction is preserved: this is what the symmetry and temporal audits need.
    """
    gold_col = next((c for c in _GOLD_COLS if c in df.columns), None)
    if gold_col is None or _QUERY_COL not in df.columns:
        raise ValueError(
            f"pairs table needs {_QUERY_COL!r} and one of {_GOLD_COLS}; got {list(df.columns)}"
        )

    out: set[tuple[int, int]] = set()
    for a, b in zip(df[_QUERY_COL], df[gold_col]):
        a_ids = _IDS.findall(str(a))
        if not a_ids:
            continue
        a_id = int(a_ids[0])
        for part in _IDS.findall(str(b)):
            b_id = int(part)
            if a_id != b_id:
                out.add((a_id, b_id))
    return out


def _normalise_pairs(df: pd.DataFrame, known: set[int]) -> tuple[list[tuple[int, int]], int, int]:
    """Canonicalise a pairs table to unordered, corpus-resident links.

    Returns (unordered unique pairs, raw row count, pairs dropped for unknown id).
    """
    raw_rows = len(df)
    canonical = {(min(a, b), max(a, b)) for a, b in parse_directed_rows(df)}
    resident = {p for p in canonical if p[0] in known and p[1] in known}
    return sorted(resident), raw_rows, len(canonical) - len(resident)


def load_project(name: str, root: Path | str = DEFAULT_ROOT) -> Project:
    """Load one project from `root/<name>/{bugs,pairs}.csv`."""
    base = Path(root) / name
    reports = _normalise_reports(
        pd.read_csv(base / "bugs.csv", low_memory=False)
    )
    pairs, raw_rows, dropped = _normalise_pairs(
        pd.read_csv(base / "pairs.csv", low_memory=False),
        set(reports.index),
    )
    return Project(name=name, reports=reports, pairs=pairs,
                   raw_pair_rows=raw_rows, dropped_unknown_id=dropped)


def load_raw_pairs(name: str, split: str, root: Path | str = DEFAULT_ROOT) -> pd.DataFrame | None:
    """Load an upstream pair split (`train` / `test`) verbatim, if shipped."""
    path = Path(root) / name / f"pairs_{split}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, low_memory=False)


def available_projects(root: Path | str = DEFAULT_ROOT) -> list[str]:
    root = Path(root)
    if not root.exists():
        return []
    return sorted(d.name for d in root.iterdir()
                  if (d / "bugs.csv").exists() and (d / "pairs.csv").exists())
