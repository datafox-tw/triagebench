"""Fetch GitBugs (CC BY 4.0) into data/raw/.

Source: https://github.com/av9ash/gitbugs
Paper:  Patil, A. "GitBugs: Bug Reports for Duplicate Detection, Retrieval
        Augmented Generation, Triage, and More." arXiv:2504.09651 (2025).

Downloaded CSVs are untrusted input: they are only ever read as data
(pandas), never executed, and they live in their own directory.
"""

from __future__ import annotations

import argparse
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

RAW = "https://raw.githubusercontent.com/av9ash/gitbugs/main"

# project -> (directory in the upstream repo, corpus filename, pairs filename)
PROJECTS: dict[str, tuple[str, str, str]] = {
    "cassandra":   ("cassandra",      "cassandra_bugs.csv",      "cassandra_bugs-combined.csv"),
    "hadoop":      ("hadoop",         "hadoop_bugs.csv",         "hadoop_bugs-combined.csv"),
    "hbase":       ("hbase",          "hbase_bugs.csv",          "hbase_bugs-combined.csv"),
    "spark":       ("spark",          "spark_bugs.csv",          "spark_bugs-combined.csv"),
    "seamonkey":   ("seamonkey",      "SeaMonkey_bugs.csv",      "SeaMonkey_bugs-combined.csv"),
    "thunderbird": ("thunderbird",    "Thunderbird_bugs.csv",    "Thunderbird_bugs-combined.csv"),
    "firefox":     ("firefox",        "firefox_bugs.csv",        "firefox_bugs-combined.csv"),
    "mozillacore": ("mozillacore",    "mozillacore_bugs.csv",    "mozillacore_bugs-combined.csv"),
    "vscode":      ("ms_vscode_bugs", "vscode_bugs.csv",         "vscode_bugs-combined.csv"),
}

# Upstream also ships -train/-test pair splits for some projects. We download
# them when present so the split audit can inspect them, but TriageBench does
# not evaluate on them (see triagebench/audit.py for why).
SPLIT_SUFFIXES = ("train", "test")


def _get(url: str) -> bytes | None:
    try:
        with urllib.request.urlopen(url, timeout=120) as r:
            return r.read()
    except Exception as e:  # noqa: BLE001 - network best effort, reported to caller
        print(f"    ! {url.rsplit('/', 1)[-1]}: {e}", file=sys.stderr)
        return None


def _write(dest: Path, blob: bytes) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(blob)
    print(f"    + {dest.name} ({len(blob):,} bytes)")


def fetch(project: str, out_root: Path) -> bool:
    updir, corpus, pairs = PROJECTS[project]
    out = out_root / project
    print(f"  {project}")

    blob = _get(f"{RAW}/{updir}/{corpus}")
    if blob is None:
        # A few projects ship the corpus zipped instead of raw.
        zblob = _get(f"{RAW}/{updir}/{corpus}.zip")
        if zblob is None:
            return False
        with zipfile.ZipFile(io.BytesIO(zblob)) as zf:
            inner = next(n for n in zf.namelist() if n.endswith(".csv"))
            blob = zf.read(inner)
        print(f"    (unzipped {inner})")
    _write(out / "bugs.csv", blob)

    blob = _get(f"{RAW}/{updir}/{pairs}")
    if blob is None:
        return False
    _write(out / "pairs.csv", blob)

    stem = pairs.replace("-combined.csv", "")
    for suffix in SPLIT_SUFFIXES:
        blob = _get(f"{RAW}/{updir}/{stem}-{suffix}.csv")
        if blob is not None:
            _write(out / f"pairs_{suffix}.csv", blob)

    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("projects", nargs="*", default=None,
                    help="projects to fetch (default: all)")
    ap.add_argument("--out", type=Path, default=Path("data/raw"))
    args = ap.parse_args()

    targets = args.projects or list(PROJECTS)
    unknown = [p for p in targets if p not in PROJECTS]
    if unknown:
        ap.error(f"unknown project(s): {unknown}. known: {list(PROJECTS)}")

    print(f"GitBugs -> {args.out.resolve()}")
    ok = [p for p in targets if fetch(p, args.out)]
    print(f"\ndone: {len(ok)}/{len(targets)} projects")
    return 0 if len(ok) == len(targets) else 1


if __name__ == "__main__":
    raise SystemExit(main())
