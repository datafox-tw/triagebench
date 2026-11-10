"""Parsing GitHub Actions job logs into individual test-failure records.

A job log is one long file; a failing job usually contains *several* failing
tests. The unit this benchmark needs is one failure, not one job, so the log is
split into per-test blocks.

pytest's output gives two usable structures:

    =================================== FAILURES ===================================
    ____________________ TestClass.test_name ____________________
    Traceback (most recent call last):
      File ".../test_x.py", line 337, in test_name
        ...
    AssertionError: Tensor-likes are not close!

and the short summary line, which carries the fully qualified identity:

    FAILED test/inductor/test_x.py::TestClass::test_name - AssertionError: ...

The qualified id is the label this benchmark groups on: two failures of the
same `file::Class::test` are failures of the same underlying problem, which is
a far stronger notion of "duplicate" than a human-linked bug report pair.

Log text is untrusted input. It is only ever parsed as text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Iterator

# Actions prefixes every line with an ISO-8601 timestamp.
_TS = re.compile(r"^(?P<ts>\d{4}-\d{2}-\d{2}T[\d:.]+Z)\s?(?P<rest>.*)$")

# Any "===== ... =====" banner. pytest pads banners to the terminal width, so
# these also mark where a FAILURES/RERUNS block *ends* -- a single job log can
# contain a dozen pytest sessions back to back.
_BANNER_EQ = re.compile(r"^={3,}.*?={3,}$")
_SECTION = re.compile(r"^={3,}\s*(FAILURES|RERUNS|ERRORS)\s*={3,}\s*$")

# Per-test banner inside a section: "_____ TestClass.test_name _____".
# Two things make this awkward. The name is itself full of underscores
# (`test_matmul_output_precision`), so the delimiters are found by the
# whitespace around the name rather than by the name being underscore-free.
# And because pytest pads to a fixed width, a long test name leaves as little
# as a single underscore on each side:
#   __ TestBasicGEMMXPU.test__int8_mm_m_32_k_32_n_48_compile_True_slice_False_xpu __
# Requiring three underscores silently drops exactly the longest, most
# parameterised test names.
_TEST_BANNER = re.compile(r"^_+\s+(?P<name>\S.*?\S|\S)\s+_+$")

# Short-summary line carrying the fully qualified id.
_SUMMARY = re.compile(
    r"^(?P<status>FAILED|ERROR)\s+(?P<path>[\w./\-]+\.py)::(?P<rest>[\w:.\[\]\-]+)"
    r"(?:\s+-\s+(?P<msg>.*))?$"
)

# Progress line: "path::Class::test FAILED [12.3s] [ 56%]"
_PROGRESS = re.compile(
    r"^(?P<path>[\w./\-]+\.py)::(?P<rest>[\w:.\[\]\-]+)\s+"
    r"(?:\('(?P<rerun>RERUN)'[^)]*\)|(?P<status>FAILED|ERROR))\b"
)

_EXC = re.compile(
    r"^(?P<exc>[A-Za-z_][\w.]*(?:Error|Exception|Failure|Warning))\b\s*:?\s*(?P<msg>.*)$"
)

_BANNER_NOISE = re.compile(r"^[=_\-\s]*$")


@dataclass
class TestFailure:
    """One failing test, extracted from one job log."""

    test_id: str               # file.py::Class::test  -- the grouping label
    test_name: str             # Class.test_name as the banner spells it
    file_path: str | None
    exception: str | None      # e.g. AssertionError
    message: str               # first line of the exception message
    excerpt: str               # traceback block, timestamps stripped
    section: str               # FAILURES | RERUNS | ERRORS
    job_id: int | None = None
    run_id: int | None = None
    repo: str | None = None
    timestamp: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def text(self) -> str:
        """Retrieval text: identity line plus the traceback."""
        head = f"{self.test_id}\n{self.exception or ''}: {self.message}".strip()
        return f"{head}\n\n{self.excerpt}".strip()


def strip_timestamps(log: str) -> list[tuple[str | None, str]]:
    """Split a raw log into (timestamp, content) pairs."""
    out: list[tuple[str | None, str]] = []
    for line in log.splitlines():
        line = line.lstrip("﻿")
        m = _TS.match(line)
        out.append((m.group("ts"), m.group("rest")) if m else (None, line))
    return out


def _qualified_ids(lines: list[tuple[str | None, str]]) -> dict[str, str]:
    """Map `Class.test_name` -> `file.py::Class::test_name` from summary lines.

    The per-test banner only gives `Class.test_name`; the file path lives in the
    summary and progress lines. Joining them is what makes the id stable across
    jobs that run different shards of the same suite.
    """
    mapping: dict[str, str] = {}
    for _, content in lines:
        for pat in (_SUMMARY, _PROGRESS):
            m = pat.match(content.strip())
            if not m:
                continue
            path, rest = m.group("path"), m.group("rest")
            parts = [p for p in rest.split("::") if p]
            if not parts:
                continue
            short = ".".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
            mapping.setdefault(short, f"{path}::{rest}")
    return mapping


def _summary_messages(lines: list[tuple[str | None, str]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for _, content in lines:
        m = _SUMMARY.match(content.strip())
        if m and m.group("msg"):
            parts = [p for p in m.group("rest").split("::") if p]
            short = ".".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
            out.setdefault(short, m.group("msg").strip())
    return out


def iter_failures(log: str, *, max_excerpt_lines: int = 60) -> Iterator[TestFailure]:
    """Yield one TestFailure per failing test found in a job log."""
    lines = strip_timestamps(log)
    qualified = _qualified_ids(lines)
    messages = _summary_messages(lines)

    section: str | None = None
    current: str | None = None
    buf: list[str] = []
    first_ts: str | None = None

    def flush() -> Iterator[TestFailure]:
        if current is None:
            return
        body = [ln for ln in buf if not _BANNER_NOISE.match(ln)]
        if not body:
            return

        exception = message = None
        for ln in reversed(body):
            m = _EXC.match(ln.strip())
            if m:
                exception = m.group("exc")
                message = m.group("msg").strip()
                break

        test_id = qualified.get(current, current)
        yield TestFailure(
            test_id=test_id,
            test_name=current,
            file_path=test_id.split("::")[0] if "::" in test_id else None,
            exception=exception,
            message=message or messages.get(current, ""),
            excerpt="\n".join(body[:max_excerpt_lines]).strip(),
            section=section or "FAILURES",
            timestamp=first_ts,
        )

    for ts, content in lines:
        stripped = content.strip()

        # Every '====' banner is a boundary: it either opens a failure section
        # or closes the one in progress (short test summary, session start,
        # pass/fail totals). Without this, a block's excerpt runs on into the
        # next pytest session in the same log.
        if _BANNER_EQ.match(stripped):
            yield from flush()
            current, buf = None, []
            m = _SECTION.match(stripped)
            section = m.group(1) if m else None
            continue

        if section is None:
            continue

        m = _TEST_BANNER.match(stripped)
        if m:
            yield from flush()
            current, buf, first_ts = m.group("name").strip(), [], ts
            continue

        if current is not None:
            buf.append(content.rstrip())

    yield from flush()


def extract_failures(log: str, *, repo: str | None = None,
                     run_id: int | None = None,
                     job_id: int | None = None,
                     dedupe: bool = True) -> list[TestFailure]:
    """All failures in one job log, tagged with their provenance.

    pytest reports a flaky test once per RERUN plus once in FAILURES, so the
    same test appears several times in one log. Those are retries of a single
    event, not independent failures; keeping them would inflate any duplicate
    count. `dedupe` keeps the first occurrence per test id.
    """
    out: list[TestFailure] = []
    seen: set[str] = set()
    for f in iter_failures(log):
        if dedupe and f.test_id in seen:
            continue
        seen.add(f.test_id)
        f.repo, f.run_id, f.job_id = repo, run_id, job_id
        out.append(f)
    return out
