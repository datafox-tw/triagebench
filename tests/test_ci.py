"""Tests for CI log parsing.

Every fixture here is shaped after a real pytorch/pytorch job log. Two of these
cases are regressions: the short-underscore banner and the multi-session log
each silently produced zero failures in the first version of the parser, which
looks identical to "this job had no test failures".
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from triagebench.ci.extract import extract_failures, iter_failures, strip_timestamps
from triagebench.ci.github import GitHub

TS = "2026-11-04T17:51:52.2701657Z "


def log(*lines: str) -> str:
    return "\n".join(TS + ln for ln in lines)


SIMPLE = log(
    "inductor/test_x.py::TestThing::test_alpha FAILED [16.7s] [ 56%]",
    "=================================== FAILURES ===================================",
    "__________________ TestThing.test_alpha __________________",
    "Traceback (most recent call last):",
    '  File "/pytorch/test/inductor/test_x.py", line 337, in test_alpha',
    "    self.assertEqual(a, b)",
    "AssertionError: Tensor-likes are not equal!",
    "=========================== short test summary info ============================",
    "FAILED inductor/test_x.py::TestThing::test_alpha - AssertionError: Tensor-likes are not equal!",
    "==================== 1 failed, 52 passed in 5.12s =====================",
)


def test_strips_timestamps():
    pairs = strip_timestamps(SIMPLE)
    assert all(ts is not None for ts, _ in pairs)
    assert pairs[0][1].startswith("inductor/test_x.py")


def test_extracts_identity_exception_and_excerpt():
    fs = extract_failures(SIMPLE, repo="o/r", job_id=7, run_id=9)
    assert len(fs) == 1
    f = fs[0]
    assert f.test_id == "inductor/test_x.py::TestThing::test_alpha"
    assert f.test_name == "TestThing.test_alpha"
    assert f.file_path == "inductor/test_x.py"
    assert f.exception == "AssertionError"
    assert "Tensor-likes are not equal" in f.message
    assert "line 337" in f.excerpt
    assert (f.repo, f.job_id, f.run_id) == ("o/r", 7, 9)


def test_long_test_names_leave_only_two_underscores():
    """Regression: pytest pads banners to a fixed width, so a long name gets
    `__ name __`. Requiring three underscores drops the most parameterised
    tests -- exactly the ones most likely to be flaky."""
    name = "TestBasicGEMMXPU.test__int8_mm_m_32_k_32_n_48_compile_True_slice_False_xpu"
    src = log(
        "=================================== FAILURES ===================================",
        f"__ {name} __",
        "Traceback (most recent call last):",
        "RuntimeError: boom",
        "=========================== short test summary info ============================",
    )
    fs = extract_failures(src)
    assert [f.test_name for f in fs] == [name]
    assert fs[0].exception == "RuntimeError"


def test_excerpt_does_not_run_into_the_next_pytest_session():
    """Regression: one job log holds many pytest sessions back to back."""
    src = log(
        "=================================== FAILURES ===================================",
        "______ TestA.test_one ______",
        "AssertionError: first failure",
        "==================== 1 failed in 1.0s =====================",
        "============================= test session starts ==============================",
        "collected 400 items",
        "PASSED some/other_test.py::TestB::test_two",
        "==================== 400 passed in 90.0s =====================",
    )
    fs = extract_failures(src)
    assert len(fs) == 1
    assert "first failure" in fs[0].excerpt
    assert "collected 400 items" not in fs[0].excerpt
    assert "400 passed" not in fs[0].excerpt


def test_reruns_of_one_test_collapse_to_a_single_record():
    """pytest reports a flaky test once per RERUN and again under FAILURES.
    Those are retries of one event, not independent failures."""
    src = log(
        "==================================== RERUNS ====================================",
        "______ TestA.test_one ______",
        "AssertionError: attempt 1",
        "==================================== RERUNS ====================================",
        "______ TestA.test_one ______",
        "AssertionError: attempt 2",
        "=================================== FAILURES ===================================",
        "______ TestA.test_one ______",
        "AssertionError: final",
        "=========================== short test summary info ============================",
    )
    assert len(extract_failures(src, dedupe=True)) == 1
    assert len(extract_failures(src, dedupe=False)) == 3


def test_multiple_distinct_tests_are_separate_records():
    src = log(
        "=================================== FAILURES ===================================",
        "______ TestA.test_one ______",
        "AssertionError: one",
        "______ TestA.test_two ______",
        "ValueError: two",
        "=========================== short test summary info ============================",
    )
    fs = extract_failures(src)
    assert [f.test_name for f in fs] == ["TestA.test_one", "TestA.test_two"]
    assert [f.exception for f in fs] == ["AssertionError", "ValueError"]


def test_text_combines_identity_and_traceback():
    f = extract_failures(SIMPLE)[0]
    assert f.test_id in f.text
    assert "AssertionError" in f.text
    assert "line 337" in f.text


def test_log_without_failures_yields_nothing():
    src = log(
        "============================= test session starts ==============================",
        "collected 100 items",
        "==================== 100 passed in 10.0s =====================",
    )
    assert extract_failures(src) == []


def test_unqualified_banner_falls_back_to_the_short_name():
    """When no summary line carries the file path, the short name is the id."""
    src = log(
        "=================================== FAILURES ===================================",
        "______ TestOrphan.test_x ______",
        "RuntimeError: no summary line in this log",
        "=========================== short test summary info ============================",
    )
    fs = extract_failures(src)
    assert fs[0].test_id == "TestOrphan.test_x"
    assert fs[0].file_path is None


# --------------------------------------------------------------------------
# issue parsing
# --------------------------------------------------------------------------

@pytest.mark.parametrize("title,expected", [
    ("DISABLED test_alpha (__main__.TestThing)", "TestThing.test_alpha"),
    ("DISABLED test_b_xpu (module.sub.TestXPU)", "TestXPU.test_b_xpu"),
    ("DISABLED test_p[cuda] (__main__.TestP)", "TestP.test_p[cuda]"),
])
def test_parses_disabled_issue_titles(title, expected):
    item = {"number": 1, "title": title, "state": "open",
            "created_at": "2026-11-05T00:00:00Z",
            "body": "Platforms: xpu\n\n(see https://github.com/o/r/actions/runs/42/job/99)"}
    d = GitHub._parse_issue(item)
    assert d is not None
    assert d.test_name == expected
    assert (d.run_id, d.job_id) == (42, 99)
    assert d.platforms == ["xpu"]


def test_mask_identity_removes_the_label_but_keeps_the_evidence():
    from triagebench.ci.task import mask_identity
    test_id = "inductor/test_x.py::TestThing::test_alpha_beta"
    text = (f"{test_id} AssertionError\n"
            'File "/pytorch/test/inductor/test_x.py", line 337, in test_alpha_beta\n'
            "    TestThing().check()\n"
            "AssertionError: Tensor-likes are not equal!")
    out = mask_identity(text, test_id)

    assert "test_alpha_beta" not in out
    assert "TestThing" not in out
    # Evidence a real triage system legitimately has must survive.
    assert "line 337" in out
    assert "inductor/test_x.py" in out
    assert "Tensor-likes are not equal" in out


def test_mask_identity_handles_parametrised_names():
    from triagebench.ci.task import mask_identity
    test_id = "t/test_g.py::TestG::test_mm_compile_True_xpu"
    out = mask_identity(f"{test_id} failed in test_mm_compile_True_xpu", test_id)
    assert "test_mm_compile_True_xpu" not in out


def test_masked_corpus_text_does_not_contain_the_grouping_label(tmp_path):
    """The duplicate label must not be a retrievable feature."""
    from triagebench.ci.task import load_ci_project
    rows = [
        {"test_id": "a/test_x.py::TestA::test_one", "test_name": "TestA.test_one",
         "exception": "AssertionError", "message": "values differ",
         "excerpt": 'File "a/test_x.py", line 10, in test_one\nAssertionError',
         "section": "FAILURES", "timestamp": "2026-11-01T00:00:00Z",
         "job_id": 1, "run_id": 1, "flagged_flaky": True},
        {"test_id": "a/test_x.py::TestA::test_one", "test_name": "TestA.test_one",
         "exception": "AssertionError", "message": "values differ",
         "excerpt": 'File "a/test_x.py", line 10, in test_one\nAssertionError',
         "section": "FAILURES", "timestamp": "2026-11-02T00:00:00Z",
         "job_id": 2, "run_id": 2, "flagged_flaky": True},
    ]
    corpus = tmp_path / "failures.jsonl"
    corpus.write_text("\n".join(__import__("json").dumps(r) for r in rows))

    masked = load_ci_project(tmp_path, mask_test_identity=True)
    assert len(masked.pairs) == 1          # same test, two jobs -> one pair
    for text in masked.reports["text"]:
        assert "test_one" not in text
        assert "TestA" not in text

    visible = load_ci_project(tmp_path, mask_test_identity=False)
    assert any("test_one" in t for t in visible.reports["text"])


def test_ignores_issues_that_are_not_disabled_reports():
    item = {"number": 2, "title": "Segfault in autograd", "state": "open",
            "created_at": "2026-11-05T00:00:00Z", "body": ""}
    assert GitHub._parse_issue(item) is None


def test_issue_without_a_job_link_is_kept_but_flagged():
    item = {"number": 3, "title": "DISABLED test_z (__main__.TestZ)", "state": "open",
            "created_at": "2026-11-05T00:00:00Z", "body": "Platforms: linux"}
    d = GitHub._parse_issue(item)
    assert d is not None and not d.has_job


# --------------------------------------------------------------------------
# scheduled collection
# --------------------------------------------------------------------------

def test_collector_pulls_in_no_heavy_dependencies():
    """The scheduled workflow installs nothing, which only holds while the
    collection path stays standard-library only. An `import pandas` added to
    extract.py or github.py would turn a 40-second job into a multi-minute one
    (or a red build), so this is pinned."""
    import subprocess

    root = Path(__file__).resolve().parents[1]
    code = (
        "import sys; sys.path.insert(0, %r);"
        "import triagebench.ci.extract, triagebench.ci.github;"
        "print(','.join(m for m in ('pandas','numpy','torch','sklearn',"
        "'sentence_transformers','openai','rank_bm25') if m in sys.modules))"
        % str(root)
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, cwd=root)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "", f"heavy imports reached the collector: {out.stdout}"


def test_collection_workflow_is_valid_and_scheduled():
    yaml = pytest.importorskip("yaml")
    path = Path(__file__).resolve().parents[1] / ".github/workflows/collect-ci-corpus.yml"
    assert path.exists(), "the scheduled collector workflow is missing"

    spec = yaml.safe_load(path.read_text())
    # PyYAML parses a bare `on:` key as the boolean True.
    triggers = spec.get("on", spec.get(True))
    assert "schedule" in triggers, "collection must run on a schedule, not only by hand"
    assert "workflow_dispatch" in triggers, "keep a manual trigger for backfills"

    # Writing the corpus back is the whole point; without this the job is a no-op.
    assert spec.get("permissions", {}).get("contents") == "write"
    # Two concurrent collectors would race on failures.jsonl.
    assert spec.get("concurrency", {}).get("cancel-in-progress") is False

    steps = spec["jobs"]["collect"]["steps"]
    run_blocks = " ".join(s.get("run", "") for s in steps)
    assert "scripts/build_ci_dataset.py" in run_blocks
    assert "--max-jobs" in run_blocks, "an unbounded scheduled download is a trap"
