# TriageBench

**A time-aware benchmark for duplicate-report triage — and an audit of the evaluation protocol the public datasets ship with.**

Duplicate detection on bug reports is usually evaluated as a retrieval problem: index the corpus, pull the top 10 most similar reports, check whether the known duplicate is in there, report Recall@10. This repo starts from the observation that the protocol does not describe a decision any system can actually make, and rebuilds the evaluation around the one it can.

Two things come out of that:

1. **An audit** of the ground truth and splits shipped with [GitBugs](https://github.com/av9ash/gitbugs), the most convenient public duplicate-report dataset. Three defects, all measurable, all reproducible from this repo.
2. **A corrected task** — given a report *as it was filed*, decide whether it duplicates something already in the tracker, and if so which. Candidates are restricted to the past, NEW is a representable answer, and a link counts once.

---

## 1. What the audit finds

Run it yourself:

```bash
python scripts/download.py && python scripts/audit_dataset.py
```

### 1.1 The ground truth is symmetric, so rows are not examples

`pairs.csv` lists a duplicate link in both directions: if `a,b` is present, `b,a` usually is too. A row is therefore not an independent example, and anything that splits, samples or counts rows double-counts.

| project | rows | unordered links | symmetric | share |
|---|---:|---:|---:|---:|
| cassandra | 292 | 164 | 132 | 80.5% |
| hadoop | 126 | 66 | 61 | 92.4% |
| hbase | 107 | 61 | 50 | 82.0% |
| spark | 496 | 269 | 231 | 85.9% |
| thunderbird | 4,288 | 3,460 | 1,975 | 57.1% |
| vscode | 9,238 | 8,645 | 3,873 | 44.8% |

### 1.2 The shipped train/test splits leak, because they split on rows

If `a,b` lands in train and `b,a` in test, the test item is already in train — same link, same two texts. Three quarters of every shipped test split is recoverable from its train split:

| project | train links | test links | leaked | **leak share** |
|---|---:|---:|---:|---:|
| cassandra | 153 | 54 | 43 | **79.6%** |
| hadoop | 61 | 24 | 19 | **79.2%** |
| hbase | 57 | 22 | 18 | **81.8%** |
| spark | 247 | 88 | 66 | **75.0%** |

Any supervised duplicate detector tuned on these splits is reporting a number it did not earn.

### 1.3 Roughly half the evaluation queries are unanswerable in production

Because the links are symmetric, every link is evaluated twice — once with the older report as the query, once with the newer. In the second case the gold answer was filed *after* the query. No triage system could retrieve it; the report did not exist yet.

| project | queries | gold filed after query | share | median gap |
|---|---:|---:|---:|---:|
| cassandra | 296 | 145 | 49.0% | 31.7 d |
| hadoop | 127 | 63 | 49.6% | 47.9 d |
| hbase | 111 | 52 | 46.8% | 53.0 d |
| spark | 500 | 247 | 49.4% | 18.9 d |
| thunderbird | 3,950 | 1,974 | 50.0% | 28.6 d |
| vscode | 605 | 267 | 44.1% | 26.5 d |

---

## 2. The protocol error does not point the way you would guess

The obvious conclusion from §1.3 — that the published numbers are inflated — is **wrong**, and this repo measures it rather than assuming it.

`scripts/compare_protocols.py` runs one TF-IDF index under both protocols and splits the reference protocol by direction:

| project | reference R@10 | ...gold older | ...gold newer | time-aware R@10 | Δ |
|---|---:|---:|---:|---:|---:|
| cassandra | 0.614 | 0.568 | 0.662 | **0.667** | +0.052 |
| hadoop | 0.740 | 0.734 | 0.746 | **0.800** | +0.060 |
| hbase | 0.509 | 0.500 | 0.519 | **0.607** | +0.098 |
| spark | 0.451 | 0.461 | 0.441 | **0.529** | +0.078 |

Two errors run in opposite directions:

- Searching a corpus that contains the future adds distractors, pushing the reference number **down** 5–10 points against the same retriever restricted to the past.
- Half the reference queries are impossible, so half the measurement is of a task nobody performs.

They partially cancel. That is worse than a clean bias: the published figure looks plausible while being a blend of a measurable task and an imaginary one. Direction itself barely changes difficulty (gold-older vs gold-newer differ by ~1–9 points, inconsistently signed) — **the pool is what moves the number**, and the reference protocol gets the pool wrong.

---

## 3. The corrected task

For each duplicate link `{a, b}`:

- the **later-filed** report is the query, the earlier one is gold — one query per link, so symmetry stops double-counting;
- **candidates are reports filed strictly before the query**, so nothing is credited for retrieving the future (and the query excludes itself for free, since the filter is strict);
- **NEW queries** are sampled from reports never linked as duplicates, matched on filing time so the classes cannot be separated on date alone;
- **splits are by time**, holding out the most recent queries — the only split that cannot put a link's symmetric partner on the other side of the boundary.

The time filter is applied as a mask on the score vector *before* the top-k, not as a post-filter on a retrieved list, so there is no truncation artifact.

---

## 4. Results

### 4.1 Candidate retrieval (time-aware)

```bash
python scripts/run_retrieval.py
```

| project | retriever | R@1 | R@5 | R@10 | R@25 | MRR | ms/query |
|---|---|---:|---:|---:|---:|---:|---:|
| cassandra | tfidf | 0.410 | 0.603 | 0.667 | 0.737 | 0.496 | 1.4 |
| cassandra | bm25 | 0.385 | 0.577 | 0.647 | 0.686 | 0.478 | 141.4 |
| cassandra | keyword | 0.327 | 0.468 | 0.519 | 0.603 | 0.391 | 2.3 |
| cassandra | dense | 0.410 | 0.615 | **0.692** | 0.750 | **0.502** | 99.3 |
| cassandra | rrf-weighted | 0.404 | **0.622** | 0.679 | **0.756** | 0.499 | 170.5 |
| hadoop | tfidf | 0.508 | 0.723 | 0.800 | 0.862 | 0.606 | 0.8 |
| hadoop | bm25 | 0.492 | 0.723 | 0.785 | 0.831 | 0.590 | 33.3 |
| hadoop | keyword | 0.338 | 0.554 | 0.646 | 0.738 | 0.443 | 1.1 |
| hadoop | dense | 0.477 | **0.800** | **0.846** | 0.877 | 0.610 | 273.9 |
| hadoop | rrf-weighted | 0.508 | 0.738 | 0.831 | **0.892** | **0.623** | 71.5 |

`rrf-weighted` fuses dense / BM25 / keyword at 0.6 / 0.3 / 0.1 by reciprocal rank.

Worth noting before reaching for anything heavier: **TF-IDF at ~1 ms/query is within a few points of every more expensive option**, and beats BM25 on both projects. Rank fusion buys the deepest recall (R@25) but does not dominate at R@10. This is consistent with the duplicate-report literature's recurring finding that simple lexical baselines are hard to beat.

### 4.2 The NEW-vs-DUPLICATE decision

Retrieval cannot answer this — there is always a top-1. A system is only useful if it can also say NEW.

```bash
python scripts/run_decision.py cassandra --model gpt-4o-mini
```

Baseline: the top-1 retrieval score with a threshold **tuned on the held-out-by-time tune split**, which favours the baseline over the untuned LLM profiles — deliberately, since the question is whether a verifier earns its cost against a well-configured cheap system.

All systems see the identical `rrf-weighted` shortlist (top 5). Verifier: `gpt-4o-mini`.

**cassandra** — 94 test queries (47 duplicate / 47 new)

| system | precision | recall | F1 | accuracy | **master correct** |
|---|---:|---:|---:|---:|---:|
| top1-threshold (tuned) | 0.571 | **0.851** | 0.684 | 0.606 | 0.450 |
| llm-balanced | 0.762 | 0.681 | 0.719 | 0.734 | 0.750 |
| llm-precision | **0.795** | 0.660 | **0.721** | **0.745** | **0.806** |
| llm-recall | 0.708 | 0.723 | 0.716 | 0.713 | 0.706 |

**hadoop** — 39 test queries (19 duplicate / 20 new)

| system | precision | recall | F1 | accuracy | **master correct** |
|---|---:|---:|---:|---:|---:|
| top1-threshold (tuned) | 0.722 | 0.684 | 0.703 | 0.718 | 0.538 |
| llm-balanced | 0.812 | 0.684 | 0.743 | 0.769 | 0.692 |
| llm-precision | **0.833** | 0.526 | 0.645 | 0.718 | **0.800** |
| llm-recall | 0.773 | **0.895** | **0.829** | **0.821** | 0.706 |

**hbase** — 34 test queries (17 duplicate / 17 new)

| system | precision | recall | F1 | accuracy | **master correct** |
|---|---:|---:|---:|---:|---:|
| top1-threshold (tuned) | 0.500 | 1.000 | 0.667 | 0.500 | 0.294 |
| llm-balanced | 0.714 | 0.588 | 0.645 | 0.676 | 0.600 |
| llm-precision | 0.727 | 0.471 | 0.571 | 0.647 | **0.750** |
| llm-recall | **0.750** | 0.706 | **0.727** | **0.735** | 0.583 |

**spark** — 153 test queries (76 duplicate / 77 new) — the largest split, and the one to trust most

| system | precision | recall | F1 | accuracy | **master correct** |
|---|---:|---:|---:|---:|---:|
| top1-threshold (tuned) | 0.591 | 0.724 | 0.651 | 0.614 | 0.345 |
| llm-balanced | 0.745 | 0.500 | 0.598 | 0.667 | 0.658 |
| llm-precision | **0.786** | 0.434 | 0.559 | 0.660 | **0.727** |
| llm-recall | 0.768 | 0.566 | **0.652** | **0.699** | 0.605 |

On spark the verifier's F1 is *worse* than the baseline's (0.598 vs 0.651) while `master correct` more than doubles (0.345 → 0.727). A benchmark reporting F1 would record this as a regression.

Four things to read off these:

**F1 can reward a degenerate system.** On hbase the tuned threshold baseline collapses to answering DUPLICATE for everything — recall 1.000, precision 0.500, which is exactly the base rate. Its F1 of 0.667 is the highest F1 in that column, and it is worth nothing: accuracy is 0.500 (chance) and it names the right master 29% of the time. Any benchmark that ranks on F1 at a 50% base rate will crown this system.


**The verifier's win is precision, not F1.** On cassandra F1 moves 0.684 → 0.721, which is nearly noise at this sample size. Precision moves 0.571 → 0.795 and *master correct* moves 0.450 → 0.806. That distinction matters operationally: the threshold baseline merges aggressively and names the wrong master more than half the time it merges. Reporting F1 alone would hide the entire effect.

**`master correct` is the metric that should be headline.** A system can call DUPLICATE correctly and still attach the report to the wrong parent — which in a real tracker buries a live defect under an unrelated ticket. Precision counts that as a success; `master correct` does not.

**Prompt profiles move the operating point without touching the retriever.** `precision` buys +0.04–0.11 precision for −0.02–0.16 recall against `balanced`; `recall` trades the other way. One deployment can serve a team that fears missed duplicates and a team that fears wrong merges, with no re-tuning and no re-indexing.

Caveat: these test splits are small (39–94 queries). The direction is consistent across projects and metrics, but the individual figures carry wide intervals — treat them as a working baseline, not a settled result.

---

## 5. Layer 2 — real CI failures

Layer 1 runs on prose bug reports. Layer 2 runs the **identical stack** on raw CI logs, which is the experiment: does a retriever-plus-verifier tuned on written reports survive contact with stack traces?

```bash
python scripts/build_ci_dataset.py --repo pytorch/pytorch --issues 600
python scripts/run_ci_benchmark.py --mask-identity
```

`triagebench/ci/task.py` loads the CI corpus into the same `Project` object `triagebench/data.py` produces, so `build_task`, every retriever, both verifiers and all metrics run unmodified.

### 5.1 Where the ground truth comes from

The obvious label is a **re-run**: same commit, failed then passed, therefore flaky. It does not survive contact with the data. Across the busiest public repositories only 8–15% of failed runs are ever re-run (elasticsearch and home-assistant: zero in 100), and sampling pytorch's re-runs gave 16 job-level transitions that were *all* `failure → failure`. There is no usable flaky class there.

Projects that run a **flaky-test bot** publish the label directly instead. pytorch/pytorch has 7,596 bot-filed `DISABLED <test> (<TestClass>)` issues, each citing the CI job that triggered it. One job log holds many failing tests, not just the flagged one, so the same logs supply both flagged and unflagged failures in identical format.

**The duplicate label is stronger here than in Layer 1.** Two bug reports are duplicates because a human noticed and linked them — missing every pair nobody linked. Two CI failures are duplicates when they are failures of the same `file.py::Class::test`. Nothing is missed. The cost is a narrower notion of "same problem": one root cause breaking three tests appears as three groups, so Layer 2 recall is measured against a stricter target and the two layers' numbers are **not** directly comparable.

### 5.2 The label must not be a feature

Run as-is, every retriever scores a perfect 1.000 at R@1. That is not a result — failures are grouped *by test id*, and the test id is sitting in the retrieval text, so the task degenerates into exact string matching.

`--mask-identity` removes the test id, class and test names from the text, including the `in <test_name>` traceback frames. File paths and line numbers are deliberately **kept**: they are evidence a real triage system legibly has, not the label. Both modes are reported, because the gap between them is itself the measurement.

### 5.3 Results

Corpus: 64 failure records, 52 distinct tests, 13 duplicate pairs, 35 jobs, spanning 2026-08-20 → 2026-11-08.

| retriever | R@1 (identity visible) | **R@1 (masked)** | MRR (masked) | ms/query |
|---|---:|---:|---:|---:|
| tfidf | 1.000 | **0.818** | 0.909 | 0.5 |
| bm25 | 1.000 | 0.727 | 0.864 | 2.0 |
| dense | 1.000 | **0.182** | 0.344 | 54.4 |
| rrf-weighted (0.6 dense) | 1.000 | 0.364 | 0.516 | 9.3 |

**Dense retrieval collapses on stack traces.** On bug reports it led every metric (R@10 0.692 / 0.846). On masked CI logs it finds the right prior failure first 18% of the time, while TF-IDF — the cheapest thing in the repo — *rises* to 82%. Sentence embeddings trained on natural language have little to say about a traceback; the signal is in shared identifiers, paths and exception types, which is exactly what term weighting captures.

**The fusion weighting that was right for prose is actively harmful for logs.** `rrf-weighted` puts 0.6 on dense, and scores 0.364 against TF-IDF's 0.818 — rank fusion faithfully averages in a signal that has become noise. A deployed triage stack carrying these weights from a bug-report evaluation into log-based triage would be substantially worse than deleting its most expensive component.

Decision layer, 9 held-out queries (5 duplicate / 4 new), identity masked:

| system | precision | recall | F1 | accuracy | master correct |
|---|---:|---:|---:|---:|---:|
| top1-threshold (tuned) | 0.625 | 1.000 | 0.769 | 0.667 | 0.200 |
| llm-balanced | 0.714 | 1.000 | 0.833 | 0.778 | 0.400 |
| llm-precision | 0.833 | 1.000 | 0.909 | 0.889 | 0.400 |

> **These are 11 retrieval queries and 9 decisions.** They are a working pipeline and a strong directional signal, not a result. The dense-vs-lexical gap (0.182 vs 0.818) is large and mechanistically plausible; everything else here needs an order of magnitude more data before it means anything.

### 5.4 Why the corpus is small, and what fixes it

GitHub Actions log retention is short and set per repository. Of 600 pytorch DISABLED issues reaching back to 2026-06-08, only 128 cite a job at all, and only 43 of those logs were still downloadable. A one-shot build is therefore permanently small.

The collector is **incremental** — it keeps previously collected records and skips jobs already fetched, keyed on `(job_id, test_id)` — so the corpus grows by running it regularly. Collecting on a schedule is the intended use; a one-off run is a sample, not a dataset.

### 5.5 Scheduled collection

[`.github/workflows/collect-ci-corpus.yml`](.github/workflows/collect-ci-corpus.yml) harvests daily and commits the corpus back to the repository.

The collection path is **pure standard library** — no pandas, no torch, nothing to install — so the job needs no dependency step and finishes in well under a minute. `tests/test_ci.py` pins that property, because an `import pandas` added to the parser would quietly turn this into a multi-minute build.

Four things the workflow does deliberately:

- **`--max-jobs` caps downloads per run.** An unbounded scheduled download is a trap: one slow day and the job runs until the timeout, collecting nothing.
- **Newest jobs first.** Logs expire, so the most recent jobs are precisely the ones that will not be collectable tomorrow.
- **`cancel-in-progress: false`** with a concurrency group. Two collectors racing on `failures.jsonl` would interleave writes.
- **It fails loudly on auth errors.** If every download returns 401/403 the collector exits non-zero rather than reporting an empty but successful run. This is the same failure that made every log look "expired" during development, and a silent scheduled collector is worse than a broken one.

The default `GITHUB_TOKEN` is scoped to its own repository. If reading another repository's Actions logs is rejected, add a PAT with `public_repo` as a secret named `COLLECTOR_TOKEN`; the workflow prefers it automatically. Note also that GitHub disables scheduled workflows in repositories with 60 days of no activity — a committing collector keeps itself alive, but a long pause needs a manual re-enable.

To collect locally instead (or before the repo is pushed), [`scripts/collect_local.sh`](scripts/collect_local.sh) does the same thing, with a launchd template beside it:

```bash
sed "s|__REPO_DIR__|$PWD|g" scripts/com.triagebench.collect.plist \
  > ~/Library/LaunchAgents/com.triagebench.collect.plist
launchctl load ~/Library/LaunchAgents/com.triagebench.collect.plist
```

launchd runs with a minimal environment and no keychain access, so `gh auth token` can fail there; the wrapper resolves the token up front and exits with a clear message rather than collecting nothing.

One implementation note worth repeating, because it cost an hour: the job-logs endpoint does not serve the log, it 302s to blob storage, and forwarding the GitHub `Authorization` header to that host returns a 401 that is trivially mistaken for an expired log. `GitHub.job_log` follows the redirect unauthenticated and returns an explicit reason string rather than `None`.

---

## 6. Reproducing

```bash
python -m venv --system-site-packages .venv && ./.venv/bin/pip install -r requirements.txt
```

```bash
python scripts/download.py          # fetch GitBugs into data/raw/
python scripts/audit_dataset.py     # §1
python scripts/compare_protocols.py # §2
python scripts/run_retrieval.py     # §4.1
python scripts/run_decision.py cassandra --model gpt-4o-mini   # §4.2

python scripts/build_ci_dataset.py --issues 600                # §5, incremental
python scripts/run_ci_benchmark.py --mask-identity             # §5.3
python scripts/summarize.py                                    # regenerate tables
```

LLM calls are cached on disk keyed by the exact prompt, so re-running costs nothing and published numbers are reproducible without re-billing. Corpus embeddings are cached the same way.

---

## 7. Limitations

- **Ground-truth coverage is partial.** Trackers only record the duplicates somebody noticed and linked. Unmarked duplicates are scored as NEW, so absolute precision is pessimistic for every system here.
- **Some projects lose most of their links.** vscode ships 8,645 duplicate links but only 371 where both reports are in the corpus; thunderbird loses 1,485 of 3,460. The audit reports this per project.
- **NEW queries are sampled, not observed.** Reports never linked as duplicates stand in for the non-duplicate stream. Time-matching controls the most obvious confound, not every one.
- **Report text only.** No logs, stack-trace artifacts, CI history or repository state — the signals a real triage agent would have. That is the next layer, not this one.
- **The duplicate rate is a knob.** `--new-ratio` defaults to a balanced mix, which is far denser than a real tracker. Sweep it rather than trusting one value; precision at a realistic base rate is a much harder number.

Layer 2 specifically:

- **The CI corpus is far too small to conclude from.** 11 retrieval queries, 9 decisions. Only the dense-vs-lexical gap is large enough to take seriously, and only because it is an order-of-magnitude difference with an obvious mechanism.
- **One repository, one test runner.** Everything is pytorch/pytorch and pytest. The parser handles pytest's `FAILURES`/`RERUNS` banners; roughly a fifth of downloaded logs still yield no failures, mostly build errors and non-pytest runners. Generalising means more parsers, not more data.
- **Selection bias toward flaky.** Logs are reached via flaky-test issues, so the corpus over-represents tests the bot already flagged (52 of 64 records). The unflagged failures come from the same logs, which controls format but not the sampling frame.
- **Test identity is a narrow notion of "same problem".** One root cause breaking several different tests is counted as several distinct problems. Grouping failures by root cause rather than by test is the obvious next step, and it has no free label.

---

## 8. Credits

Report data: **GitBugs** (CC BY 4.0) — Patil, A. *GitBugs: Bug Reports for Duplicate Detection, Retrieval Augmented Generation, Triage, and More*, [arXiv:2504.09651](https://arxiv.org/abs/2504.09651). This repo redistributes none of it; `scripts/download.py` fetches from the upstream repository.

The audit in §1–§2 concerns the evaluation protocol in that repo's reference notebook, not the correctness of the underlying reports and links, which are real tracker data.
