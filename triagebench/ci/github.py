"""Minimal GitHub client for collecting real CI failures.

Scope is deliberately narrow: list a repository's bot-filed flaky-test issues,
resolve the CI job each one cites, and download that job's log.

Why flaky-test issues rather than workflow re-runs. The obvious ground truth
for "was this failure real?" is a re-run: same commit, failed then passed. In
practice that signal is too thin to build on -- across the busiest public
repositories only 8-15% of failed runs are ever re-run, and sampling those
showed job-level transitions that were overwhelmingly failure -> failure.
Projects that run a flaky-test bot instead publish the label directly, at a
scale (thousands of issues) that re-run mining does not reach.

Authentication uses `GITHUB_TOKEN`, or the token from an authenticated `gh`
CLI. Unauthenticated access is 60 requests/hour and will not get far.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

API = "https://api.github.com"

# "DISABLED test_name (__main__.TestClass)" / "DISABLED test_name (module.TestClass)"
_DISABLED_TITLE = re.compile(
    r"^DISABLED\s+(?P<test>[\w\[\]\-.]+)\s*\(\s*(?P<cls>[\w.]+)\s*\)\s*$"
)

# ".../actions/runs/<run_id>/job/<job_id>" as cited in the issue body.
_JOB_LINK = re.compile(r"actions/runs/(?P<run>\d+)/job/(?P<job>\d+)")

_PLATFORMS = re.compile(r"^Platforms?:\s*(?P<p>.+)$", re.MULTILINE)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Surfaces a 302 as an HTTPError so the redirect can be followed by hand."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


@dataclass
class DisabledTest:
    """A bot-filed flaky-test issue."""

    issue_number: int
    title: str
    test_name: str          # Class.test_name
    created_at: str
    platforms: list[str]
    run_id: int | None
    job_id: int | None
    state: str

    @property
    def has_job(self) -> bool:
        return self.job_id is not None


def _token() -> str | None:
    tok = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if tok:
        return tok
    try:
        out = subprocess.run(["gh", "auth", "token"], capture_output=True,
                             text=True, timeout=15)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


class GitHub:
    def __init__(self, token: str | None = None, *, timeout: float = 120.0,
                 max_retries: int = 4) -> None:
        self.token = token or _token()
        self.timeout = timeout
        self.max_retries = max_retries
        self.n_requests = 0

    def _request(self, url: str, *, raw: bool = False) -> bytes | None:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "triagebench",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        delay = 2.0
        for attempt in range(self.max_retries):
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    self.n_requests += 1
                    return r.read()
            except urllib.error.HTTPError as e:
                # 403/429 here is nearly always secondary rate limiting.
                if e.code in (403, 429) and attempt < self.max_retries - 1:
                    reset = e.headers.get("x-ratelimit-reset")
                    wait = delay
                    if reset:
                        try:
                            wait = max(delay, min(300.0, int(reset) - time.time() + 2))
                        except ValueError:
                            pass
                    time.sleep(wait)
                    delay *= 2
                    continue
                if e.code in (404, 410):   # log expired or job removed
                    return None
                if attempt == self.max_retries - 1:
                    return None
                time.sleep(delay)
                delay *= 2
            except (urllib.error.URLError, TimeoutError, OSError):
                if attempt == self.max_retries - 1:
                    return None
                time.sleep(delay)
                delay *= 2
        return None

    # -- flaky-test issues --------------------------------------------------

    def disabled_tests(self, repo: str, *, limit: int = 200,
                       state: str = "all") -> list[DisabledTest]:
        """List bot-filed `DISABLED <test>` issues, newest first.

        Uses the issues list endpoint rather than search: search caps at 1000
        results and is rate limited far more aggressively.
        """
        out: list[DisabledTest] = []
        page = 1
        while len(out) < limit:
            url = (f"{API}/repos/{repo}/issues?state={state}&labels=skipped"
                   f"&per_page=100&page={page}&sort=created&direction=desc")
            blob = self._request(url)
            if not blob:
                break
            try:
                items = json.loads(blob)
            except json.JSONDecodeError:
                break
            if not items:
                break

            for it in items:
                if "pull_request" in it:
                    continue
                parsed = self._parse_issue(it)
                if parsed is not None:
                    out.append(parsed)
                    if len(out) >= limit:
                        break
            page += 1
            if page > 50:
                break
        return out

    @staticmethod
    def _parse_issue(item: dict) -> DisabledTest | None:
        title = (item.get("title") or "").strip()
        m = _DISABLED_TITLE.match(title)
        if not m:
            return None

        body = item.get("body") or ""
        link = _JOB_LINK.search(body)
        plat = _PLATFORMS.search(body)

        cls = m.group("cls").split(".")[-1]
        return DisabledTest(
            issue_number=item["number"],
            title=title,
            test_name=f"{cls}.{m.group('test')}",
            created_at=item["created_at"],
            platforms=[p.strip() for p in plat.group("p").split(",")] if plat else [],
            run_id=int(link.group("run")) if link else None,
            job_id=int(link.group("job")) if link else None,
            state=item.get("state", "unknown"),
        )

    # -- logs ---------------------------------------------------------------

    def job_log(self, repo: str, job_id: int) -> tuple[str | None, str]:
        """Download one job's log. Returns (text, reason).

        The logs endpoint does not serve the log: it 302s to blob storage. The
        redirect must be followed *without* the Authorization header, because
        the storage host rejects a GitHub bearer token with its own 401 -- which
        is easy to mistake for an expired log. Hence the explicit reason string.
        """
        url = f"{API}/repos/{repo}/actions/jobs/{job_id}/logs"
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "triagebench",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        opener = urllib.request.build_opener(_NoRedirect)
        delay = 2.0
        for attempt in range(self.max_retries):
            try:
                opener.open(urllib.request.Request(url, headers=headers),
                            timeout=self.timeout)
                return None, "unexpected-200-without-redirect"
            except urllib.error.HTTPError as e:
                self.n_requests += 1
                if e.code == 302:
                    location = e.headers.get("Location")
                    if not location:
                        return None, "redirect-without-location"
                    blob = self._fetch_unauthenticated(location)
                    if blob is None:
                        return None, "blob-fetch-failed"
                    return blob.decode("utf-8", errors="replace"), "ok"
                if e.code in (404, 410):
                    return None, "expired"          # past the 90-day retention
                if e.code in (403, 429) and attempt < self.max_retries - 1:
                    time.sleep(delay)
                    delay *= 2
                    continue
                return None, f"http-{e.code}"
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                if attempt == self.max_retries - 1:
                    return None, f"network-{type(e).__name__}"
                time.sleep(delay)
                delay *= 2
        return None, "retries-exhausted"

    def _fetch_unauthenticated(self, url: str) -> bytes | None:
        """Fetch a pre-signed storage URL, deliberately sending no credentials."""
        for attempt in range(self.max_retries):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "triagebench"})
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return r.read()
            except (urllib.error.HTTPError, urllib.error.URLError,
                    TimeoutError, OSError):
                if attempt == self.max_retries - 1:
                    return None
                time.sleep(2.0 * (attempt + 1))
        return None

    def rate_limit(self) -> dict:
        blob = self._request(f"{API}/rate_limit")
        if not blob:
            return {}
        return json.loads(blob).get("resources", {}).get("core", {})
