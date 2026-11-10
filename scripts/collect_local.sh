#!/bin/bash
# Local scheduled collection, for before the repo is pushed to GitHub (or
# alongside it). Harvests new CI failures and appends them to the corpus.
#
# The collector is idempotent: it skips jobs already collected, so running it
# more often than necessary costs only API quota.
#
# Install as a daily launchd job:
#   sed "s|__REPO_DIR__|$PWD|g" scripts/com.triagebench.collect.plist \
#     > ~/Library/LaunchAgents/com.triagebench.collect.plist
#   launchctl load ~/Library/LaunchAgents/com.triagebench.collect.plist
# Remove:
#   launchctl unload ~/Library/LaunchAgents/com.triagebench.collect.plist

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

PYTHON="$REPO_DIR/.venv/bin/python"
[ -x "$PYTHON" ] || PYTHON="$(command -v python3)"

LOG_DIR="$REPO_DIR/data/ci/_logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/collect-$(date +%Y%m%d).log"

# launchd gives a minimal environment with no keychain access, so `gh auth
# token` can fail there. Resolve the token now and pass it explicitly.
if [ -z "${GITHUB_TOKEN:-}" ] && command -v gh >/dev/null 2>&1; then
  GITHUB_TOKEN="$(gh auth token 2>/dev/null || true)"
  export GITHUB_TOKEN
fi

if [ -z "${GITHUB_TOKEN:-}" ]; then
  echo "$(date -Iseconds) no GITHUB_TOKEN; set it in the plist or the environment" >>"$LOG"
  exit 1
fi

{
  echo "=== $(date -Iseconds) ==="
  "$PYTHON" scripts/build_ci_dataset.py \
      --repo "${TRIAGEBENCH_REPO:-pytorch/pytorch}" \
      --issues "${TRIAGEBENCH_ISSUES:-600}" \
      --max-jobs "${TRIAGEBENCH_MAX_JOBS:-120}" \
      --workers 6
} >>"$LOG" 2>&1

# Keep two weeks of run logs.
find "$LOG_DIR" -name 'collect-*.log' -mtime +14 -delete 2>/dev/null || true
