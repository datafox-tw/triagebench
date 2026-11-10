"""Retriever interface with exact time-aware filtering.

Every retriever here scores the whole corpus for a query. That makes the
time filter *exact* rather than approximate: instead of retrieving top-N and
discarding future reports afterwards -- which silently truncates when the
future crowds the head of the ranking -- the mask is applied to the score
vector before the top-k is taken.

A query's own report has `created == cutoff`, and the mask is strict (`<`), so
self-matches are excluded without a special case.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class Retriever(ABC):
    """Scores every indexed report against a query string."""

    name: str = "retriever"

    def __init__(self) -> None:
        self._ids: np.ndarray | None = None
        self._created: np.ndarray | None = None

    # -- indexing -----------------------------------------------------------

    def index(self, reports: pd.DataFrame) -> "Retriever":
        """Index a corpus. `reports` needs `text` and `created`, id-indexed."""
        usable = reports[reports["created"].notna()]
        self._ids = usable.index.to_numpy()
        self._created = usable["created"].astype("int64").to_numpy()
        self._fit(usable["text"].tolist())
        return self

    @abstractmethod
    def _fit(self, texts: list[str]) -> None:
        """Build whatever index the retriever needs over `texts`."""

    @abstractmethod
    def _score_all(self, query_text: str) -> np.ndarray:
        """Similarity of `query_text` against every indexed document."""

    # -- search -------------------------------------------------------------

    def search(self, query_text: str, cutoff: pd.Timestamp,
               top_k: int = 10) -> list[tuple[int, float]]:
        """Top-k reports filed strictly before `cutoff`, best first."""
        if self._ids is None or self._created is None:
            raise RuntimeError(f"{self.name}: index() must be called first")

        scores = self._score_all(query_text)
        eligible = self._created < np.int64(cutoff.value)
        n_eligible = int(eligible.sum())
        if n_eligible == 0:
            return []

        masked = np.where(eligible, scores, -np.inf)
        k = min(top_k, n_eligible)
        top = np.argpartition(-masked, k - 1)[:k]
        top = top[np.argsort(-masked[top], kind="stable")]

        return [(int(self._ids[i]), float(masked[i])) for i in top
                if np.isfinite(masked[i])]

    def search_untimed(self, query_text: str, exclude_id: int,
                       top_k: int = 10) -> list[tuple[int, float]]:
        """Top-k over the **whole** corpus, excluding only the query itself.

        This is the reference protocol used by the GitBugs duplicate-detection
        notebook. It is kept here solely so the two protocols can be run
        side by side on identical code; it is not a realistic triage setting,
        because roughly half of its gold answers are filed after the query.
        """
        if self._ids is None:
            raise RuntimeError(f"{self.name}: index() must be called first")

        scores = self._score_all(query_text)
        masked = np.where(self._ids != exclude_id, scores, -np.inf)
        k = min(top_k, len(masked) - 1)
        if k <= 0:
            return []
        top = np.argpartition(-masked, k - 1)[:k]
        top = top[np.argsort(-masked[top], kind="stable")]
        return [(int(self._ids[i]), float(masked[i])) for i in top
                if np.isfinite(masked[i])]

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        n = 0 if self._ids is None else len(self._ids)
        return f"{type(self).__name__}(name={self.name!r}, docs={n:,})"
