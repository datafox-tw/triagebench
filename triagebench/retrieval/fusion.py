"""Reciprocal Rank Fusion over several retrievers.

    RRF(d) = sum_i  w_i / (k + rank_i(d))

Rank fusion rather than score fusion, because BM25 scores, cosine similarities
and Jaccard overlaps live on incomparable scales; normalising them per query is
possible but brittle, and ranks sidestep it entirely.

Only each retriever's top `depth` eligible results contribute, which is how
fusion is actually deployed -- a document no retriever ranks highly should not
be rescued by being mediocre everywhere.

One property worth knowing before tuning `rrf_k`: 1/(k + r) is convex, so a
document ranked 1st and 3rd slightly outscores one ranked 2nd by both
(1/61 + 1/63 > 2/62 at k=60). RRF rewards *being found at all* far more than it
rewards agreement on position, and raising k flattens the difference further.
What it reliably buys is coverage -- a document two retrievers surface beats one
that only a single retriever surfaces -- which is why fusion helps deep recall
(R@25) much more than it helps R@1.
"""

from __future__ import annotations

from collections import defaultdict

import pandas as pd

from .base import Retriever

RRF_K = 60  # standard smoothing constant; damps the top ranks' dominance


class RRFusion:
    """Fuses several retrievers. Same index/search interface as `Retriever`."""

    def __init__(self, retrievers: list[Retriever],
                 weights: list[float] | None = None,
                 depth: int = 50,
                 rrf_k: int = RRF_K,
                 name: str | None = None) -> None:
        if weights is not None and len(weights) != len(retrievers):
            raise ValueError("weights must match retrievers")
        self.retrievers = retrievers
        self.weights = weights or [1.0] * len(retrievers)
        self.depth = depth
        self.rrf_k = rrf_k
        self.name = name or "rrf(" + "+".join(r.name for r in retrievers) + ")"

    def index(self, reports: pd.DataFrame) -> "RRFusion":
        for r in self.retrievers:
            r.index(reports)
        return self

    def search(self, query_text: str, cutoff: pd.Timestamp,
               top_k: int = 10) -> list[tuple[int, float]]:
        fused: dict[int, float] = defaultdict(float)

        for retriever, weight in zip(self.retrievers, self.weights):
            if weight == 0:
                continue
            hits = retriever.search(query_text, cutoff, top_k=self.depth)
            for rank, (doc_id, _score) in enumerate(hits, start=1):
                fused[doc_id] += weight / (self.rrf_k + rank)

        ranked = sorted(fused.items(), key=lambda kv: -kv[1])
        return ranked[:top_k]

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        parts = ", ".join(f"{r.name}:{w}" for r, w in zip(self.retrievers, self.weights))
        return f"RRFusion({parts}, depth={self.depth})"
