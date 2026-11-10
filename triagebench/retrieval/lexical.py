"""Lexical retrievers: TF-IDF cosine and BM25.

TfidfRetriever reproduces the reference protocol's scoring so that the only
difference from the published Recall@10 is the evaluation protocol itself, not
the method. BM25Retriever is the strong lexical baseline that duplicate-report
work repeatedly finds hard to beat.
"""

from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from .base import Retriever

_TOKEN = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens; keeps identifiers like `NullPointerException`."""
    return _TOKEN.findall(text.lower())


class TfidfRetriever(Retriever):
    name = "tfidf"

    def __init__(self, max_features: int = 10_000) -> None:
        super().__init__()
        self.max_features = max_features
        self._vec: TfidfVectorizer | None = None
        self._matrix = None

    def _fit(self, texts: list[str]) -> None:
        self._vec = TfidfVectorizer(stop_words="english",
                                    max_features=self.max_features)
        self._matrix = self._vec.fit_transform(texts)
        # Rows are L2-normalised by TfidfVectorizer, so a dot product is cosine.

    def _score_all(self, query_text: str) -> np.ndarray:
        q = self._vec.transform([query_text])
        return np.asarray((self._matrix @ q.T).todense()).ravel()


class BM25Retriever(Retriever):
    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        super().__init__()
        self.k1, self.b = k1, b
        self._bm25: BM25Okapi | None = None

    def _fit(self, texts: list[str]) -> None:
        self._bm25 = BM25Okapi([tokenize(t) for t in texts], k1=self.k1, b=self.b)

    def _score_all(self, query_text: str) -> np.ndarray:
        return np.asarray(self._bm25.get_scores(tokenize(query_text)))


class KeywordOverlapRetriever(Retriever):
    """Jaccard overlap on the token set.

    Deliberately crude. It stands in for the threshold-on-token-overlap rules
    that production triage systems accumulate, and gives the fusion a signal
    that is decorrelated from term-weighted scoring.

    Computed as a sparse product rather than per-document set intersections:
    |q & d| is one matrix-vector product against a binary term matrix, which
    keeps a whole-corpus scan practical on the larger projects.
    """

    name = "keyword"

    def __init__(self, max_features: int = 50_000) -> None:
        super().__init__()
        self.max_features = max_features
        self._vec: CountVectorizer | None = None
        self._matrix = None
        self._sizes: np.ndarray | None = None

    def _fit(self, texts: list[str]) -> None:
        self._vec = CountVectorizer(binary=True, tokenizer=tokenize,
                                    token_pattern=None,
                                    max_features=self.max_features)
        self._matrix = self._vec.fit_transform(texts).astype(np.float32)
        self._sizes = np.asarray(self._matrix.sum(axis=1)).ravel()

    def _score_all(self, query_text: str) -> np.ndarray:
        q = self._vec.transform([query_text]).astype(np.float32)
        q_size = float(q.sum())
        if q_size == 0:
            return np.zeros(self._matrix.shape[0], dtype=np.float64)
        inter = np.asarray((self._matrix @ q.T).todense()).ravel()
        union = q_size + self._sizes - inter
        return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
