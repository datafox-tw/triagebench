"""Dense retrieval with sentence-transformers, with an on-disk embedding cache.

Encoding dominates runtime, and the corpus is fixed for a given project, so
embeddings are cached under `data/cache/` keyed by model and corpus content.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .base import Retriever

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CACHE_DIR = Path("data/cache")


class DenseRetriever(Retriever):
    name = "dense"

    def __init__(self, model_name: str = DEFAULT_MODEL,
                 batch_size: int = 128,
                 cache_dir: Path | str = CACHE_DIR) -> None:
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.cache_dir = Path(cache_dir)
        self._model = None
        self._emb: np.ndarray | None = None
        # Query vectors are reused whenever the same retriever appears both
        # standalone and inside a fusion, so encoding is done at most once per
        # distinct query text.
        self._query_cache: dict[str, np.ndarray] = {}

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _cache_path(self, texts: list[str]) -> Path:
        h = hashlib.sha256()
        h.update(self.model_name.encode())
        h.update(str(len(texts)).encode())
        for t in texts:
            h.update(t.encode("utf-8", "ignore")[:512])
            h.update(b"\0")
        return self.cache_dir / f"emb-{h.hexdigest()[:16]}.npy"

    def _fit(self, texts: list[str]) -> None:
        path = self._cache_path(texts)
        if path.exists():
            self._emb = np.load(path)
            return

        model = self._load_model()
        emb = model.encode(texts, batch_size=self.batch_size,
                           convert_to_numpy=True, normalize_embeddings=True,
                           show_progress_bar=True)
        self._emb = emb.astype(np.float32)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self._emb)

    def _score_all(self, query_text: str) -> np.ndarray:
        q = self._query_cache.get(query_text)
        if q is None:
            model = self._load_model()
            q = model.encode([query_text], convert_to_numpy=True,
                             normalize_embeddings=True)[0].astype(np.float32)
            self._query_cache[query_text] = q
        return self._emb @ q

    def index(self, reports: pd.DataFrame) -> "DenseRetriever":
        super().index(reports)
        return self

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        """Batch-encode queries once, instead of one call per search."""
        model = self._load_model()
        return model.encode(texts, batch_size=self.batch_size,
                            convert_to_numpy=True,
                            normalize_embeddings=True).astype(np.float32)

    def score_from_vector(self, vec: np.ndarray) -> np.ndarray:
        return self._emb @ vec
