"""Retrievers for duplicate-report candidate generation."""

from .base import Retriever
from .dense import DEFAULT_MODEL, DenseRetriever
from .fusion import RRFusion
from .lexical import BM25Retriever, KeywordOverlapRetriever, TfidfRetriever, tokenize

__all__ = [
    "Retriever",
    "TfidfRetriever",
    "BM25Retriever",
    "KeywordOverlapRetriever",
    "DenseRetriever",
    "RRFusion",
    "DEFAULT_MODEL",
    "tokenize",
]
