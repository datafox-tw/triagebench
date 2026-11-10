"""Verifier interface: turn a candidate list into a triage decision.

Retrieval produces an ordered list of plausible masters. It cannot say NEW --
there is always a top-1. The verifier is the component that makes the call,
and it is where precision is won or lost: a retriever with excellent Recall@25
is worthless if everything it surfaces gets merged.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Candidate:
    doc_id: int
    score: float
    text: str


@dataclass
class Decision:
    """One triage call."""

    query_id: int
    master_id: int | None          # None => NEW
    reason: str = ""
    latency_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    failed: bool = False           # call errored and fell back

    @property
    def is_duplicate(self) -> bool:
        return self.master_id is not None


class Verifier(ABC):
    name: str = "verifier"

    @abstractmethod
    def verify(self, query_id: int, query_text: str,
               candidates: list[Candidate]) -> Decision:
        """Decide whether `query_id` duplicates one of `candidates`."""


class TopOneThreshold(Verifier):
    """Accept the top candidate when its retrieval score clears a threshold.

    The baseline every LLM verifier has to beat. It is what a production system
    does before anyone adds a model, and it is a genuinely strong baseline once
    the threshold is tuned on held-out data.
    """

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold
        self.name = f"top1@{threshold:.4f}"

    def verify(self, query_id: int, query_text: str,
               candidates: list[Candidate]) -> Decision:
        if candidates and candidates[0].score >= self.threshold:
            return Decision(query_id, candidates[0].doc_id, reason="score>=threshold")
        return Decision(query_id, None, reason="score<threshold")
