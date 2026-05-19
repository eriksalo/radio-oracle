"""Retrieval modes — snappy first answer vs deeper reasoned answer.

Tier 1 ("snappy"): fast top-K from each backend, no reranker. Default for first
answer to a query.

Tier 2 ("deep"): wider candidate pool + cross-encoder reranker. Triggered by
user follow-up intent ("go deeper", "tell me more", etc.).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

RetrievalMode = Literal["snappy", "deep"]


@dataclass(frozen=True)
class ModeParams:
    """Per-mode retrieval knobs."""

    mode: RetrievalMode
    per_collection_top_k: int
    rerank_pool: int  # candidates fed to reranker (0 = no reranker)
    final_top_k: int  # results returned to LLM
    max_collections: int = 0  # 0 = all available


def params_for(mode: RetrievalMode, settings) -> ModeParams:
    if mode == "snappy":
        return ModeParams(
            mode="snappy",
            per_collection_top_k=settings.tier1_top_k,
            rerank_pool=0,
            final_top_k=settings.tier1_top_k,
        )
    return ModeParams(
        mode="deep",
        per_collection_top_k=settings.tier2_top_k,
        rerank_pool=settings.tier2_rerank_pool,
        final_top_k=settings.tier2_final_top_k,
        max_collections=3,
    )


_DEEPER_TRIGGERS = re.compile(
    r"\b("
    r"go deeper|dig deeper|tell me more|more detail|elaborate|expand on (that|this)|"
    r"anything else|what else|is there more|more sources|cite (more|sources)|"
    r"deeper (look|dive|answer)"
    r")\b",
    flags=re.IGNORECASE,
)


def detect_mode(user_input: str, default: RetrievalMode = "snappy") -> RetrievalMode:
    """Pick a retrieval mode from the user's wording."""
    return "deep" if _DEEPER_TRIGGERS.search(user_input) else default
