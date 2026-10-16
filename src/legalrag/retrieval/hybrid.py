"""Reciprocal Rank Fusion (RRF) algorithm for combining dense and sparse retrieval results."""

from collections import defaultdict


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    weights: list[float] | None = None,
    rrf_k: int = 60,
) -> list[tuple[str, float]]:
    """Compute Reciprocal Rank Fusion scores across multiple ranked lists.

    score(doc) = sum_{list i} (weight_i / (rrf_k + rank_i(doc)))
    rank is 1-indexed (1st place = 1, 2nd = 2, ...).

    Returns sorted list of (doc_id, score) tuples in descending order of score.
    """
    if not ranked_lists:
        return []

    if weights is None:
        weights = [1.0] * len(ranked_lists)
    elif len(weights) != len(ranked_lists):
        raise ValueError(f"Mismatch between number of lists ({len(ranked_lists)}) and weights ({len(weights)})")

    scores: dict[str, float] = defaultdict(float)

    for list_idx, ranked_list in enumerate(ranked_lists):
        w = weights[list_idx]
        for rank, item_id in enumerate(ranked_list, start=1):
            scores[item_id] += w / (rrf_k + rank)

    sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results
