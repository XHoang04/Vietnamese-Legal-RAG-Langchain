"""Cross-encoder reranker for legal query-passage ranking."""

import logging
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class BaseReranker(Protocol):
    """Protocol for cross-encoder rerankers."""

    def rank(self, query: str, passages: list[str]) -> list[float]:
        """Return raw logit scores for each passage given the query."""
        ...


class FakeReranker:
    """Mock reranker computing deterministic scores based on token overlap for fast testing."""

    def __init__(self, name: str = "fake-reranker") -> None:
        self.name = name

    def rank(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        q_tokens = set(query.lower().split())
        scores: list[float] = []
        for p in passages:
            p_tokens = set(p.lower().split())
            overlap = len(q_tokens & p_tokens)
            scores.append(float(overlap))
        return scores


class VietnameseCrossEncoderReranker:
    """Cross-encoder reranker wrapping AITeamVN/Vietnamese_Reranker with AutoModelForSequenceClassification."""

    def __init__(
        self,
        model_name: str = "AITeamVN/Vietnamese_Reranker",
        max_length: int = 1024,
        batch_size: int = 8,
        device: str = "cpu",
    ) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device

        logger.info("Loading CrossEncoder reranker %s on %s...", model_name, device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

    def rank(self, query: str, passages: list[str]) -> list[float]:
        """Score (query, passage) pairs using model forward pass and return raw logits."""
        if not passages:
            return []

        import torch

        all_scores: list[float] = []
        for i in range(0, len(passages), self.batch_size):
            batch_passages = passages[i : i + self.batch_size]
            pairs = [[query, p] for p in batch_passages]

            features = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**features)
                logits = outputs.logits
                if logits.shape[1] == 1:
                    batch_scores = logits.squeeze(-1).tolist()
                else:
                    batch_scores = logits[:, 1].tolist() if logits.shape[1] > 1 else logits[:, 0].tolist()

                if isinstance(batch_scores, float):
                    batch_scores = [batch_scores]
                all_scores.extend(batch_scores)

        return all_scores
