"""Dataset models and I/O for Golden Set evaluation."""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class GoldenArticle(BaseModel):
    """Article target for retrieval evaluation ground truth."""
    doc_id: str
    article_no: str


class GoldenQuestion(BaseModel):
    """Single annotated golden question for RAG evaluation."""
    id: str
    category: Literal["single", "multi_step", "validity", "out_of_scope", "followup"]
    question: str
    history: list[dict[str, str]] = Field(default_factory=list)
    gold_articles: list[GoldenArticle] = Field(default_factory=list)
    reference_answer: str = ""
    expected_behavior: Literal["answer", "refuse", "clarify"] = "answer"
    expected_validity: Literal["in_force", "expired", "partial", "none"] = "none"
    source_parent_ids: list[str] = Field(default_factory=list)
    split: Literal["dev", "test"] = "test"
    review_status: Literal["pending", "accepted", "edited", "rejected"] = "pending"
    reviewer_note: str = ""


def load_golden_set(path: Path | str = "eval/golden_set.jsonl") -> list[GoldenQuestion]:
    """Load Golden Set items from JSON Lines file."""
    file_path = Path(path)
    if not file_path.exists():
        return []
    items: list[GoldenQuestion] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                data = json.loads(line_str)
                items.append(GoldenQuestion(**data))
    return items


def save_golden_set(items: list[GoldenQuestion], path: Path | str = "eval/golden_set.jsonl") -> None:
    """Save Golden Set items to JSON Lines file."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(item.model_dump_json() + "\n")
