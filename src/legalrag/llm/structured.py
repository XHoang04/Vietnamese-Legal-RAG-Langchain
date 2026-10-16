"""Structured output schemas and invoke utility."""

import json
import re
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field

from legalrag.core.errors import StructuredOutputError


class RouterOutput(BaseModel):
    intent: Literal["legal_query", "chitchat", "out_of_scope", "needs_clarification"]
    scope: Literal["corpus", "upload", "both"]
    complexity: Literal["simple", "multi_step"]
    standalone_question: str
    clarifying_question: str | None = None


class GradeOutput(BaseModel):
    sufficient: bool
    relevant_ctx: list[int] = Field(default_factory=list)


class RewriteOutput(BaseModel):
    query: str


class PlannerOutput(BaseModel):
    sub_questions: list[str] = Field(default_factory=list)


class GenerationOutput(BaseModel):
    answer: str
    used_ctx: list[int] = Field(default_factory=list)
    insufficient_context: bool = False


class VerificationOutput(BaseModel):
    unsupported_claims: list[str] = Field(default_factory=list)


def extract_json_from_text(text: Any) -> dict[str, Any]:
    """Extract JSON object from string, handling markdown fences and block lists if present."""
    if isinstance(text, list):
        text = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in text)
    text = str(text).strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        text = match.group(1).strip()
    return json.loads(text)


def invoke_structured(
    llm: BaseChatModel,
    schema: type[BaseModel],
    messages: list[BaseMessage] | list[dict[str, Any]],
    *,
    max_repair: int = 2,
) -> BaseModel:
    """
    Invoke LLM with structured output schema, with fallback and error repair.
    """
    # 1. Try with_structured_output if model supports it
    if hasattr(llm, "with_structured_output"):
        try:
            structured_llm = llm.with_structured_output(schema)
            res = structured_llm.invoke(messages)
            if isinstance(res, schema):
                return res
            if isinstance(res, dict):
                return schema(**res)
        except Exception:
            pass  # Fall back to raw invoke and parse

    # 2. Raw invoke and JSON parsing with repair
    current_messages = list(messages)
    for attempt in range(max_repair + 1):
        response = llm.invoke(current_messages)
        content = response.content if hasattr(response, "content") else str(response)
        if isinstance(content, list):
            content = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)

        try:
            parsed_dict = extract_json_from_text(content)
            return schema(**parsed_dict)
        except Exception as e:
            if attempt == max_repair:
                raise StructuredOutputError(
                    f"Failed to parse structured output for {schema.__name__} after {max_repair} repairs: {e}"
                ) from e
            # Ask the model to fix the JSON
            current_messages.append({"role": "assistant", "content": content})
            current_messages.append({
                "role": "user",
                "content": f"The response was not valid JSON conforming to {schema.__name__}: {e}. Please respond with ONLY valid JSON."
            })

    raise StructuredOutputError(f"Failed to get structured output for {schema.__name__}")
