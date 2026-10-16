"""Fake LLM and mock factory for tests without network/GPU dependencies."""

from typing import Any, Callable

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field


class FakeChatModel(BaseChatModel):
    """
    Fake Chat Model for testing graph nodes.
    Supports scripted responses or custom response generator function,
    and counts invocations.
    """
    scripted_responses: list[str] = Field(default_factory=list)
    response_fn: Callable[[list[BaseMessage]], str] | None = None
    call_count: int = 0

    @property
    def _llm_type(self) -> str:
        return "fake-chat-model"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.call_count += 1

        if self.response_fn is not None:
            content = self.response_fn(messages)
        elif self.scripted_responses:
            idx = min(self.call_count - 1, len(self.scripted_responses) - 1)
            content = self.scripted_responses[idx]
        else:
            content = '{"answer": "Theo quy định pháp luật [1], nội dung được hướng dẫn cụ thể.", "used_ctx": [1], "insufficient_context": false, "intent": "legal_query", "scope": "corpus", "complexity": "simple", "standalone_question": "câu hỏi", "sufficient": true, "query": "truy vấn", "sub_questions": ["câu hỏi 1"], "unsupported_claims": []}'

        message = AIMessage(content=content)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])


class FakeLLMFactory:
    """Factory returning FakeChatModel for all roles."""

    def __init__(self, responses: dict[str, list[str]] | None = None) -> None:
        self.responses = responses or {}
        self.created_models: dict[str, FakeChatModel] = {}

    def get_llm(self, role: str, *, contains_upload: bool = False) -> FakeChatModel:
        if role not in self.created_models:
            scripted = self.responses.get(role, [])
            self.created_models[role] = FakeChatModel(scripted_responses=scripted)
        return self.created_models[role]
