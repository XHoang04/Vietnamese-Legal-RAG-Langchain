"""Unit tests for LLM factory, structured output, cache, rate limits, and privacy rules."""

import pytest
from pydantic import BaseModel

from legalrag.core.config import load_config
from legalrag.core.errors import StructuredOutputError, UploadRequiresLocalLLMError
from legalrag.llm.cache import LLMCache
from legalrag.llm.factory import get_llm
from legalrag.llm.fakes import FakeChatModel
from legalrag.llm.limits import TokenBucketRateLimiter
from legalrag.llm.structured import invoke_structured


class DummyOutput(BaseModel):
    name: str
    score: int


def test_invoke_structured_direct():
    fake = FakeChatModel(scripted_responses=['{"name": "test", "score": 100}'])
    res = invoke_structured(fake, DummyOutput, [{"role": "user", "content": "hi"}])
    assert res.name == "test"
    assert res.score == 100


def test_invoke_structured_repair():
    # First response invalid, second response valid JSON
    fake = FakeChatModel(
        scripted_responses=[
            'Here is the result: {"name": "invalid",',
            '{"name": "repaired", "score": 99}',
        ]
    )
    res = invoke_structured(fake, DummyOutput, [{"role": "user", "content": "hi"}], max_repair=2)
    assert res.name == "repaired"
    assert res.score == 99


def test_invoke_structured_fail_max_repairs():
    fake = FakeChatModel(scripted_responses=["not json 1", "not json 2", "not json 3"])
    with pytest.raises(StructuredOutputError):
        invoke_structured(fake, DummyOutput, [{"role": "user", "content": "hi"}], max_repair=1)


def test_privacy_guardrail_upload_remote_blocked():
    cfg = load_config()
    cfg.app.privacy.allow_remote_for_uploads = False
    cfg.models.profiles["private"] = {
        "default": {"provider": "remote_provider", "model": "remote-model"}
    }

    with pytest.raises(UploadRequiresLocalLLMError):
        get_llm("generator", contains_upload=True, config_bundle=cfg)


def test_privacy_guardrail_upload_with_ollama_allowed():
    cfg = load_config()
    cfg.app.privacy.allow_remote_for_uploads = False
    cfg.models.profiles["private"] = {
        "default": {"provider": "ollama", "model": "qwen2.5:7b"}
    }
    # Should not raise UploadRequiresLocalLLMError
    llm = get_llm("generator", contains_upload=True, config_bundle=cfg)
    assert llm is not None


def test_llm_cache(tmp_path):
    cache = LLMCache(tmp_path / "test_cache.sqlite")
    key = LLMCache.compute_key(
        role="test",
        provider="mock",
        model="m1",
        params={"t": 0.0},
        messages=[{"role": "user", "content": "hello"}],
    )
    assert cache.get(key) is None
    cache.set(key, '{"answer": "cached"}', role="test", model="m1")
    assert cache.get(key) == '{"answer": "cached"}'


def test_token_bucket_limiter():
    limiter = TokenBucketRateLimiter(rpm=60)
    assert limiter.acquire(1.0, wait=False) is True
