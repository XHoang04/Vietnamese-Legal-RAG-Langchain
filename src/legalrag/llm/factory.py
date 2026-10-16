"""LLM factory supporting multiple providers, privacy guardrails, and role configurations."""

import logging
import os
from typing import Any

from langchain_core.language_models import BaseChatModel

from legalrag.core.config import ConfigBundle, load_config
from legalrag.core.errors import UploadRequiresLocalLLMError
from legalrag.llm.fakes import FakeChatModel

logger = logging.getLogger(__name__)


def get_llm(
    role: str = "generator",
    *,
    contains_upload: bool = False,
    config_bundle: ConfigBundle | None = None,
) -> BaseChatModel:
    """Factory function to instantiate BaseChatModel based on role, privacy rules, and configuration."""
    cfg = config_bundle or load_config()
    app_cfg = cfg.app
    models_cfg = cfg.models

    # Rule 13.5: Privacy guardrail for uploaded documents
    if contains_upload:
        if not app_cfg.privacy.allow_remote_for_uploads:
            # Must use private profile
            profile_name = "private"
        else:
            profile_name = app_cfg.llm.profile
    else:
        profile_name = app_cfg.llm.profile

    profile_dict = models_cfg.profiles.get(profile_name, {})
    default_cfg = profile_dict.get("default", {})
    role_overrides = profile_dict.get("roles", {}).get(role, {})

    # Merge default and role overrides
    merged_cfg: dict[str, Any] = {**default_cfg, **role_overrides}

    provider = str(merged_cfg.get("provider", "fake")).strip().lower()
    model = str(merged_cfg.get("model", "")).strip()
    temperature = float(merged_cfg.get("temperature", 0.0))
    max_tokens = int(merged_cfg.get("max_tokens", 1024))
    timeout_s = int(merged_cfg.get("timeout_s", 60))

    if contains_upload and not app_cfg.privacy.allow_remote_for_uploads and provider != "ollama" and provider != "fake":
        raise UploadRequiresLocalLLMError(
            "Processing uploaded documents requires a local LLM (Ollama) under current privacy settings"
        )

    # Provider instantiations
    if provider in ["fake", "mock", "test"]:
        return FakeChatModel(responses=['{"answer": "Câu trả lời giả lập", "used_ctx": [1]}'])

    if provider in ["google_genai", "google", "gemini"]:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                logger.warning("GOOGLE_API_KEY is not set in environment. Falling back to FakeChatModel.")
                return FakeChatModel()
            return ChatGoogleGenerativeAI(
                model=model or "gemini-1.5-flash",
                temperature=temperature,
                max_output_tokens=max_tokens,
                timeout=timeout_s,
                google_api_key=api_key,
            )
        except ImportError:
            logger.warning("langchain-google-genai is not installed. Falling back to FakeChatModel.")
            return FakeChatModel()

    if provider in ["groq"]:
        try:
            from langchain_groq import ChatGroq
            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                logger.warning("GROQ_API_KEY is not set in environment. Falling back to FakeChatModel.")
                return FakeChatModel()
            return ChatGroq(
                model=model or "llama-3.1-8b-instant",
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout_s,
                groq_api_key=api_key,
            )
        except ImportError:
            logger.warning("langchain-groq is not installed. Falling back to FakeChatModel.")
            return FakeChatModel()

    if provider in ["openai", "openrouter"]:
        try:
            from langchain_openai import ChatOpenAI
            api_key = os.environ.get("OPENAI_API_KEY")
            base_url = os.environ.get("OPENAI_BASE_URL")
            if not api_key:
                logger.warning("OPENAI_API_KEY is not set in environment. Falling back to FakeChatModel.")
                return FakeChatModel()
            return ChatOpenAI(
                model=model or "gpt-4o-mini",
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout_s,
                api_key=api_key,
                base_url=base_url,
            )
        except ImportError:
            logger.warning("langchain-openai is not installed. Falling back to FakeChatModel.")
            return FakeChatModel()

    if provider in ["ollama"]:
        try:
            from langchain_ollama import ChatOllama
            base_url = merged_cfg.get("base_url") or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
            return ChatOllama(
                model=model or "qwen2.5:7b",
                base_url=base_url,
                temperature=temperature,
            )
        except ImportError:
            logger.warning("langchain-ollama is not installed. Falling back to FakeChatModel.")
            return FakeChatModel()


    # Fallback to Fake if unconfigured or dev default
    logger.warning("Unrecognized provider '%s', falling back to FakeChatModel", provider)
    return FakeChatModel()


class LLMFactory:
    """Factory object providing get_llm method for dependency injection."""

    def __init__(self, config_bundle: ConfigBundle | None = None) -> None:
        self.config_bundle = config_bundle

    def get_llm(self, role: str = "generator", *, contains_upload: bool = False) -> BaseChatModel:
        return get_llm(role, contains_upload=contains_upload, config_bundle=self.config_bundle)

