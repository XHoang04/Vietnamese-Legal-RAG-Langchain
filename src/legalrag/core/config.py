"""Configuration loader with environment variable substitution and pydantic models."""

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


def substitute_env_vars(obj: Any) -> Any:
    """Recursively replace ${VAR} or ${VAR:-default} with environment variables."""
    pattern = re.compile(r"\$\{([A-Za-z0-9_]+)(?::-([^}]*))?\}")

    def _replace_match(match: re.Match) -> str:
        var_name = match.group(1)
        default_val = match.group(2) if match.group(2) is not None else ""
        return os.environ.get(var_name, default_val)

    if isinstance(obj, str):
        return pattern.sub(_replace_match, obj)
    elif isinstance(obj, dict):
        return {k: substitute_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [substitute_env_vars(item) for item in obj]
    return obj


def load_yaml(file_path: Path) -> dict[str, Any]:
    """Load YAML file with environment variable substitution."""
    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        raw_data = yaml.safe_load(f) or {}
    return substitute_env_vars(raw_data)


# Configuration Models
class DBConfig(BaseModel):
    url: str = "sqlite:///data/processed/legalrag.db"


class QdrantConfig(BaseModel):
    url: str | None = None
    path: str | None = "data/processed/qdrant_local"
    collection: str = "legal_chunks"


class FiltersConfig(BaseModel):
    min_text_chars: int = 200
    mismatch_overlap_min: float = 0.3


class ChunkingConfig(BaseModel):
    min_article_tokens_for_split: int = 400
    max_chunk_tokens: int = 900
    flat_window_tokens: int = 600
    flat_overlap_tokens: int = 80
    min_chunk_tokens: int = 20
    min_newline_density: int = 3
    index_preamble: bool = True
    preamble_max_tokens: int = 800


class CheckpointerConfig(BaseModel):
    backend: Literal["memory", "sqlite", "postgres"] = "sqlite"
    sqlite_path: str = "data/processed/checkpoints.db"
    postgres_url: str | None = None


class GuardrailConfig(BaseModel):
    max_chars: int = 2000
    llm_enabled: bool = False
    patterns: list[str] = Field(default_factory=list)


class RouterConfig(BaseModel):
    merge_rewrite: bool = True


class GradeConfig(BaseModel):
    mode: Literal["threshold", "llm", "hybrid"] = "threshold"
    tau_high: float | None = None
    tau_low: float | None = None


class VerifyConfig(BaseModel):
    mode: Literal["code", "llm"] = "code"


class MultiStepConfig(BaseModel):
    enabled: bool = True
    max_sub_questions: int = 4


class GraphConfig(BaseModel):
    mode: Literal["lean", "full"] = "lean"
    max_retries: int = 2
    max_clarifications: int = 1
    recursion_limit: int = 25
    checkpointer: CheckpointerConfig = Field(default_factory=CheckpointerConfig)
    guardrail: GuardrailConfig = Field(default_factory=GuardrailConfig)
    router: RouterConfig = Field(default_factory=RouterConfig)
    grade: GradeConfig = Field(default_factory=GradeConfig)
    verify: VerifyConfig = Field(default_factory=VerifyConfig)
    multi_step: MultiStepConfig = Field(default_factory=MultiStepConfig)

    def apply_presets(self, raw_data: dict[str, Any]) -> None:
        """Apply lean/full mode presets unless explicitly overridden."""
        graph_raw = raw_data.get("graph", {})
        mode = graph_raw.get("mode", self.mode)

        if mode == "lean":
            if "router" not in graph_raw or "merge_rewrite" not in graph_raw["router"]:
                self.router.merge_rewrite = True
            if "grade" not in graph_raw or "mode" not in graph_raw["grade"]:
                self.grade.mode = "threshold"
            if "verify" not in graph_raw or "mode" not in graph_raw["verify"]:
                self.verify.mode = "code"
            if "guardrail" not in graph_raw or "llm_enabled" not in graph_raw["guardrail"]:
                self.guardrail.llm_enabled = False
        elif mode == "full":
            if "grade" not in graph_raw or "mode" not in graph_raw["grade"]:
                self.grade.mode = "hybrid"
            if "verify" not in graph_raw or "mode" not in graph_raw["verify"]:
                self.verify.mode = "llm"


class ValidityConfig(BaseModel):
    enabled: bool = True
    expand_replacements: bool = False


class PrivacyConfig(BaseModel):
    allow_remote_for_uploads: bool = False


class UploadConfig(BaseModel):
    max_size_mb: int = 20
    max_pages: int = 150
    max_chunks: int = 800
    min_chars_per_page: int = 100
    ttl_hours: int = 24
    keep_original: bool = False


class APIConfig(BaseModel):
    rate_limit_per_minute: int = 30
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:8501"])


class LLMCacheConfig(BaseModel):
    enabled: bool = True
    path: str = "data/cache/llm_cache.sqlite"


class LLMConfig(BaseModel):
    profile: str = "api_dev"
    cache: LLMCacheConfig = Field(default_factory=LLMCacheConfig)


class PromptsConfig(BaseModel):
    version: str = "v1"


class AppConfig(BaseModel):
    snapshot_as_of: str = "2024-09-10"
    db: DBConfig = Field(default_factory=DBConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    filters: FiltersConfig = Field(default_factory=FiltersConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    validity: ValidityConfig = Field(default_factory=ValidityConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    upload: UploadConfig = Field(default_factory=UploadConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    prompts: PromptsConfig = Field(default_factory=PromptsConfig)


class RerankConfig(BaseModel):
    enabled: bool = True
    model: str = "AITeamVN/Vietnamese_Reranker"
    max_length: int = 1024
    batch_size: int = 8
    top_k: int = 8


class HybridConfig(BaseModel):
    enabled: bool = True


class ParentChildConfig(BaseModel):
    enabled: bool = True
    max_parents: int = 5


class SparseConfig(BaseModel):
    tokenizer: Literal["syllable_bigram", "syllable"] = "syllable_bigram"
    k1: float = 1.2
    b: float = 0.75


class RetrievalConfig(BaseModel):
    dense_top_k: int = 40
    sparse_top_k: int = 40
    rrf_k: int = 60
    fused_top_k: int = 30
    hybrid: HybridConfig = Field(default_factory=HybridConfig)
    rerank: RerankConfig = Field(default_factory=RerankConfig)
    parent_child: ParentChildConfig = Field(default_factory=ParentChildConfig)
    expired_policy: Literal["annotate", "penalize", "exclude"] = "annotate"
    status_penalty: float = 2.0
    context_token_budget: int = 4000
    sparse: SparseConfig = Field(default_factory=SparseConfig)


class DomainConfig(BaseModel):
    domain: str = "lao_dong"
    national_only: bool = True
    allowed_doc_types: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    seed_doc_numbers: list[str] = Field(default_factory=list)
    expand_hops: int = 1
    max_docs: int = 150


class ModelsConfig(BaseModel):
    profiles: dict[str, Any] = Field(default_factory=dict)
    embeddings: dict[str, Any] = Field(default_factory=dict)
    reranker: dict[str, Any] = Field(default_factory=dict)


class ConfigBundle(BaseModel):
    app: AppConfig
    retrieval: RetrievalConfig
    domain: DomainConfig
    models: ModelsConfig


def load_config(config_dir: str | Path = "configs") -> ConfigBundle:
    """Load full configuration bundle from config directory."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

    cfg_path = Path(config_dir)

    app_dict = load_yaml(cfg_path / "app.yaml")
    retrieval_dict = load_yaml(cfg_path / "retrieval.yaml")
    domain_dict = load_yaml(cfg_path / "domain.yaml")
    models_dict = load_yaml(cfg_path / "models.yaml")

    app_cfg = AppConfig(**app_dict)
    app_cfg.graph.apply_presets(app_dict)

    return ConfigBundle(
        app=app_cfg,
        retrieval=RetrievalConfig(**retrieval_dict),
        domain=DomainConfig(**domain_dict),
        models=ModelsConfig(**models_dict),
    )


def load_app_config(config_dir: str | Path = "configs") -> AppConfig:
    return load_config(config_dir).app


def load_retrieval_config(config_dir: str | Path = "configs") -> RetrievalConfig:
    return load_config(config_dir).retrieval


def load_models_config(config_dir: str | Path = "configs") -> ModelsConfig:
    return load_config(config_dir).models

